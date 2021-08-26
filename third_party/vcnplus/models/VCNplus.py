import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import pdb
import time
import cv2
import kornia

from .submodule import pspnet, bfmodule, bfmodule_feat, conv, compute_geo_costs, get_skew_mat, get_intrinsics, F_ngransac
from .conv4d import sepConv4d, butterfly4D

class flow_reg(nn.Module):
    """
    Soft winner-take-all that selects the most likely diplacement.
    Set ent=True to enable entropy output.
    Set maxdisp to adjust maximum allowed displacement towards one side.
        maxdisp=4 searches for a 9x9 region.
    Set fac to squeeze search window.
        maxdisp=4 and fac=2 gives search window of 9x5
    """
    def __init__(self, size, ent=False, maxdisp = int(4), fac=1):
        B,W,H = size
        super(flow_reg, self).__init__()
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3  # by default using truncation 7x7

        flowrangey = range(-maxdisp,maxdisp+1)
        flowrangex = range(-int(maxdisp//self.fac),int(maxdisp//self.fac)+1)
        meshgrid = np.meshgrid(flowrangex,flowrangey)
        flowy = np.tile( np.reshape(meshgrid[0],[1,2*maxdisp+1,2*int(maxdisp//self.fac)+1,1,1]), (B,1,1,H,W) )
        flowx = np.tile( np.reshape(meshgrid[1],[1,2*maxdisp+1,2*int(maxdisp//self.fac)+1,1,1]), (B,1,1,H,W) )
        self.register_buffer('flowx',torch.Tensor(flowx))
        self.register_buffer('flowy',torch.Tensor(flowy))

        self.pool3d = nn.MaxPool3d((self.wsize*2+1,self.wsize*2+1,1),stride=1,padding=(self.wsize,self.wsize,0))

    def forward(self, x):
        b,u,v,h,w = x.shape
        oldx = x

        if self.truncated:
            # truncated softmax
            x = x.view(b,u*v,h,w)

            idx = x.argmax(1)[:,np.newaxis]
            if x.is_cuda:
                mask = Variable(torch.cuda.HalfTensor(b,u*v,h,w)).fill_(0)
            else:
                mask = Variable(torch.FloatTensor(b,u*v,h,w)).fill_(0)
            mask.scatter_(1,idx,1)
            mask = mask.view(b,1,u,v,-1)
            mask = self.pool3d(mask)[:,0].view(b,u,v,h,w)

            ninf = x.clone().fill_(-np.inf).view(b,u,v,h,w)
            x = torch.where(mask.byte(),oldx,ninf)
        else:
            self.wsize = (np.sqrt(u*v)-1)/2

        b,u,v,h,w = x.shape
        x = F.softmax(x.view(b,-1,h,w),1).view(b,u,v,h,w)
        if np.isnan(x.min().detach().cpu()):
            #pdb.set_trace()
            x[torch.isnan(x)] = F.softmax(oldx[torch.isnan(x)])
        outx = torch.sum(torch.sum(x*self.flowx,1),1,keepdim=True)
        outy = torch.sum(torch.sum(x*self.flowy,1),1,keepdim=True)

        if self.ent:
            # local
            local_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.
            else:
                local_entropy /= np.log((self.wsize*2+1)**2)

            # global
            x = F.softmax(oldx.view(b,-1,h,w),1).view(b,u,v,h,w)
            global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            global_entropy /= np.log(x.shape[1]*x.shape[2])
            return torch.cat([outx,outy],1),torch.cat([local_entropy, global_entropy],1)
        else:
            return torch.cat([outx,outy],1),None


class WarpModule(nn.Module):
    """
    taken from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    """
    def __init__(self, size):
        super(WarpModule, self).__init__()
        B,W,H = size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        self.register_buffer('grid',torch.cat((xx,yy),1).float())

    def forward(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        vgrid = self.grid + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        #output = nn.functional.grid_sample(x, vgrid)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = ((vgrid[:,:,:,0].abs()<1) * (vgrid[:,:,:,1].abs()<1)) >0
        return output*mask.unsqueeze(1).float(), mask


def get_grid(B,H,W):
    meshgrid_base = np.meshgrid(range(0,W), range(0,H))[::-1]
    basey = np.reshape(meshgrid_base[0],[1,1,1,H,W])
    basex = np.reshape(meshgrid_base[1],[1,1,1,H,W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1,H,W,1)),basey.reshape((-1,H,W,1))),-1)).cuda().float()
    return grid.view(1,1,H,W,2)


class VCN(nn.Module):
    """
    VCN.
    md defines maximum displacement for each level, following a coarse-to-fine-warping scheme
    fac defines squeeze parameter for the coarsest level
    """
    def __init__(self, size, md=[4,4,4,4,4], fac=1., exp_unc=True):
        super(VCN,self).__init__()
        self.md = md
        self.fac = fac
        use_entropy = True
        withbn = True

        ## pspnet
        self.pspnet = pspnet(is_proj=False)

        ### Volumetric-UNet
        fdima1 = 128 # 6/5/4
        fdima2 = 64 # 3/2
        fdimb1 = 16 # 6/5/4/3
        fdimb2 = 12 # 2

        full=False
        self.f6 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p6 = sepConv4d(fdimb1,fdimb1, with_bn=False, full=full)

        self.f5 = butterfly4D(fdima1, fdimb1,withbn=withbn, full=full)
        self.p5 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f4 = butterfly4D(fdima1, fdimb1,withbn=withbn,full=full)
        self.p4 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        self.f3 = butterfly4D(fdima2, fdimb1,withbn=withbn,full=full)
        self.p3 = sepConv4d(fdimb1,fdimb1, with_bn=False,full=full)

        full=True
        self.f2 = butterfly4D(fdima2, fdimb2,withbn=withbn,full=full)
        self.p2 = sepConv4d(fdimb2,fdimb2, with_bn=False,full=full)
    
        self.flow_reg64 = flow_reg([fdimb1*size[0],size[1]//64,size[2]//64], ent=use_entropy, maxdisp=self.md[0], fac=self.fac)
        self.flow_reg32 = flow_reg([fdimb1*size[0],size[1]//32,size[2]//32], ent=use_entropy, maxdisp=self.md[1])
        self.flow_reg16 = flow_reg([fdimb1*size[0],size[1]//16,size[2]//16], ent=use_entropy, maxdisp=self.md[2])
        self.flow_reg8 =  flow_reg([fdimb1*size[0],size[1]//8,size[2]//8]  , ent=use_entropy, maxdisp=self.md[3])
        self.flow_reg4 =  flow_reg([fdimb2*size[0],size[1]//4,size[2]//4]  , ent=use_entropy, maxdisp=self.md[4])

        self.warp5 = WarpModule([size[0],size[1]//32,size[2]//32])
        self.warp4 = WarpModule([size[0],size[1]//16,size[2]//16])
        self.warp3 = WarpModule([size[0],size[1]//8,size[2]//8])
        self.warp2 = WarpModule([size[0],size[1]//4,size[2]//4])
        if self.training:
            self.warpx = WarpModule([size[0],size[1],size[2]])

        ## hypotheses fusion modules, adopted from the refinement module of PWCNet
        # https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
        # c6
        self.dc6_conv1 = conv(128+4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc6_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc6_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc6_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc6_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc6_conv7 = nn.Conv2d(32,2*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)

        # c5
        self.dc5_conv1 = conv(128+4*fdimb1*2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc5_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc5_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc5_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc5_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc5_conv7 = nn.Conv2d(32,2*fdimb1*2,kernel_size=3,stride=1,padding=1,bias=True)

        # c4
        self.dc4_conv1 = conv(128+4*fdimb1*3, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc4_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc4_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc4_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc4_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc4_conv7 = nn.Conv2d(32,2*fdimb1*3,kernel_size=3,stride=1,padding=1,bias=True)

        # c3
        self.dc3_conv1 = conv(64+16*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc3_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc3_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc3_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc3_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc3_conv7 = nn.Conv2d(32,8*fdimb1,kernel_size=3,stride=1,padding=1,bias=True)

        # c2
        self.dc2_conv1 = conv(64+16*fdimb1+4*fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc2_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc2_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc2_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc2_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc2_conv7 = nn.Conv2d(32,4*2*fdimb1 + 2*fdimb2,kernel_size=3,stride=1,padding=1,bias=True)

        self.dc6_conv = nn.Sequential(  self.dc6_conv1,
                                        self.dc6_conv2,
                                        self.dc6_conv3,
                                        self.dc6_conv4,
                                        self.dc6_conv5,
                                        self.dc6_conv6,
                                        self.dc6_conv7)
        self.dc5_conv = nn.Sequential(  self.dc5_conv1,
                                        self.dc5_conv2,
                                        self.dc5_conv3,
                                        self.dc5_conv4,
                                        self.dc5_conv5,
                                        self.dc5_conv6,
                                        self.dc5_conv7)
        self.dc4_conv = nn.Sequential(  self.dc4_conv1,
                                        self.dc4_conv2,
                                        self.dc4_conv3,
                                        self.dc4_conv4,
                                        self.dc4_conv5,
                                        self.dc4_conv6,
                                        self.dc4_conv7)
        self.dc3_conv = nn.Sequential(  self.dc3_conv1,
                                        self.dc3_conv2,
                                        self.dc3_conv3,
                                        self.dc3_conv4,
                                        self.dc3_conv5,
                                        self.dc3_conv6,
                                        self.dc3_conv7)
        self.dc2_conv = nn.Sequential(  self.dc2_conv1,
                                        self.dc2_conv2,
                                        self.dc2_conv3,
                                        self.dc2_conv4,
                                        self.dc2_conv5,
                                        self.dc2_conv6,
                                        self.dc2_conv7)

        ## Out-of-range detection
        self.dc6_convo = nn.Sequential(conv(128+4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc5_convo = nn.Sequential(conv(128+2*4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc4_convo = nn.Sequential(conv(128+3*4*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc3_convo = nn.Sequential(conv(64+16*fdimb1, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        self.dc2_convo = nn.Sequential(conv(64+16*fdimb1+4*fdimb2, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                            conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2),
                            conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4),
                            conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8),
                            conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16),
                            conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1),
                            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1,bias=True))

        # affine-exp
        self.f3d2v1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2v2 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2v3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2v4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2v5 = conv(64,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2v6 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.f3d2 = bfmodule(128-64,1)

        # depth change net
        self.dcnetv1 = conv(64, 32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.dcnetv2 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.dcnetv3 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.dcnetv4 = conv(1,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.dcnetv5 = conv(12*81,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.dcnetv6 = conv(4,   32, kernel_size=3, stride=1, padding=1,dilation=1) # 
        if exp_unc:
            self.dcnet = bfmodule(128,2)
        else:
            self.dcnet = bfmodule(128,1)
            
        # moseg net
        self.fgnetv1 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv2 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv3 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv4 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv5 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv6 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv7 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv8 = conv(1,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv9 = conv(3,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnetv10 = conv(3,   16, kernel_size=3, stride=1, padding=1,dilation=1) # 
        self.fgnet = bfmodule_feat(208-3*16,7)

        #from midas.midas_net import MidasNet
        #self.midas = MidasNet('/data/gengshay/midas.pt', non_negative=True)
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        
        # detection branch
        from models.det import create_model, load_model, save_model
        self.det = create_model('dla_34', {'hm': 2, 'wh': 36}, 256,num_input=14)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

        self.facs = [self.fac,1,1,1,1]
        self.warp_modules = nn.ModuleList([None, self.warp5, self.warp4, self.warp3, self.warp2])
        self.f_modules = nn.ModuleList([self.f6, self.f5, self.f4, self.f3, self.f2])
        self.p_modules = nn.ModuleList([self.p6, self.p5, self.p4, self.p3, self.p2])
        self.reg_modules = nn.ModuleList([self.flow_reg64, self.flow_reg32, self.flow_reg16, self.flow_reg8, self.flow_reg4])
        self.oor_modules = nn.ModuleList([self.dc6_convo, self.dc5_convo, self.dc4_convo, self.dc3_convo, self.dc2_convo])
        self.fuse_modules = nn.ModuleList([self.dc6_conv, self.dc5_conv, self.dc4_conv, self.dc3_conv, self.dc2_conv])
  
    def corrf(self, refimg_fea, targetimg_fea,maxdisp, fac=1):
        if self.training:
            #fast correlation function
            b,c,h,w = refimg_fea.shape
            targetimg_fea = F.unfold(targetimg_fea, (2*int(maxdisp)//fac+1,2*maxdisp+1), padding=(int(maxdisp)//fac,maxdisp)).view(b,c, 2*int(maxdisp)//fac+1,2*maxdisp+1,h,w).permute(0,1,3,2,4,5).contiguous()
            cost = refimg_fea.view(b,c,h,w)[:,:,np.newaxis, np.newaxis]*targetimg_fea
            cost = F.leaky_relu(cost, 0.1,inplace=True)
        else:
            #slow correlation function
            b,c,height,width = refimg_fea.shape
            if refimg_fea.is_cuda:
                cost = Variable(torch.cuda.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//fac)+1,height,width)).fill_(0.) # b,c,u,v,h,w
            else:
                cost = Variable(torch.FloatTensor(b,c,2*maxdisp+1,2*int(maxdisp//fac)+1,height,width)).fill_(0.) # b,c,u,v,h,w
            for i in range(2*maxdisp+1):
                ind = i-maxdisp
                for j in range(2*int(maxdisp//fac)+1):
                    indd = j-int(maxdisp//fac)
                    feata = refimg_fea[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                    featb = targetimg_fea[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
                    diff = (feata*featb)
                    cost[:, :, i,j,max(0,-indd):height-indd,max(0,-ind):width-ind]   = diff  # standard
            cost = F.leaky_relu(cost, 0.1,inplace=True)
        return cost

    def cost_matching(self,up_flow, c1, c2, flowh, enth, level):
        """
        up_flow: upsample coarse flow
        c1: normalized feature of image 1
        c2: normalized feature of image 2
        flowh: flow hypotheses
        enth: entropy
        """

        # normalize
        c1n = c1 / (c1.norm(dim=1, keepdim=True)+1e-9)
        c2n = c2 / (c2.norm(dim=1, keepdim=True)+1e-9)

        # cost volume
        if level == 0:
            warp = c2n
        else:
            warp,_ = self.warp_modules[level](c2n, up_flow)

        feat = self.corrf(c1n,warp,self.md[level],fac=self.facs[level])
        feat = self.f_modules[level](feat) 
        cost = self.p_modules[level](feat) # b, 16, u,v,h,w

        # soft WTA
        b,c,u,v,h,w = cost.shape
        cost = cost.view(-1,u,v,h,w)  # bx16, 9,9,h,w, also predict uncertainty from here
        flowhh,enthh = self.reg_modules[level](cost) # bx16, 2, h, w
        flowhh = flowhh.view(b,c,2,h,w)
        if level > 0:
            flowhh = flowhh + up_flow[:,np.newaxis]
        flowhh = flowhh.view(b,-1,h,w) # b, 16*2, h, w
        enthh =  enthh.view(b,-1,h,w) # b, 16*1, h, w

        # append coarse hypotheses
        if level == 0:
            flowh = flowhh
            enth = enthh
        else:
            flowh = torch.cat((flowhh, F.upsample(flowh.detach()*2, [flowhh.shape[2],flowhh.shape[3]], mode='bilinear')),1) # b, k2--k2, h, w
            enth = torch.cat((enthh, F.upsample(enth, [flowhh.shape[2],flowhh.shape[3]], mode='bilinear')),1)

        if self.training or level==4:
            x = torch.cat((enth.detach(), flowh.detach(), c1),1)
            oor = self.oor_modules[level](x)[:,0]
        else: oor = None

        # hypotheses fusion
        x = torch.cat((enth.detach(), flowh.detach(), c1),1)
        va = self.fuse_modules[level](x)
        va = va.view(b,-1,2,h,w)
        flow = ( flowh.view(b,-1,2,h,w) * F.softmax(va,1) ).sum(1) # b, 2k, 2, h, w

        return flow, flowh, enth, oor

    def affine(self,pref,flow, pw=1):
        b,_,lh,lw=flow.shape
        ptar = pref + flow
        pw = 1
        pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
        ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
        pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
        ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

        prefprefT = pref.matmul(pref.permute(0,2,1))
        ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
        ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

        Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
        Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

        Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
        exp = Avol.sqrt()
        mask = (exp>0.5) & (exp<2) & (Error<0.1)
        mask = mask[:,0]

        exp = exp.clamp(0.5,2)
        exp[Error>0.1]=1
        return exp, Error, mask

    def affine_mask(self,pref,flow, pw=3):
        """
        pref: reference coordinates
        pw: patch width
        """
        flmask = flow[:,2:]
        flow = flow[:,:2]
        b,_,lh,lw=flow.shape
        ptar = pref + flow
        pref = F.unfold(pref, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-pref[:,:,np.newaxis]
        ptar = F.unfold(ptar, (pw*2+1,pw*2+1), padding=(pw)).view(b,2,(pw*2+1)**2,lh,lw)-ptar[:,:,np.newaxis] # b, 2,9,h,w
        
        conf_flow = flmask
        conf_flow = F.unfold(conf_flow,(pw*2+1,pw*2+1), padding=(pw)).view(b,1,(pw*2+1)**2,lh,lw)
        count = conf_flow.sum(2,keepdims=True)
        conf_flow = ((pw*2+1)**2)*conf_flow / count
        pref = pref * conf_flow
        ptar = ptar * conf_flow
    
        pref = pref.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)
        ptar = ptar.permute(0,3,4,1,2).reshape(b*lh*lw,2,(pw*2+1)**2)

        prefprefT = pref.matmul(pref.permute(0,2,1))
        ppdet = prefprefT[:,0,0]*prefprefT[:,1,1]-prefprefT[:,1,0]*prefprefT[:,0,1]
        ppinv = torch.cat((prefprefT[:,1,1:],-prefprefT[:,0,1:], -prefprefT[:,1:,0], prefprefT[:,0:1,0]),1).view(-1,2,2)/ppdet.clamp(1e-10,np.inf)[:,np.newaxis,np.newaxis]

        Affine = ptar.matmul(pref.permute(0,2,1)).matmul(ppinv)
        Error = (Affine.matmul(pref)-ptar).norm(2,1).mean(1).view(b,1,lh,lw)

        Avol = (Affine[:,0,0]*Affine[:,1,1]-Affine[:,1,0]*Affine[:,0,1]).view(b,1,lh,lw).abs().clamp(1e-10,np.inf)
        exp = Avol.sqrt()
        mask = (exp>0.5) & (exp<2) & (Error<0.2) & (flmask.bool()) & (count[:,0]>4)
        mask = mask[:,0]

        exp = exp.clamp(0.5,2)
        exp[Error>0.2]=1
        return exp, Error, mask

    def get_oor_loss(self, flowl0, oor3, maxdisp, occ_mask,mask):
        """ 
        return out-of-range loss
        """
        oor3_gt = (flowl0.abs() > maxdisp).detach() #  (8*self.md[3])
        oor3_gt = (((oor3_gt.sum(1)>0) + occ_mask)>0).float()  # oor, or occluded

        #weights = oor3_gt.sum().float()/(oor3_gt.shape[0]*oor3_gt.shape[1]*oor3_gt.shape[2])
        oor3_gt = oor3_gt[mask]
        weights = oor3_gt.sum().float()/(oor3_gt.shape[0])

        weights = oor3_gt * (1-weights) + (1-oor3_gt) * weights
        loss_oor3 = F.binary_cross_entropy_with_logits(oor3[mask],oor3_gt,size_average=True, weight=weights)
        return loss_oor3

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self,im,disc_aux=None,disp_input=None):
        bs = im.shape[0]//2

        if self.training and disc_aux[-1]: # if only fine-tuning expansion 
            reset=True
            self.eval()
            torch.set_grad_enabled(False)
        else: reset=False

        c06,c05,c04,c03,c02 = self.pspnet(im)
        c16 = c06[:bs];  c26 = c06[bs:]
        c15 = c05[:bs];  c25 = c05[bs:]
        c14 = c04[:bs];  c24 = c04[bs:]
        c13 = c03[:bs];  c23 = c03[bs:]
        c12 = c02[:bs];  c22 = c02[bs:]

        ## matching 6
        flow6, flow6h, ent6h, oor6 = self.cost_matching(None, c16, c26, None, None,level=0)

        ## matching 5
        up_flow6 = F.upsample(flow6, [im.size()[2]//32,im.size()[3]//32], mode='bilinear')*2
        flow5, flow5h, ent5h, oor5 = self.cost_matching(up_flow6, c15, c25, flow6h, ent6h,level=1)

        ## matching 4
        up_flow5 = F.upsample(flow5, [im.size()[2]//16,im.size()[3]//16], mode='bilinear')*2
        flow4, flow4h, ent4h, oor4 = self.cost_matching(up_flow5, c14, c24, flow5h, ent5h,level=2)

        ## matching 3
        up_flow4 = F.upsample(flow4, [im.size()[2]//8,im.size()[3]//8], mode='bilinear')*2
        flow3, flow3h, ent3h, oor3 = self.cost_matching(up_flow4, c13, c23, flow4h, ent4h,level=3)

        ## matching 2
        up_flow3 = F.upsample(flow3, [im.size()[2]//4,im.size()[3]//4], mode='bilinear')*2
        flow2, flow2h, ent2h, oor2 = self.cost_matching(up_flow3, c12, c22, flow3h, ent3h,level=4)

        if reset and disc_aux[-1] == 1: 
            torch.set_grad_enabled(True)          
            self.train()

        if not self.training or disc_aux[-1]:
            # expansion
            b,_,h,w = flow2.shape 
            exp2,err2,_ = self.affine(get_grid(b,h,w)[:,0].permute(0,3,1,2).repeat(b,1,1,1).clone(), flow2.detach(),pw=1)
            x = torch.cat((
                           self.f3d2v2(-exp2.log()),
                           self.f3d2v3(err2),
                           ),1)
            dchange2 = -exp2.log()+1./200*self.f3d2(x)[0]
    
            # depth change net
            iexp2 = F.upsample(dchange2.clone(), [im.size()[2],im.size()[3]], mode='bilinear')
            x = torch.cat((self.dcnetv1(c12.detach()),
                           self.dcnetv2(dchange2.detach()),
                           self.dcnetv3(-exp2.log()),
                           self.dcnetv4(err2),
                        ),1)
            dcneto = 1./200*self.dcnet(x)[0]
            dchange2 = dchange2.detach() + dcneto[:,:1]
            dchange2 = F.upsample(dchange2, [im.size()[2],im.size()[3]], mode='bilinear')

            if dcneto.shape[1]>1:
                dc_unc = dcneto[:,1:2]
            else:
                dc_unc = torch.zeros_like(dcneto)
            dc_unc = F.upsample(dc_unc, [im.size()[2],im.size()[3]], mode='bilinear')[:,0]

        if reset and disc_aux[-1] == 2: 
            torch.set_grad_enabled(True)          
            self.train()

        if not self.training or disc_aux[-1]==2:
            # segmentation
            Kinv, Kinv_n = get_intrinsics(disc_aux[3], noise=False)
            # full res flow/expansion/depth inputs
            H,W = im.size()[2:4]
            flow = 4*F.upsample(flow2, [H,W], mode='bilinear').detach()
            oor2 = F.upsample(oor2[:,np.newaxis], [H,W], mode='bilinear').detach()[:,0]
            tau = (-dchange2[:,0]).exp().detach()
            #tau[:]=1
            if self.training:
                fscale=1./4; fscalex=1./8;fscaled=1./1
            else:
                fscale=128./H; fscalex=32./H;fscaled=448./H
            hp0o = torch.cat( [torch.arange(0, W,out=torch.cuda.FloatTensor()).view(1,-1).repeat(H,1)[np.newaxis],  # 1,2,H,W
                              torch.arange(0, H,out=torch.cuda.FloatTensor()).view(-1,1).repeat(1,W)[np.newaxis]], 0)[np.newaxis]
            hp1o = hp0o + flow  # b,2,H,W
            if not self.training:
                hp0o[:,0] *= disc_aux[3][10]
                hp0o[:,1] *= disc_aux[3][11]
                hp1o[:,0] *= disc_aux[3][10]
                hp1o[:,1] *= disc_aux[3][11]

            # det res: fscaled
            hp0d = F.interpolate(hp0o,scale_factor=fscaled,mode='nearest')
            hp1d = F.interpolate(hp1o,scale_factor=fscaled,mode='nearest')
            _,_,hd,wd=hp0d.shape
            hp0d = hp0d.view(1,2,-1).permute(0,2,1)
            hp1d = hp1d.view(bs,2,-1).permute(0,2,1)
            hp0d = torch.cat((hp0d,torch.ones(1,hp0d.shape[1],1).cuda()),-1)
            hp1d = torch.cat((hp1d,torch.ones(bs,hp0d.shape[1],1).cuda()),-1)
            uncd = torch.cat((F.interpolate(oor2[:,np.newaxis],scale_factor=fscaled,mode='nearest'),
                  F.interpolate(dc_unc[:,np.newaxis].detach(),scale_factor=fscaled,mode='nearest')),1)
            taud = F.interpolate(tau[:,np.newaxis],scale_factor=fscaled,mode='nearest').view(bs,1,-1)

            # fg/bg res: fscale
            hp0 = F.interpolate(hp0o,scale_factor=fscale,mode='nearest')
            hp1 = F.interpolate(hp1o,scale_factor=fscale,mode='nearest')
            _,_,h,w=hp0.shape
            hp0 = hp0.view(1,2,-1).permute(0,2,1)
            hp1 = hp1.view(bs,2,-1).permute(0,2,1)
            hp0 = torch.cat((hp0,torch.ones(1,hp0.shape[1],1).cuda()),-1)
            hp1 = torch.cat((hp1,torch.ones(bs,hp0.shape[1],1).cuda()),-1)
            unc = torch.cat((F.interpolate(oor2[:,np.newaxis],scale_factor=fscale,mode='nearest'),
                  F.interpolate(dc_unc[:,np.newaxis].detach(),scale_factor=fscale,mode='nearest')),1)
            tau = F.interpolate(tau[:,np.newaxis],scale_factor=fscale,mode='nearest').view(bs,1,-1)
            
            # cam res: fscalex
            hp0x = F.interpolate(hp0o,scale_factor=fscalex,mode='nearest')
            hp1x = F.interpolate(hp1o,scale_factor=fscalex,mode='nearest')
            hp0x = hp0x.view(1,2,-1).permute(0,2,1)
            hp1x = hp1x.view(bs,2,-1).permute(0,2,1)
            hp0x = torch.cat((hp0x,torch.ones(1,hp0x.shape[1],1).cuda()),-1)
            hp1x = torch.cat((hp1x,torch.ones(bs,hp0x.shape[1],1).cuda()),-1)

            # rotation estimation
            if self.training:
                rot = disc_aux[-2][:,:3].detach()
                trans = disc_aux[-2][:,3:].detach()
                rot = rot + torch.Tensor(np.random.normal(loc=0.,scale=5e-4,size=(bs,3))).cuda()
                trans = trans + torch.Tensor(np.random.normal(loc=0.,scale=5e-2,size=(bs,3))).cuda() * trans.norm(2,1)[:,np.newaxis]
                trans = trans/trans.norm(2,1)[:,np.newaxis]
                Ex = get_skew_mat(trans.cpu(),rot.cpu())
            else:
                rand=False
                unc_occ = F.interpolate(oor2[:,np.newaxis],scale_factor=fscalex,mode='nearest').view(bs,-1)
                rotx,transx,Ex = F_ngransac(hp0x,hp1x,Kinv.inverse(),rand,unc_occ, Kn = Kinv_n.inverse(),cv=False)
                rot = rotx.cuda().detach()
                trans = transx.cuda().detach()

            # cost compute
            mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag,_ = compute_geo_costs(rot, trans, Ex, Kinv, hp0, hp1, tau, Kinv_n = Kinv_n)
            if disp_input is None:
                with torch.no_grad():
                    self.midas.eval()
                    input_im  = (disc_aux[4].permute(0,3,1,2) -\
                            torch.Tensor([0.485, 0.456, 0.406]).cuda()[np.newaxis,:,np.newaxis,np.newaxis]) /\
                            torch.Tensor([0.229, 0.224, 0.225]).cuda()[np.newaxis,:,np.newaxis,np.newaxis]
                    wsize = int((input_im.shape[3] * 448./input_im.shape[2])//32*32)
                    input_im = F.interpolate(input_im, (448, wsize), mode='bilinear')
                    #dispo = self.midas.forward(input_im)[0][None].clamp(1e-6,np.inf)
                    dispo = self.midas.forward(input_im)[None].clamp(1e-6,np.inf)
            else:
                dispo = disp_input
#            dispo[:]=1

            disp = F.interpolate(dispo, [h,w], mode='bilinear')
            med_dgt = torch.median(disp.view(bs,-1),dim=-1)[0]
            med_dp3d = torch.median(p3dmag.view(bs,-1),dim=-1)[0]
            med_ratio = (med_dgt/med_dp3d)[:,np.newaxis,np.newaxis,np.newaxis]
            log_dratio = ( med_ratio * p3dmag.view(bs,1,h,w) / disp.view(bs,1,h,w) ).log()
            #pdb.set_trace()
            #cv2.imwrite('/data/gengshay/a0.png',np.asarray(disp[0,0].cpu()/1000))
            #cv2.imwrite('/data/gengshay/a1.png',np.asarray(log_dratio[0,0].exp().cpu()*100))

            # pseudo 3D point compute
            #depth = disc_aux[2][:,:,:,0].view(bs,1,-1)
            depth = (1./ disp).view(bs,1,-1)
            depth = depth.clamp(depth.median()/10, depth.median()*10)
            #depth = depth.log()
            p03d = depth *      Kinv.matmul(hp0.permute(0,2,1))
            p13d = depth/tau*Kinv_n.matmul(hp1.permute(0,2,1))
            p13d = kornia.angle_axis_to_rotation_matrix(rot).matmul(p13d)  # remove rotation
            pts = torch.cat([p03d, p13d],-1) # bs, 3, 2*N
            # normalize it 
            for i in range(bs):
                pts[i] = pts[i] - pts[i].mean(-1,keepdims=True)  # zero mean
                pts[i] = pts[i] / pts[i].flatten().std() # unit std
            p03d = pts[:,:,:p03d.shape[-1]]
            p13d = pts[:,:,p03d.shape[-1]:]

            # fg/bg segmentation network
            costs = torch.cat((
                           self.fgnetv1( 0.01*(mcost00+mcost01).view(bs,1,h,w).detach()),
                           self.fgnetv2( 2e3*       mcost1.view(bs,1,h,w).detach()),
                           self.fgnetv3(            mcost2.view(bs,1,h,w).detach()),
                           self.fgnetv4(   30*      mcost3.view(bs,1,h,w).detach()),
                           self.fgnetv5(            mcost4.view(bs,1,h,w).detach()),
                           self.fgnetv6(  0.2*      unc[:,:1].view(bs,1,h,w).detach()),
                           self.fgnetv7(  0.2*      unc[:,1:].view(bs,1,h,w).detach()),
                           self.fgnetv8(   3*      log_dratio.view(bs,1,h,w).detach()),
                           self.fgnetv9( p03d.view(bs,3,h,w).detach()),
                          self.fgnetv10( p13d.view(bs,3,h,w).detach()),
                        ),1)
            x,featx = self.fgnet(costs)
            fg_va =  1./20*x[:,:-1]
            fg_res = 1./200*x[:,-1:]
            fg_hps = torch.cat( (
                                  0.01*(mcost00+mcost01).view(bs,1,h,w).detach(),
                                   2e3* mcost1.view(bs,1,h,w).detach(),
                                        mcost2.view(bs,1,h,w).detach(),
                                   30*  mcost3.view(bs,1,h,w).detach(),
                                        mcost4.view(bs,1,h,w).detach(),
                            3*      log_dratio.view(bs,1,h,w).detach(),
                                ),1)
            fgmask = (fg_va * fg_hps).sum(1, keepdims=True) + fg_res
 #           fgmask = 1./200*x[:,-1:]
            fgmask = F.upsample(fgmask, [im.size()[2],im.size()[3]], mode='bilinear')

            # detection branch
            # cost compute
            mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag,_ = compute_geo_costs(rot, trans, Ex, Kinv, hp0d, hp1d, taud, Kinv_n = Kinv_n)
            disp = F.interpolate(dispo, [hd,wd], mode='bilinear')
            med_dgt = torch.median(disp.view(bs,-1),dim=-1)[0]
            med_dp3d = torch.median(p3dmag.view(bs,-1),dim=-1)[0]
            med_ratio = (med_dgt/med_dp3d)[:,np.newaxis,np.newaxis,np.newaxis]
            log_dratio = ( med_ratio * p3dmag.view(bs,1,hd,wd) / disp.view(bs,1,hd,wd) ).log()
            # pseudo 3D point compute
            #depth = disc_aux[2][:,:,:,0].view(bs,1,-1)
            depth = (1./ disp).view(bs,1,-1)
            depth = depth.clamp(depth.median()/10, depth.median()*10)
            #depth = depth.log()
            p03d = depth *      Kinv.matmul(hp0d.permute(0,2,1))
            p13d = depth/taud*Kinv_n.matmul(hp1d.permute(0,2,1))
            p13d = kornia.angle_axis_to_rotation_matrix(rot).matmul(p13d)  # remove rotation
            pts = torch.cat([p03d, p13d],-1) # bs, 3, 2*N
            # normalize it 
            for i in range(bs):
                pts[i] = pts[i] - pts[i].mean(-1,keepdims=True)  # zero mean
                pts[i] = pts[i] / pts[i].flatten().std() # unit std
            p03d = pts[:,:,:p03d.shape[-1]]
            p13d = pts[:,:,p03d.shape[-1]:]

            costs = torch.cat((
                            0.01*(mcost00+mcost01).view(bs,1,hd,wd).detach(),
                                 2e3*       mcost1.view(bs,1,hd,wd).detach(),
                                            mcost2.view(bs,1,hd,wd).detach(),
                                   30*      mcost3.view(bs,1,hd,wd).detach(),
                                            mcost4.view(bs,1,hd,wd).detach(),
                               0.2*      uncd[:,:1].view(bs,1,hd,wd).detach(),
                               0.2*      uncd[:,1:].view(bs,1,hd,wd).detach(),
                                3*      log_dratio.view(bs,1,hd,wd).detach(),
                                p03d.view(bs,3,hd,wd).detach(),
                                p13d.view(bs,3,hd,wd).detach(),
                        ),1)
            outputs = self.det(F.interpolate(costs, im.shape[2:],mode='bilinear' ))[0]
            heatmap_logits = (F.interpolate(outputs['hm'] ,im.shape[2:],mode='bilinear'))
            heatmap = heatmap_logits.softmax(1)[:,:1]
            pdist=(40*F.interpolate(outputs['wh'] ,im.shape[2:],mode='bilinear'))
            
            if not self.training:
                import os, sys
                currentdir = os.path.dirname(os.path.realpath(__file__))
                parentdir = os.path.dirname(currentdir)
                sys.path.append(parentdir)
                from flowutils.detlib import ctdet_decode
                ### angular
                #heatmap[:] = 0
                #fgmask = F.interpolate(0.5+-mcost2.view(1,1,hd,wd), (H,W),mode='bilinear')

                p03d = F.interpolate(p03d.view(1,3,hd,wd),im.shape[2:],mode='bilinear')
                polarmask = ctdet_decode(heatmap, pdist,p03d=p03d)

        if self.training:
            if disc_aux[-1]==1:
                # expansion
                flowl0 = disc_aux[0].permute(0,3,1,2).clone()
                gt_depth = disc_aux[2][:,:,:,0]
                gt_f3d =  disc_aux[2][:,:,:,4:7].permute(0,3,1,2).clone()
                gt_dchange = (1+gt_f3d[:,2]/gt_depth)
                maskdc = (gt_dchange < 2) & (gt_dchange > 0.5) & disc_aux[1]

                gt_expi,gt_expi_err,maskoe = self.affine_mask(get_grid(b,4*h,4*w)[:,0].permute(0,3,1,2).repeat(b,1,1,1), flowl0,pw=3)
                gt_exp = 1./gt_expi[:,0]

                loss =  0.1* (dchange2[:,0]-gt_dchange.log()).abs()[maskdc].mean()
                loss += 0.1* (iexp2[:,0]-gt_exp.log()).abs()[maskoe].mean()

                unc_loss = dc_unc[maskdc] + (dchange2[:,0].detach()-gt_dchange.log()).pow(2)[maskdc] / dc_unc.exp()[maskdc]
                loss += (1.8379 + unc_loss.mean())*0.001

                return flow2*4, flow3*8,flow4*16,flow5*32,flow6*64,loss, dchange2[:,0], iexp2[:,0]
            elif disc_aux[-1]==2:
                # mask
                valid_mask = (disc_aux[2][:,:,:,0]<100) & disc_aux[1]  # depth<100 and valid for flow
                Tglobal_gt = -disc_aux[-2][:,3:,np.newaxis,np.newaxis] # bg translation
                Tlocal_gt = disc_aux[2][:,:,:,1:4].permute(0,3,1,2)  # pixel translation (after rot)
                m3d_gt = (Tlocal_gt-Tglobal_gt).norm(2,1)           # abs. motion
                for i in range(bs):
                    if Tglobal_gt[i].norm().abs()==1: # kitti
                        m3d_gt[i] = (disc_aux[2][i,:,:,7]>0).float()
                fgmask_gt = (m3d_gt*100>1)[valid_mask].float()

                weights = fgmask_gt.sum().float()/(fgmask_gt.flatten().size()[0])
                weights = fgmask_gt * (1-weights) + (1-fgmask_gt) * weights
                loss = 0.01*F.binary_cross_entropy_with_logits(fgmask[:,0][valid_mask],fgmask_gt,size_average=True, weight=weights)
                
                # detection
                # rotation + translation
                gt_depth = disc_aux[2][:,:,:,0].view(bs,1,H,W)
                gt_p03d = (gt_depth.view(bs,1,-1) * Kinv.matmul(hp0d.permute(0,2,1))).view(bs,3,H,W)
                gt_f3d =  disc_aux[2][:,:,:,4:7].permute(0,3,1,2).clone()
                gt_p13d = gt_p03d + gt_f3d
                # option 2
                #flowl0 = disc_aux[0].permute(0,3,1,2)[:,:2]
                #gt_dchange = (1+gt_f3d[:,2]/gt_depth.view(bs,H,W)).view(bs,1,-1)
                #hp1o = (hp0o + flowl0).view(bs,2,-1)  # b,2,H,W
                #hp1o = torch.cat((hp1o,torch.ones(bs,1,hp1o.shape[2]).cuda()),1)
                #gt_p13d = gt_depth*gt_dchange*Kinv_n.matmul(hp1o)
#                pdb.set_trace()  # here, flow is sparse!
#                pts = torch.cat([gt_p03d.view(bs,3,-1), gt_p13d.view(bs,3,-1)],-1) # bs, 3, 2*N
#                # normalize it 
#                for i in range(bs):
#                    pts[i] = pts[i] - pts[i].mean(-1,keepdims=True)  # zero mean
#                    pts[i] = pts[i] / pts[i].flatten().std() # unit std
#                gt_p03d = pts[:,:,:gt_p03d.shape[-1]*gt_p03d.shape[-2]].view(bs,3,gt_p03d.shape[2],gt_p03d.shape[3])
#                gt_p13d = pts[:,:,gt_p03d.shape[-1]*gt_p03d.shape[-2]:].view(bs,3,gt_p03d.shape[2],gt_p03d.shape[3])


                from models.det_losses import FocalLoss
                from utils.detlib import distance2mask,draw_umich_gaussian, gaussian_radius, get_polarmask, polar_reg, pose_reg
                obj_idx = disc_aux[2][:,:,:,7] * ( (m3d_gt*100>1) & (disc_aux[2][:,:,:,0]<100) ).float()  # remove background 
                obj_idx = obj_idx.round().int()
                max_obj = obj_idx.max()+1
                heatmap_gt = np.zeros((im.shape[0]//2,) + im.shape[2:])
                pdist_gt = np.zeros((im.shape[0]//2,max_obj,36))
                pdist_ct = np.zeros((im.shape[0]//2,max_obj,2))
                pdist_mask = np.zeros((im.shape[0]//2,max_obj))
                pdist_ind = np.zeros((im.shape[0]//2,max_obj))
                pose_px_ind = [ [] ]*int(max_obj)*bs
                for j in range(bs):
                    label_set = obj_idx[j].unique()
                    for i in label_set:
                        if i==0: continue
                        obj_mask = obj_idx[j]==i
                        if obj_mask.sum()>200 and (obj_mask.sum()>(disc_aux[2][j,:,:,7].int()==i).sum()//2):
                            indices = torch.nonzero(obj_mask).float() # Nx2
                            center = indices.mean(0)
                            radius = (indices.max(0)[0] - indices.min(0)[0])/2
                            # get polarmask
                            #cv2.imwrite('/data/gengshay/0.png', 255*np.asarray(disc_aux[4][j,:,:,:3].cpu())[:,:,::-1])
                            #cv2.imwrite('/data/gengshay/1.png', np.asarray(disc_aux[2][j,:,:,7].cpu()))
                            #cv2.imwrite('/data/gengshay/2.png', np.asarray((obj_idx[j]).cpu()) )
                            pdist_gtx,centerx = get_polarmask(obj_mask)
                    #        print(i)
                    #        print(obj_mask.sum())
                    #        print(center)
                    #        print(radius)
                        
                            radius = gaussian_radius(np.asarray(radius.cpu()))
                            radius = max(0, int(radius))
                            draw_umich_gaussian(heatmap_gt[j], centerx, radius)
                            pdist_gt[j,i] = pdist_gtx
                            pdist_mask[j,i] = 1
                            pdist_ind[j,i] = centerx[1]*im.shape[3]+centerx[0]
                            pdist_ct[j,i]=centerx
                            pose_px_ind[j*max_obj+i] = obj_mask

                    #cv2.imwrite('/data/gengshay/3.png', 255*heatmap_gt[j]) 
                    #cv2.imwrite('/data/gengshay/4.png', 255*np.asarray(heatmap[j,0].detach().cpu())) 
                heatmap_gt = torch.Tensor(heatmap_gt)[:,None].cuda()
                loss += 0.05* FocalLoss()(heatmap, heatmap_gt, heatmap_logits) / (im.shape[2]*im.shape[3])
                
                pdist_ind = torch.Tensor(pdist_ind).cuda().long()
                pdist_mask = torch.Tensor(pdist_mask).cuda().bool()
                pdist_gt = torch.Tensor(pdist_gt).cuda()  # bs, nobj, 36
                pdist_ct = torch.Tensor(pdist_ct).cuda()  # bs, nobj, 2
                loss_polar,pdist = polar_reg(pdist, pdist_mask, pdist_ind, pdist_gt)
                loss += loss_polar*1e-8
                            
                # RT prediction
#                loss_pose,vis_pose = pose_reg(pred_quat, pred_tran, pose_px_ind, pdist_ind, gt_p03d, gt_p13d, gt_depth,max_obj,p03d,disc_aux[4][:bs])
#                loss += loss_pose*1e-6

                heatmap_vis = heatmap/heatmap.view(bs,-1).max(-1)[0][:,None,None,None]
                contour_gt = np.zeros((bs,)+im.shape[2:]+(3,))
                contour_pd = np.zeros((bs,)+im.shape[2:]+(3,))
                angles = torch.range(0, 350, 10).cuda() / 180 * math.pi
                for i in range(bs):
                    if pdist_mask[i].sum()>0:
                        contour = distance2mask(pdist_ct[i][pdist_mask[i]], pdist_gt[i][pdist_mask[i]], angles, im.shape[2:])
                        contour = np.asarray(contour.permute(0,2,1).cpu()[:,:,None],dtype=int)
                        contour_gt[i] = cv2.drawContours(contour_gt[i], contour, -1,1,3)
                        #cv2.imwrite('/data/gengshay/0.png',contour_gt[i])
                        
                        contour = distance2mask(pdist_ct[i][pdist_mask[i]], pdist[i][pdist_mask[i]], angles, im.shape[2:])
                        contour = np.asarray(contour.permute(0,2,1).detach().cpu()[:,:,None],dtype=int)
                        contour_pd[i] = cv2.drawContours(contour_pd[i], contour, -1,1,3)
                        #cv2.imwrite('/data/gengshay/0.png',contour_pd[i])
                contour_gt = torch.Tensor(contour_gt).cuda()
                contour_pd = torch.Tensor(contour_pd).cuda()
                m3d_gt = torch.cat([ disc_aux[2][:,:,:,7], (m3d_gt*1000).clamp(0,255), 
                                 255*heatmap_gt[:,0], 255*heatmap_vis[:,0], 
                                 255*contour_gt[:,:,:,0], 255*contour_pd[:,:,:,0], 
       (100*F.interpolate(log_dratio,[H,W], mode='bilinear')[:,0].exp()).clamp(0,255),
                            #(10000*vis_pose[:,0]).clamp(0,255)],1)
                           ],1)

                if np.isnan(loss.min().detach().cpu()):
                    pdb.set_trace()
                    loss = 0.01* FocalLoss()(heatmap, heatmap_gt)
                return flow2*4, flow3*8,flow4*16,flow5*32,flow6*64,loss, fgmask[:,0], m3d_gt
            else:
                flow2 = F.upsample(flow2, [im.size()[2],im.size()[3]], mode='bilinear')
                flow3 = F.upsample(flow3, [im.size()[2],im.size()[3]], mode='bilinear')
                flow4 = F.upsample(flow4, [im.size()[2],im.size()[3]], mode='bilinear')
                flow5 = F.upsample(flow5, [im.size()[2],im.size()[3]], mode='bilinear')
                flow6 = F.upsample(flow6, [im.size()[2],im.size()[3]], mode='bilinear')
                # flow supervised
                flowl0 = disc_aux[0].permute(0,3,1,2).clone()
                mask = disc_aux[1].clone()
                loss =  1.0*torch.norm((flow2*4-flowl0[:,:2]),2,1)[mask].mean() +\
                        0.5*torch.norm((flow3*8-flowl0[:,:2]),2,1)[mask].mean() + \
                      0.25*torch.norm((flow4*16-flowl0[:,:2]),2,1)[mask].mean() + \
                      0.25*torch.norm((flow5*32-flowl0[:,:2]),2,1)[mask].mean() + \
                      0.25*torch.norm((flow6*64-flowl0[:,:2]),2,1)[mask].mean()

                # out-of-range loss
                im_warp,_ = self.warpx(im[bs:], flowl0[:,:2])
                occ_mask = (im_warp - im[:bs]).norm(dim=1)>0.3

                up_flow3 = F.upsample(up_flow3, [im.size()[2],im.size()[3]], mode='bilinear')*4
                up_flow4 = F.upsample(up_flow4, [im.size()[2],im.size()[3]], mode='bilinear')*8
                up_flow5 = F.upsample(up_flow5, [im.size()[2],im.size()[3]], mode='bilinear')*16
                up_flow6 = F.upsample(up_flow6, [im.size()[2],im.size()[3]], mode='bilinear')*32
                oor2 = F.upsample(oor2[:,np.newaxis], [im.size()[2],im.size()[3]], mode='bilinear')[:,0]
                oor3 = F.upsample(oor3[:,np.newaxis], [im.size()[2],im.size()[3]], mode='bilinear')[:,0]
                oor4 = F.upsample(oor4[:,np.newaxis], [im.size()[2],im.size()[3]], mode='bilinear')[:,0]
                oor5 = F.upsample(oor5[:,np.newaxis], [im.size()[2],im.size()[3]], mode='bilinear')[:,0]
                oor6 = F.upsample(oor6[:,np.newaxis], [im.size()[2],im.size()[3]], mode='bilinear')[:,0]
                loss += self.get_oor_loss(flowl0[:,:2]-0,        oor6, (64* self.flow_reg64.flowx.max()),occ_mask,mask)
                loss += self.get_oor_loss(flowl0[:,:2]-up_flow6, oor5, (32* self.flow_reg32.flowx.max()),occ_mask,mask)
                loss += self.get_oor_loss(flowl0[:,:2]-up_flow5, oor4, (16* self.flow_reg16.flowx.max()),occ_mask,mask)
                loss += self.get_oor_loss(flowl0[:,:2]-up_flow4, oor3, (8* self.flow_reg8.flowx.max())  ,occ_mask,mask)
                loss += self.get_oor_loss(flowl0[:,:2]-up_flow3, oor2, (4* self.flow_reg4.flowx.max())  ,occ_mask,mask)
                return flow2*4, flow3*8,flow4*16,flow5*32,flow6*64,loss, oor2, oor2
        else:
            flow2 = F.upsample(flow2.detach(), [im.size()[2],im.size()[3]], mode='bilinear')*4
            return flow2, oor2[0],  dchange2[0,0], iexp2[0,0], fgmask[0,0], heatmap[0,0], polarmask, disp[0,0]
