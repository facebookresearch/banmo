# MIT License
# 
# Copyright (c) 2019 Carnegie Mellon University
# Copyright (c) 2021 Google LLC
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# modified from https://github.com/gengshan-y/VCN

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

        flow2 = F.upsample(flow2.detach(), [im.size()[2],im.size()[3]], mode='bilinear')*4
        return flow2, oor2[0]
