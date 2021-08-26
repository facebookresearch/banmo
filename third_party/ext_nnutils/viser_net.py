from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import cv2
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh
import pytorch3d
import pytorch3d.loss

import sys
sys.path.insert(0, 'third_party')
import pdb
from ext_utils import mesh
from ext_utils import geometry as geom_utils
from . import net_blocks as nb
import kornia
import configparser
import soft_renderer as sr
from ext_nnutils.geom_utils import pinhole_cam, obj_to_cam, raycast, render_multiplex, rotation_multiplex
from ext_nnutils import loss_utils
from ext_nnutils.geom_utils import label_colormap
from ext_utils.quatlib import q_rnd_m, q_scale_m
citylabs = label_colormap()
from ext_nnutils.nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse, get_joint_embedding_function, get_time_embedding_function,
      load_blender_data, load_llff_data, meshgrid_xy, models,
      run_network, run_one_iter_of_nerf) 
from ext_nnutils.nerf.nerf_helpers import positional_encoding
from ext_nnutils.vcnunet import pspnet, SurfaceMatchNet

        
def get_grid(B,H,W):
    meshgrid_base = np.meshgrid(range(0,W), range(0,H))[::-1]
    basey = np.reshape(meshgrid_base[0],[1,1,1,H,W])
    basex = np.reshape(meshgrid_base[1],[1,1,1,H,W])
    grid = torch.tensor(np.concatenate((basex.reshape((-1,H,W,1)),basey.reshape((-1,H,W,1))),-1)).cuda().float()
    return grid.view(1,1,H,W,2)

def affine(flow, flmask,pw=2):
    err_th = 1e-2
    b,_,lh,lw=flow.shape
    pref = get_grid(b,lh,lw)[:,0].permute(0,3,1,2).repeat(b,1,1,1).clone()
    pref[:,0] = pref[:,0]/(lw-1)*2-1 # x
    pref[:,1] = pref[:,1]/(lh-1)*2-1 # x

    ptar = pref + flow
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
    mask = (exp>0.5) & (exp<2) & (Error<err_th)
    #mask = (exp>0.5) & (exp<2) & (Error<err_th) & (flmask.bool()) & (count[:,0]==(pw*2+1)**2)
    mask = mask[:,0]

    # for vis
    #exp = exp.clamp(0.5,2)
    #exp[Error>err_th]=1
    return exp, mask


def laplacian_decay(curr_steps,max_steps, min_wt,max_wt):
    #current = np.exp(curr_steps/float(max_steps)*(np.log(min_wt)-np.log(max_wt))) * max_wt 
    current = max_wt
    #print( current)
    return current

def reg_decay(curr_steps, max_steps, min_wt,max_wt):
    """
    max weight to min weight
    """
    if curr_steps>max_steps:current = min_wt
    else:
        current = np.exp(curr_steps/float(max_steps)*(np.log(min_wt)-np.log(max_wt))) * max_wt 
    #print( current)
    return current

def reg_increase(curr_steps, max_steps, min_wt,max_wt):
    """
    min weight to max weight
    """
    if curr_steps>max_steps:current = max_wt
    else:
        const = (np.exp(max_wt)-np.exp(min_wt))
        current = np.log(curr_steps/float(max_steps)*const+np.exp(min_wt))
    return current

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
    return theta
def render_flow_soft_2(renderer_soft, verts, faces, verts_pos0, verts_pos1, pp0, pp1, proj_cam0,proj_cam1):
    # flow (no splat): 1) get mask; 2) render 3D coords for 1st/2nd frame 
    n_hypo = verts.shape[0] // faces.shape[0]
    #faces = faces[:,None].repeat(1,n_hypo,1,1).view(-1,faces.shape[1],3)
    verts_pos0 = verts_pos0.clone()
    verts_pos1 = verts_pos1.clone()
    offset = torch.Tensor( renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    
    nb = verts.shape[0]
    verts_pos_px = renderer_soft.render_mesh(sr.Mesh(torch.cat([verts_pre         ,verts_pre],0),
                                                     torch.cat([faces             ,faces],0), 
                                            textures=torch.cat([verts_pos0[:,:,:3],verts_pos1[:,:,:3]],0),texture_type='vertex')).clone()
    fgmask = verts_pos_px[:nb,-1]
    verts_pos_px = verts_pos_px[:,:3]
    verts_pos0_px = verts_pos_px[:nb].permute(0,2,3,1)
    verts_pos1_px = verts_pos_px[nb:].permute(0,2,3,1)
    
    bgmask = (verts_pos0_px[:,:,:,2]<1e-9) | (verts_pos1_px[:,:,:,2]<1e-9)
    verts_pos0_px[bgmask]=10
    verts_pos1_px[bgmask]=10
    # projet 3D verts with different intrinsics
    verts_pos0_px[:,:,:,1] = pp0[:,1:2,np.newaxis]+verts_pos0_px[:,:,:,1].clone()*proj_cam0[:,1:2,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos0_px[:,:,:,0] = pp0[:,0:1,np.newaxis]+verts_pos0_px[:,:,:,0].clone()*proj_cam0[:,0:1,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,1] = pp1[:,1:2,np.newaxis]+verts_pos1_px[:,:,:,1].clone()*proj_cam1[:,1:2,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,0] = pp1[:,0:1,np.newaxis]+verts_pos1_px[:,:,:,0].clone()*proj_cam1[:,0:1,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
#    flow_fw = (verts_pos1_px - verts_pos0_px)[:,:,:,:2]
    flow_fw = (verts_pos1_px - verts_pos0_px.detach())[:,:,:,:2]
    flow_fw[bgmask] = flow_fw[bgmask].detach()
    return flow_fw, bgmask, fgmask

def render_flow_soft_3(renderer_soft, verts, verts_target, faces):
    # flow (no splat): 1) get mask; 2) render 3D coords for 1st/2nd frame 
    offset = torch.Tensor( renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]

    verts_pos_px = renderer_soft.render_mesh(sr.Mesh(verts_pre, faces,
                                            textures=verts_target[:,:,:3],texture_type='vertex')).clone()
    fgmask = verts_pos_px[:,-1]
    verts_pos_px = verts_pos_px.permute(0,2,3,1)

    bgmask = (verts_pos_px[:,:,:,2]<1e-9)
    verts_pos_px[bgmask]=10

    verts_pos0_px = torch.Tensor(np.meshgrid(range(bgmask.shape[2]), range(bgmask.shape[1]))).cuda()
    verts_pos0_px[0] = verts_pos0_px[0]*2 / (bgmask.shape[2] - 1) - 1
    verts_pos0_px[1] = verts_pos0_px[1]*2 / (bgmask.shape[1] - 1) - 1
    verts_pos0_px = verts_pos0_px.permute(1,2,0)[None]

    flow_fw = (verts_pos_px[:,:,:,:2] - verts_pos0_px)
    flow_fw[bgmask] = flow_fw[bgmask].detach()
    return flow_fw, bgmask, fgmask


def render_flow_soft_3(renderer_soft, verts, verts_target, faces):
    # flow (no splat): 1) get mask; 2) render 3D coords for 1st/2nd frame 
    offset = torch.Tensor( renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    
    verts_pos_px = renderer_soft.render_mesh(sr.Mesh(verts_pre, faces, 
                                            textures=verts_target[:,:,:3],texture_type='vertex')).clone()
    fgmask = verts_pos_px[:,-1]
    verts_pos_px = verts_pos_px.permute(0,2,3,1)
    
    bgmask = (verts_pos_px[:,:,:,2]<1e-9)
    verts_pos_px[bgmask]=10

    verts_pos0_px = torch.Tensor(np.meshgrid(range(bgmask.shape[2]), range(bgmask.shape[1]))).cuda()
    verts_pos0_px[0] = verts_pos0_px[0]*2 / (bgmask.shape[2] - 1) - 1
    verts_pos0_px[1] = verts_pos0_px[1]*2 / (bgmask.shape[1] - 1) - 1
    verts_pos0_px = verts_pos0_px.permute(1,2,0)[None] 

    flow_fw = (verts_pos_px[:,:,:,:2] - verts_pos0_px)
    flow_fw[bgmask] = flow_fw[bgmask].detach()
    return flow_fw, bgmask, fgmask


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, pretrained=True):
        super(ResNetConv, self).__init__()
        #self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc=None
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)  # 1/8
            feat3 = x.clone()
        if n_blocks >= 3:
            x = self.resnet.layer3(x)  # 1/16
            feat4 = x.clone()
        if n_blocks >= 4:              
            x = self.resnet.layer4(x)  # 1/32
            feat5 = x.clone()
        return x, [feat3, feat4, feat5]

class PretrainedCNN(nn.Module):
    def __init__(self):
        super(PretrainedCNN, self).__init__()
        #self.resnet = torchvision.models.resnet18(pretrained=True)
        #self.resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = torchvision.models.resnext101_32x8d(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        #x = self.resnet.layer4[0](x)
        #x = self.resnet.layer4[1](x)
        #identity = x
        #out = self.resnet.layer4[2].conv1(x)
        #out = self.resnet.layer4[2].bn1(out)
        #out = self.resnet.layer4[2].relu(out)
        #out = self.resnet.layer4[2].conv2(out)
        #out = self.resnet.layer4[2].bn2(out)
        #out = self.resnet.layer4[2].relu(out)
        #out = self.resnet.layer4[2].conv3(out)
        #out = self.resnet.layer4[2].bn3(out)
        #out += identity
        #x=out
        return x



class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True, pretrained=True):
        super(Encoder, self).__init__()
        torch.manual_seed(0)
        self.resnet_conv = ResNetConv(n_blocks=4, pretrained=pretrained)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)
        #self.enc_fc = nb.fc_stack(256, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat,_ = self.resnet_conv.forward(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        #_,_,h,w=out_enc_conv1.shape
        #out_enc_conv1 = F.max_pool2d(out_enc_conv1, (h,w),(h,w), 0)[:,:,0,0]
        feat = self.enc_fc.forward(out_enc_conv1)

        #if self.training:
        #    feat = torch.nn.functional.dropout(feat,0.1)

        return feat

class UncertaintyPredictor(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat,  opts, img_H=64, img_W=128, n_upconv=5, nc_init=256):
        super(UncertaintyPredictor, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=1, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        uncertainty = self.decoder.forward(uvimage_pred)
        return uncertainty

class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        # pdb.set_trace()
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred)
        self.uvimage_pred = torch.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()

import torch.nn.functional as F
class ShapeMLP(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapeMLP, self).__init__()
        W = 256
        D = 8
        self.pts_linears = nn.ModuleList(
            [nn.Linear(nz_feat, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.output_linear = nn.Linear(W, num_verts*3)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        delta_v = self.output_linear(h)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v    


class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts, n_hypo=1):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, n_hypo * num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v

class ROTPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(ROTPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 6)
        #self.pred_layer = nn.Linear(nz_feat, 9)


    @staticmethod
    def cross_product( u, v):
        batch = u.shape[0]
        #print (u.shape)
        #print (v.shape)
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
            
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
            
        return out

    @staticmethod
    def normalize_vector( v, return_mag =False):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/(v_mag+1e-9)
        if(return_mag==True):
            return v, v_mag[:,0]
        else:
            return v

    def forward(self, feat):
        import kornia
        ortho6d = self.pred_layer.forward(feat) 
        x_raw = ortho6d[:,0:3]#batch*3
        y_raw = ortho6d[:,3:6]#batch*3
            
        x = self.normalize_vector(x_raw) #batch*3
        z = self.cross_product(x,y_raw) #batch*3
        z = self.normalize_vector(z)#batch*3
        y = self.cross_product(z,x)#batch*3
            
        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        rotmat = torch.cat((x,y,z), 2) #batch*3*3

        #rotmat = self.pred_layer.forward(feat).view(-1,3,3)
        #for i in range(rotmat.shape[0]):
        #    if rotmat[i].det()==0:
        #        pdb.set_trace()
        #        rotmat[i] = rotmat[i] + torch.eye(3)*1e-4*rotmat.mean()
        #try:
        #    u,s,v = torch.svd(rotmat,some=True, compute_uv=True)
        #except:                     # torch.svd may have convergence issues for GPU and CPU.
        #    rotmat = rotmat + torch.eye(3)*1e-4*rotmat.mean()
        #    u,s,v = torch.svd(rotmat,some=True, compute_uv=True)
        #rotmat = u.matmul(v.permute(0,2,1))
        #rotmat *= ((rotmat.det()<0).detach().float()[:,np.newaxis,np.newaxis]*-2)+1
        return kornia.rotation_matrix_to_quaternion(rotmat)

class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False,n_mesh=None,n_hypo=None):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot*n_mesh*n_hypo)
        self.classify_rot = classify_rot
        self.nmesh = n_mesh
        self.nhypo = n_hypo
        self.nz_feat = nz_feat

    def forward(self, feat):
        quat = self.pred_layer.forward(feat).view(-1,4)
        quat = quat.view(-1,self.nhypo,self.nmesh,4)
        #quat[:,:,1:,:3]/=1e2
        quat[:,:,1:,3]+=10
        quat = quat.view(-1,4)
        #quat.view(-1,self.nmesh,4)[:,1:,-1] *= 10
        #quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        #quat *= ((quat[:,:1]<0).detach().float()*-2)+1
        #return quat
        import kornia
        return kornia.quaternion_to_rotation_matrix(quat).view(-1,9)

    def reinit(self, n_hypo, good_hypo):
        prev_wt = self.pred_layer.weight.clone().view(n_hypo, -1, self.nz_feat)
        prev_bi = self.pred_layer.bias.clone().view(n_hypo,-1)
        
#        torch.nn.init.xavier_uniform(self.pred_layer.weight)
#        new_wt = self.pred_layer.weight.clone()
#        new_bi = torch.zeros(pred_bi.shape)
        
        prev_wt[~good_hypo] = prev_wt[good_hypo]
        prev_bi[~good_hypo] = prev_bi[good_hypo]

        self.pred_layer.weight.data = prev_wt.view(-1,self.nz_feat)
        self.pred_layer.bias.data   = prev_bi.view(-1,)


class DepthPredictor(nn.Module):
    def __init__(self, nz,n_mesh=None,offset=10,outdim=1):
        super(DepthPredictor, self).__init__()
        #self.pred_layer = nn.Linear(nz, 8)
        self.pred_layer = nn.Linear(nz, outdim*n_mesh)
        #self.pred_layer = nn.Linear(nz, 64)
        self.nz_feat = nz
        self.offset=offset

    def forward(self, feat):
        #depth = (0.1*self.pred_layer.forward(feat)).exp()
        #depth = (0.5*self.pred_layer.forward(feat)+0.5).exp()
    #    # multi-hypothesis 
    #    depth = self.pred_layer.forward(feat)
    #    depth = depth + torch.Tensor(np.linspace(0,50,8)).cuda()[np.newaxis]
    #    depth = torch.nn.functional.relu(depth) + 1e-12
        depth = self.pred_layer.forward(feat) + self.offset # mean is 10+1e-12, min is 1e-12
        depth = torch.nn.functional.relu(depth) + 1e-12
      #  # soft-regression
      #  depth = torch.nn.functional.softmax(5*self.pred_layer.forward(feat) ,1)
      #  depth = (depth * torch.Tensor(np.linspace(0,20,64)).cuda()[np.newaxis]).sum(-1,keepdims=True)
        return depth

    def reinit(self, n_hypo, good_hypo):
        prev_wt = self.pred_layer.weight.clone().view(n_hypo, -1, self.nz_feat)
        prev_bi = self.pred_layer.bias.clone().view(n_hypo,-1)
        
        prev_wt[~good_hypo] = prev_wt[good_hypo]
        prev_bi[~good_hypo] = prev_bi[good_hypo]

        self.pred_layer.weight.data = prev_wt.view(-1,self.nz_feat)
        self.pred_layer.bias.data   = prev_bi.view(-1,)

class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True,n_mesh=None):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2*n_mesh)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat).view(-1,2)
        #trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class PPointPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(PPointPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 2)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(self, nz_feat=100, num_verts=1000,n_mesh = None, n_hypo=None,tex_code_dim=64, shape_code_dim=64):
        super(CodePredictor, self).__init__()
        self.offset = 10
        torch.manual_seed(0)
        #self.quat_predictor = ROTPredictor(nz_feat)
        self.quat_predictor = QuatPredictor(nz_feat, n_mesh=n_mesh, n_hypo=n_hypo)
        self.scale_predictor = DepthPredictor(nz_feat,n_mesh=n_hypo,offset=self.offset, outdim=2)
        self.trans_predictor = TransPredictor(nz_feat,n_mesh=n_mesh)
        self.depth_predictor = DepthPredictor(nz_feat,n_mesh=n_mesh,offset=self.offset)
        self.ppoint_predictor = PPointPredictor(nz_feat)
        self.tex_predictor = nn.Linear(nz_feat, tex_code_dim)
        self.shape_predictor = nn.Linear(nz_feat, shape_code_dim)

        self.nmesh = n_mesh
        self.nhypo = n_hypo

    def forward(self, feat):
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        
        trans_pred = self.trans_predictor.forward(feat)/10.
      #  trans_pred = trans_pred.view(-1,self.nmesh,2)
      #  trans_pred[:,1:] /= 10.
      #  trans_pred = trans_pred.view(-1,2)

        depth_pred = self.depth_predictor.forward(feat)
        depth_pred = depth_pred.view(-1,1,self.nmesh)
        depth_pred[:,:,1:] =  (depth_pred[:,:,1:]-self.offset)/10.
        #depth_pred[:,:,1:] =  (depth_pred[:,:,1:]-10)/100.
        depth_pred = depth_pred.view(feat.shape[0],-1)
        
        ppoint_pred = self.ppoint_predictor.forward(feat)/100.
        #ppoint_pred = self.ppoint_predictor.forward(feat)/10.

        tex_code = self.tex_predictor.forward(feat)
        shape_code = self.shape_predictor.forward(feat)
        return scale_pred, trans_pred, quat_pred, depth_pred, ppoint_pred, tex_code, shape_code

#------------ Mesh Net ------------#
#----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture
        self.reinit_bones = True

        # multi-vid
        sym_angle = 0
        ppx=960; ppy=540
        numvid = 1
        self.numvid = numvid

        ##  Mean mesh
        if osp.exists('tmp/sphere_%d.npy'%(opts.subdivide)):
            sphere = np.load('tmp/sphere_%d.npy'%(opts.subdivide),allow_pickle=True)[()]
            verts = sphere[0]
            faces = sphere[1]
        else:
            verts, faces = mesh.create_sphere(opts.subdivide)
            np.save('tmp/sphere_%d.npy'%(opts.subdivide),[verts,faces])
        self.num_verts = verts.shape[0]
        self.num_faces = faces.shape[0]

        self.mean_v = nn.Parameter(torch.Tensor(verts)[None])
        faces = Variable(torch.LongTensor(faces), requires_grad=False)
        self.texture_type = 'vertex'
        if self.opts.opt_tex=='yes':
            self.tex = nn.Parameter(torch.normal(torch.zeros(1,self.num_verts,3).cuda(),1))
        else:
            self.tex = torch.normal(torch.zeros(self.num_verts,3).cuda(),1)
        
        self.mean_v.data = self.mean_v.data.repeat(self.numvid*opts.n_hypo,1,1).view(self.numvid, opts.n_hypo,self.num_verts,3)
        self.tex.data = self.tex.data.repeat(self.numvid*opts.n_hypo,1,1).view(self.numvid, opts.n_hypo, self.num_verts,3)  # id, hypo, F, 3
        faces = faces[None].repeat(self.numvid*opts.n_hypo,1,1).view(self.numvid, opts.n_hypo, self.num_faces,3)
        self.joint_ts =  torch.zeros(self.numvid*opts.n_hypo*(opts.n_mesh-1),3).cuda().view(self.numvid, opts.n_hypo, -1, 3)
        self.ctl_ts =  torch.zeros(self.numvid*opts.n_hypo*(opts.n_mesh-1),3).cuda().view(self.numvid, opts.n_hypo, -1, 3)
        self.ctl_rs =  torch.Tensor([[0,0,0,1]]).repeat(self.numvid*opts.n_hypo*(opts.n_mesh-1),1).cuda().view(self.numvid, opts.n_hypo, -1, 4)
        self.log_ctl = torch.Tensor([[0,0,0]]).repeat(self.numvid*opts.n_hypo*(opts.n_mesh-1),1).cuda().view(self.numvid, opts.n_hypo, -1, 3)  # control point varuance
       
        self.faces=faces

        if self.opts.n_mesh>1:
            self.ctl_rs  = nn.Parameter(self.ctl_rs) 
            self.ctl_ts  = nn.Parameter(self.ctl_ts) 
            self.joint_ts  = nn.Parameter(self.joint_ts) 
            self.log_ctl = nn.Parameter(self.log_ctl)

        # shape basis
        self.shape_basis = nn.Parameter(torch.normal(torch.zeros(8*opts.n_hypo,self.num_verts,3).cuda(),0.1))
        self.shape_code_fix = nn.Parameter(torch.zeros(self.numvid, opts.n_hypo, 60).cuda()) # vid, hp, 8

        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

        self.encoder = nn.ModuleList([Encoder(input_shape, n_blocks=4, nz_feat=nz_feat) for i in range(opts.n_hypo)])


        self.code_predictor = nn.ModuleList([CodePredictor(nz_feat=nz_feat, \
                 num_verts=self.num_verts, n_mesh=opts.n_mesh, n_hypo = 1,\
                 tex_code_dim=8, shape_code_dim=8) for i in range(opts.n_hypo)])

        self.codedim = 60
        self.encoder_class = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor_class = CodePredictor(nz_feat=nz_feat, \
                  num_verts=self.num_verts,n_mesh=opts.n_mesh,n_hypo = opts.n_hypo,\
          tex_code_dim=self.codedim, shape_code_dim=self.codedim)
          #tex_code_dim=self.codedim*opts.n_hypo, shape_code_dim=self.codedim*opts.n_hypo)
        self.shapePred = ShapePredictor(nz_feat, num_verts=self.num_verts, n_hypo=opts.n_hypo)

        self.rotg =  nn.Parameter(torch.Tensor([[[0,0,0,1]]]).cuda().repeat(self.numvid*opts.n_hypo,opts.n_mesh,1))
        self.rotgf = torch.Tensor([[[0,0,0,1]]]).cuda().repeat(opts.n_hypo,opts.n_mesh,1)
        self.pps =   nn.Parameter(torch.Tensor([ppx, ppy]).cuda())
        self.light_params = nn.Parameter(torch.Tensor([1,0,-1,0]).cuda())  # intensity, x,y,z

        rotations = rotation_multiplex(2*opts.n_hypo)
        offset_mat = kornia.angle_axis_to_rotation_matrix(torch.Tensor([0,sym_angle/180.*np.pi,0]).cuda()[None]) # rotate along y axis
        rotations = torch.Tensor(rotations).cuda()
        rotations = offset_mat.matmul(rotations[:1].permute(0,2,1)).matmul(rotations)
        rotgf = kornia.rotation_matrix_to_quaternion(rotations)
        self.rotgf[:,:1] = rotgf[:,None]
        self.rotg.data = self.rotgf[None].repeat(self.numvid,1,1,1).view(-1,self.rotg.shape[1],4)
        
        # per-video canonical rotatoin
        self.rotvid =  nn.Parameter(torch.Tensor([[[0,0,0,1]]]).cuda().repeat(self.numvid*opts.n_hypo,opts.n_mesh,1))

        # For renderering.
        print(opts.img_size)
        self.renderer_soft = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softflf = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_softflb = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softpart = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-4, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_hardtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-12, 
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

        self.renderer_softdepth = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-3,gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.) 

        if opts.ptex:
            self.nerf_tex = nn.ModuleList([getattr(models, 'CondNeRFModel')(
                        num_encoding_fn_xyz=10,
                        num_encoding_fn_dir=4,
                        include_input_xyz=False,
                        include_input_dir=False,
                        use_viewdirs=False,
                        #codesize=0) for i in range(self.numvid*opts.n_hypo)])
                        codesize=self.codedim) for i in range(self.numvid*opts.n_hypo)])
            self.encode_position_fn_tex = get_embedding_function(
                num_encoding_functions=10,
                include_input=False,
                log_sampling=True,
            )


    def forward(self, input_imgs, imgs, masks, cams, flow, pp, occ, frameid, dataid):
        pdb.set_trace()
        if self.training:
            local_batch_size = input_imgs.shape[0]
            self.input_imgs   = input_imgs
            self.imgs         = imgs      
            self.masks        = masks     
            self.cams         = cams      
            self.flow         = flow      
            self.pp           = pp        
            self.occ          = occ       
            self.frameid      = frameid.long()
            self.dataid      = dataid.long()
        else:
            local_batch_size = len(batch_input)
            self.input_imgs = batch_input

        img = self.input_imgs
        opts = self.opts
        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()

        # assign instance texture and shape
        pred_v = torch.zeros(local_batch_size, opts.n_hypo, self.num_verts,3).cuda()
        faces = torch.zeros(local_batch_size, opts.n_hypo, self.num_faces,3).cuda().long()
        tex = torch.zeros_like(pred_v)
        joint_ts =  torch.zeros(local_batch_size, opts.n_hypo,opts.n_mesh-1,3).cuda()
        ctl_ts =  torch.zeros(local_batch_size, opts.n_hypo,opts.n_mesh-1,3).cuda()
        ctl_rs =  torch.zeros(local_batch_size, opts.n_hypo,opts.n_mesh-1,4).cuda()
        log_ctl = torch.zeros(local_batch_size, opts.n_hypo,opts.n_mesh-1,3).cuda()
        rotvid = torch.zeros(local_batch_size, opts.n_hypo, opts.n_mesh,4).cuda()
        rotg = torch.nn.functional.normalize(self.rotg,dim=-1).view(-1,opts.n_hypo,opts.n_mesh,4)[0]  # select the first rotg
        rotvid_tmp = torch.nn.functional.normalize(self.rotvid,dim=-1).view(-1,opts.n_hypo,opts.n_mesh,4)
        for i in range(local_batch_size):
            pred_v[i] = self.mean_v[self.dataid[i]]  # id, hp, n, 3
            tex[i] = self.tex[self.dataid[i]].sigmoid()
            faces[i] = self.faces[self.dataid[i]].cuda()
            joint_ts[i] = self.joint_ts[self.dataid[i]] # id, hp, nmesh-1, 3
            ctl_ts[i] = self.ctl_ts[self.dataid[i]] # id, hp, nmesh-1, 3
            ctl_rs[i] = self.ctl_rs[self.dataid[i]] # id, hp, nmesh-1, 3
            log_ctl[i] = self.log_ctl[self.dataid[i]] # id, hp, nmesh-1, 3
            rotvid[i] = rotvid_tmp[self.dataid[i]]

            if not opts.reg3d:
                # select the first ins
                pred_v[i] = self.mean_v[0]
                tex[i] = self.tex[0].sigmoid()
                faces[i] = self.faces[0]
                joint_ts[i] = self.joint_ts[0] # id, hp, nmesh-1, 3
                ctl_ts[i] = self.ctl_ts[0] # id, hp, nmesh-1, 3
                ctl_rs[i] = self.ctl_rs[0] # id, hp, nmesh-1, 3
                log_ctl[i] = self.log_ctl[0] # id, hp, nmesh-1, 3

            ## select the first hypo
            #pred_v[i] = self.mean_v[0,:1].repeat(opts.n_hypo,1,1)
            #tex[i] = self.tex[0,:1].repeat(opts.n_hypo,1,1).sigmoid()
            #faces[i] = self.faces[0,:1].repeat(opts.n_hypo,1,1)
            #ctl_ts[i] = self.ctl_ts[0,:1].repeat(opts.n_hypo,1,1) # id, hp, nmesh-1, 3
            #ctl_rs[i] = self.ctl_rs[0,:1].repeat(opts.n_hypo,1,1) # id, hp, nmesh-1, 3
            #log_ctl[i] = self.log_ctl[0,:1].repeat(opts.n_hypo,1,1) # id, hp, nmesh-1, 3

        pred_v = pred_v.reshape(-1,self.num_verts,3)
        tex = tex.reshape(-1,self.num_verts,3)
        faces = faces.reshape(-1,self.num_faces,3)
        joint_ts = joint_ts.view(local_batch_size,opts.n_hypo,opts.n_mesh-1,3)
        ctl_ts = ctl_ts.view(local_batch_size,opts.n_hypo,opts.n_mesh-1,3)
        ctl_rs = ctl_rs.view(local_batch_size,opts.n_hypo,opts.n_mesh-1,4)
        log_ctl = log_ctl.view(local_batch_size,opts.n_hypo,opts.n_mesh-1,3)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('before nerf time:%.2f'%(time.time()-start_time))

        
        # replace texture with nerf here
        if opts.ptex:
            img_feat_class = self.encoder_class.forward(img)
            _, _, _, _, _, tex_code, shape_code = self.code_predictor_class.forward(img_feat_class)
            #tex_code = tex_code.view(local_batch_size, opts.n_hypo, -1)
            #freq, 2, 3 (sinx, siny, sinz, cosx, cosy, cosz)
            pred_v_symm = pred_v[:local_batch_size*opts.n_hypo].clone().detach()
            #if opts.symmetric_loss:
            #    pred_v_symm[..., 0] = pred_v_symm[..., 0].abs()  # symmetric texture
            texture_sampled = torch.zeros_like(pred_v_symm)
            for i in range(local_batch_size):
                for j in range(opts.n_hypo):
                    if opts.reg3d:
                        nerfidx = self.dataid[i]*opts.n_hypo+j
                    else:
                        nerfidx = j  # select the first 
                    sampidx = i*opts.n_hypo+j
                    nerf_out = run_network(
                        #self.nerf_tex,
                        self.nerf_tex[nerfidx],
                        pred_v_symm[sampidx:sampidx+1],
                        None,
                        131072,
                        self.encode_position_fn_tex,
                        None,
                        code=tex_code[i:i+1],
                    )
                    texture_sampled[sampidx] = nerf_out[:,:,:-1]

                            
            tex=texture_sampled[:,:,:3] # bs, nverts, 3
            tex=tex.sigmoid()
        if opts.debug:
            torch.cuda.synchronize()
            print('before shape nerf time:%.2f'%(time.time()-start_time))


            
        if self.training and (opts.finetune) and self.epoch<3: 
            pred_v = pred_v.detach()
            ctl_ts = ctl_ts.detach()
            if 'shape_delta' in locals():
                shape_delta = shape_delta.detach()

        def skinning(pred_v, ctl_ts, ctl_rs, log_ctl, opts, local_batch_size, num_verts):
            log_ctl[:] = 1
            skin = torch.zeros(local_batch_size,opts.n_hypo,opts.n_mesh-1, num_verts,1).cuda()
            for i in range(local_batch_size):
                dis_norm = (ctl_ts[i].view(opts.n_hypo,-1,1,3) - pred_v.view(local_batch_size,opts.n_hypo,-1,3)[i,:,None].detach()) # p-v, H,J,1,3 - H,1,N,3
                dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(ctl_rs[i]).view(opts.n_hypo,-1,3,3)) # h,j,n,3
                dis_norm = log_ctl[i].exp().view(opts.n_hypo,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
                dis_norm = (-10 * dis_norm.sum(3))
                
                topk, indices = dis_norm.topk(3, 1, largest=True)
                res = torch.zeros_like(dis_norm).fill_(-np.inf)
                res = res.scatter(1, indices, topk)
                dis_norm = res
                skin[i] = dis_norm.softmax(1)[:,:,:,None] # h,j,n,1

            skin = skin.view(-1,opts.n_mesh-1,pred_v.shape[-2],1)
            return skin

        # skin computation
        if opts.n_mesh>1:
            skin = skinning(pred_v, ctl_ts, ctl_rs, log_ctl, opts, local_batch_size, self.num_verts)
        #    log_ctl[:] = 1
        #    #ctl_var = (log_ctl).exp()
        #    #ctl_var = ctl_var / ctl_var.sum(-1)[:,:,:,None] * np.exp(1)*3
        #    # GMM
        #    skin = torch.zeros(local_batch_size,opts.n_hypo,opts.n_mesh-1,self.num_verts,1).cuda()
        #    for i in range(local_batch_size):
        #        dis_norm = (ctl_ts[i].view(opts.n_hypo,-1,1,3) - pred_v.view(local_batch_size,opts.n_hypo,-1,3)[i,:,None].detach()) # p-v, H,J,1,3 - H,1,N,3
        #        dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(ctl_rs[i]).view(opts.n_hypo,-1,3,3)) # h,j,n,3
        #        #dis_norm = ctl_var[i].view(opts.n_hypo,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v)
        #        dis_norm = log_ctl[i].exp().view(opts.n_hypo,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
        #        skin[i] = (-10 * dis_norm.sum(3)).softmax(1)[:,:,:,None] # h,j,n,1
        #    skin = skin.view(-1,opts.n_mesh-1,pred_v.shape[-2],1)
        #   # fusion.meshwrite('tmp/clusters.ply', np.asarray(pred_v[0].detach().cpu()), np.asarray(self.model.faces.cpu()), colors=255*skin_colors.cpu()) 
        else:skin=None                

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        self.apply(set_bn_eval)

        scale =  torch.zeros(opts.n_hypo, local_batch_size, 2).cuda()
        ppoint = torch.zeros(opts.n_hypo, local_batch_size, 2).cuda()
        quat =   torch.zeros(opts.n_hypo, local_batch_size, opts.n_mesh * 9).cuda()
        trans =  torch.zeros(opts.n_hypo, local_batch_size, opts.n_mesh * 2).cuda()
        depth =  torch.zeros(opts.n_hypo, local_batch_size, opts.n_mesh).cuda()
        for i in range(opts.n_hypo):
            if opts.ptime:
                img_feat = run_network(
                self.nerf_time,
                #self.frameid[:,None].repeat(1,3)/100*2-1,
                self.frameid[:,None]/100*2-1,
                None,
                131072,
                self.encode_position_fn_time,
                None,
                code=None)[:,:-1]
                img_feat = F.leaky_relu(img_feat, 0.2)
                img_feat = img_feat*10
                #img_feat = img_feat*5
                scalex, transx, quatx, depthx, ppointx, _, _ = self.code_predictor[i].forward(img_feat)
            else:
                img_feat       = self.encoder[i].forward(img)
                scalex, transx, quatx, depthx, ppointx, _, _ = self.code_predictor[i].forward(img_feat)





                # random jitter central mesh
                decay_factor = np.ones(local_batch_size)
                #decay_factor[self.dataid.cpu()!=0] = 0.
                noise = q_scale_m(q_rnd_m(b=local_batch_size), decay_factor)  # wxyz
                noise = torch.Tensor(noise).cuda()
                noise = torch.cat([noise[:,1:], noise[:,:1]],-1)
                noise = kornia.quaternion_to_rotation_matrix(noise)
                pred_v = pred_v.matmul(noise)

                shape_rand = pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.meshes.Meshes(verts=pred_v, faces=faces), 1000, return_normals=False)
                shape_rand = shape_rand.permute(0,2,1).clone().detach()
                shape_feat_rand = self.shape_encoder(shape_rand)
                rot_pred = self.compress(shape_feat_rand)
                pred_v = pred_v.matmul(noise.reshape(-1,3,3).permute(0,2,1))

            quatx = quatx.view(-1,opts.n_mesh,3,3)
            quat_body = quatx[:,:1]
            quat_body = quat_body.clone().matmul(kornia.quaternion_to_rotation_matrix(rotvid[:,i].clone()).view(local_batch_size,-1,3,3)[:,:1])

            quatx = torch.cat([quat_body, quatx[:,1:]],1)

            #scalex[:,1] = scalex[:,0].clone()
            scale[i] = scalex.view(local_batch_size,-1)
            trans[i] = transx.view(local_batch_size,-1)
            quat[i] = quatx.view(local_batch_size,-1)
            depth[i] = depthx.view(local_batch_size,-1)
            ppoint[i] = ppointx.view(local_batch_size,-1)

        scale = scale.permute(1,0,2)
        ppoint=ppoint.permute(1,0,2)
        quat=quat.permute(1,0,2)
        trans=trans.permute(1,0,2)
        depth=depth.permute(1,0,2)

        # change according to intrinsics 
#        scale = self.cams[:,None,:1]*scale # here assumes intrinsics change
#        depth[:,:,:1] = self.cams[:,None,:1]* depth[:,:,:1]
        
        # # equal focal length
        scale[:,:,1] = scale[:,:,0].clone() * self.cams[:,None,1] / self.cams[:,None,0]

        if not opts.cnnpp:
            # fix pp
            ppoint = (self.pps[None,None]-self.pp[:,None])/128. * self.cams[:,None,:2] - 1
            ppoint_raw = ppoint.clone()
        if self.training and opts.cnnpp:
            ppb1 = self.cams[:local_batch_size//2,None,:2]*self.pp[:local_batch_size//2,None]/(opts.img_size/2.)
            ppb2 = self.cams[local_batch_size//2:,None,:2]*self.pp[local_batch_size//2:,None]/(opts.img_size/2.)
            ppa1 = ppoint[:local_batch_size//2] + ppb1 + 1
            ppa2 = ppa1 * (self.cams[local_batch_size//2:,None,:2] / self.cams[:local_batch_size//2,None,:2]) 
            ppoint[local_batch_size//2:]= ppa2 - ppb2 -1

        if not self.training:
            self.uncrop_scale = scale.clone() / self.cams[None] * 128
            self.uncrop_pp = (ppoint[0,0] + 1)*128/self.cams[0] + self.pp[0]
        
        quat = quat.reshape(-1,9)
        depth = depth.reshape(-1,1)
        trans = trans.reshape(-1,2)
        ppoint = ppoint.reshape(ppoint.shape)
        scale = scale.reshape(scale.shape)

        #scale = scale.detach()
        #ppoint = ppoint.detach()
        #quat = quat.reshape(-1, opts.n_mesh, 9)
        #quat = torch.cat([quat[:,:1].detach(), quat[:,1:]],1).reshape(-1,9)
        #depth = depth.reshape(-1, opts.n_mesh, 1)
        #depth = torch.cat([depth[:,:1].detach(), depth[:,1:]],1).reshape(-1,1)
        #trans = trans.reshape(-1, opts.n_mesh, 2)
        #trans = torch.cat([trans[:,:1].detach(), trans[:,1:]],1).reshape(-1,2)

        noise_rot = torch.eye(3).cuda()[None]
        ## rendering
        if self.training and opts.noise and self.epoch>2 and self.iters<100 and self.iters>1:
        #if self.training and opts.noise and self.epoch>0 and self.iters<100 and self.iters>1:
            # add noise
            decay_factor = 0.2*(1e-4)**(self.iters/100)
            decay_factor_r = decay_factor * np.ones(quat.shape[0])
            ### smaller noise for bones
            decay_factor_r = decay_factor_r.reshape((-1,opts.n_mesh))
            rotmag = pytorch3d.transforms.quaternion_to_axis_angle(pytorch3d.transforms.matrix_to_quaternion(quat.view(-1,3,3))).norm(2,-1)/(np.pi*2)
            #decay_factor_r[:,1:] = np.asarray(rotmag.detach().view(-1,opts.n_mesh)[:,1:].cpu()) * 2
            decay_factor_r[:,1:] *= 0
            decay_factor_r[:,0] *=  1
            decay_factor_r = decay_factor_r.flatten()
            noise = q_scale_m(q_rnd_m(b=quat.shape[0]), decay_factor_r)  # wxyz
            #if opts.local_rank==0:
            #    print(self.iters)
            #    print(decay_factor)
            #    print(noise[0])
            noise = torch.Tensor(noise).cuda()
            noise = torch.cat([noise[:,1:], noise[:,:1]],-1)
            #noise[noise.shape[0]//2:] = noise[:noise.shape[0]//2]
            noise_rot = kornia.quaternion_to_rotation_matrix(noise)
            quat = quat.view(-1,3,3).matmul(noise_rot).view(-1,9)

            decay_factor_s = decay_factor
            noise = (decay_factor_s*torch.normal(torch.zeros(scale.shape).cuda(),opts.rscale)).exp()
            #noise[noise.shape[0]//2:] = noise[:noise.shape[0]//2]
            scale = scale * noise
           
#            decay_factor_t = decay_factor*0.1*torch.ones(trans.shape)
#            decay_factor_d = decay_factor*0.1*torch.ones(depth.shape)
#            decay_factor_t = decay_factor_t.view(local_batch_size,opts.n_mesh,2)
#            decay_factor_t[:,1:] *= 5
#            decay_factor_t = decay_factor_t.view(-1,2)
#            decay_factor_d = decay_factor_d.view(local_batch_size,opts.n_mesh,1)
#            decay_factor_d[:,1:] *= 5
#            decay_factor_d = decay_factor_d.view(-1,1)
#            trans += torch.normal(torch.zeros(trans.shape),decay_factor_t).cuda()
#            depth += torch.normal(torch.zeros(depth.shape),decay_factor_d).cuda()

        
        ## rendering
        # obj-cam rigid transform;  proj_cam: [focal, tx,ty,qw,qx,qy,qz]; 
        # 1st/2nd frame stored as 1st:[0:local_batch_size//2], 2nd: [local_batch_size//2:-1]
        # transforms [body-to-cam, part1-to-body, ...]
        Rmat = quat.clone().view(-1,3,3).permute(0,2,1)
        Tmat = torch.cat([trans, depth],1)
        joint_ts = joint_ts.view(-1,opts.n_mesh-1,3,1)
        ctl_ts = ctl_ts.view(-1,opts.n_mesh-1,3,1)
        if opts.n_mesh>1:
            # part transform
            # Gg*Gx*Gg_inv
            Rmat = Rmat.view(-1,opts.n_mesh,3,3)
            Tmat = Tmat.view(-1,opts.n_mesh,3,1)
            Tmat[:,1:] = -Rmat[:,1:].matmul(joint_ts)+Tmat[:,1:]+joint_ts
            #Tmat[:,1:] = -Rmat[:,1:].matmul(ctl_ts)+Tmat[:,1:]+ctl_ts
            Rmat[:,1:] = Rmat[:,1:].permute(0,1,3,2)
            Rmat = Rmat.view(-1,3,3)
            Tmat = Tmat.view(-1,3)

            self.ctl_proj =    obj_to_cam(ctl_ts[:,:,:,0],  Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_mesh, opts.n_hypo, torch.eye(opts.n_mesh-1)[None,:,:,None].cuda())
            self.ctl_proj = pinhole_cam(self.ctl_proj, ppoint.detach(), scale.detach())
            self.joint_proj =    obj_to_cam(joint_ts[:,:,:,0],  Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_mesh, opts.n_hypo, torch.eye(opts.n_mesh-1)[None,:,:,None].cuda())
            self.joint_proj = pinhole_cam(self.joint_proj, ppoint.detach(), scale.detach())

        if not self.training:
           return [scale, trans, quat, depth, ppoint], pred_v, faces, tex, skin, Rmat, Tmat
        

        self.deform_v = obj_to_cam(pred_v, Rmat.view(-1,3,3), Tmat[:,np.newaxis,:],opts.n_mesh, opts.n_hypo,skin,tocam=False)


        if opts.debug:
            torch.cuda.synchronize()
            print('before rend time:%.2f'%(time.time()-start_time))
        
        # 1) flow rendering 
        verts_fl = obj_to_cam(pred_v, Rmat, Tmat[:,np.newaxis,:],opts.n_mesh, opts.n_hypo,skin)
        verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
        verts_pos0 = verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[:local_batch_size//2].clone().view(local_batch_size//2*opts.n_hypo,-1,4)
        verts_pos1 = verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[local_batch_size//2:].clone().view(local_batch_size//2*opts.n_hypo,-1,4)
        verts_fl = pinhole_cam(verts_fl, ppoint, scale)

        dmax=verts_fl[:,:,-2].max()
        dmin=verts_fl[:,:,-2].min()
        self.renderer_softflf.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflf.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softflb.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflb.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softtex.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softtex.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softdepth.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softdepth.rasterizer.far= dmax+(dmax-dmin)/2

        if opts.finetune and '-ft' in opts.name:
            self.renderer_soft.rasterizer.sigma_val= 1e-5
            self.renderer_softflf.rasterizer.sigma_val= 1e-5
            self.renderer_softflb.rasterizer.sigma_val= 1e-5
            self.renderer_softtex.rasterizer.sigma_val= 1e-5


        if False:#opts.debug:
            self.flow_fw, self.bgmask_fw, self.fgmask_flowf = render_flow_soft_3(self.renderer_softflf,
                    verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[:local_batch_size//2].view(-1,verts_fl.shape[1],4),
                    verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[local_batch_size//2:].view(-1,verts_fl.shape[1],4),
                    faces.view(local_batch_size,opts.n_hypo,-1,3)[:local_batch_size//2].view(-1,faces.shape[1],3))
            self.flow_bw, self.bgmask_bw, self.fgmask_flowb = render_flow_soft_3(self.renderer_softflb, 
                    verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[local_batch_size//2:].view(-1,verts_fl.shape[1],4),
                    verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[:local_batch_size//2].view(-1,verts_fl.shape[1],4),
                    faces.view(local_batch_size,opts.n_hypo,-1,3)[local_batch_size//2:].view(-1,faces.shape[1],3))
        else:
            self.flow_fw, self.bgmask_fw, self.fgmask_flowf = render_flow_soft_2(self.renderer_softflf, verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[:local_batch_size//2].view(-1,verts_fl.shape[1],4), 
                                                  faces.view(local_batch_size,opts.n_hypo,-1,3)[:local_batch_size//2].view(-1,faces.shape[1],3), 
                                                  verts_pos0, verts_pos1, 
                                           ppoint[:local_batch_size//2].reshape(-1,2),
                                           ppoint[local_batch_size//2:].reshape(-1,2), 
                                            scale[:local_batch_size//2].reshape(-1,2),
                                            scale[local_batch_size//2:].reshape(-1,2))
            self.flow_bw, self.bgmask_bw, self.fgmask_flowb = render_flow_soft_2(self.renderer_softflb, verts_fl.view(local_batch_size,opts.n_hypo,-1,4)[local_batch_size//2:].view(-1,verts_fl.shape[1],4), 
                                                  faces.view(local_batch_size,opts.n_hypo,-1,3)[local_batch_size//2:].view(-1,faces.shape[1],3), 
                                                  verts_pos1, verts_pos0, 
                                           ppoint[local_batch_size//2:].reshape(-1,2),
                                           ppoint[:local_batch_size//2].reshape(-1,2), 
                                            scale[local_batch_size//2:].reshape(-1,2),
                                            scale[:local_batch_size//2].reshape(-1,2))
        self.bgmask =  torch.cat([self.bgmask_fw, self.bgmask_bw],0) 
        self.fgmask_flow =  torch.cat([self.fgmask_flowf, self.fgmask_flowb],0) 
        self.flow_rd = torch.cat([self.flow_fw, self.flow_bw    ],0) 
        ## flow vis
        #import utils.dydepth as ddlib
        #pdb.set_trace()
        #for i in range(local_batch_size//2):
        #    import cv2
        #    #flow = self.flow[i+local_batch_size//2:i+local_batch_size//2+1].detach()
        #    flow = self.flow[i:i+1].detach()
        #    #flow = self.flow[i:i+1].detach()
        #    #flow = self.flow_rd[i:i+1].permute(0,3,1,2).detach()
        #    img1 = self.imgs[i:i+1]
        #    img2 = self.imgs[i+local_batch_size//2:i+local_batch_size//2+1]
        #    grid =  torch.cat( [torch.arange(0, 256,out=torch.cuda.FloatTensor()).view(1,-1).repeat(256,1)[np.newaxis],  # 1,2,H,W
        #                        torch.arange(0, 256,out=torch.cuda.FloatTensor()).view(-1,1).repeat(1,256)[np.newaxis]], 0)[np.newaxis]
        #    grid = grid/(255/2.)-1
        #    sampled_img = F.grid_sample(img2, (grid+flow[:,:2]).permute(0,2,3,1))
        #    sampled_img[0,:][self.flow[i:i+1,-1].repeat(3,1,1)!=1]=0.
        #    
        #    cv2.imwrite('tmp/%d-0.png'%i,np.asarray(255*img1[0].permute(1,2,0).cpu())[:,:,::-1])
        #    cv2.imwrite('tmp/%d-1.png'%i,np.asarray(255*sampled_img[0].permute(1,2,0).cpu())[:,:,::-1])
        #    cv2.imwrite('tmp/%d-2.png'%i,np.asarray(255*img2[0].permute(1,2,0).cpu())[:,:,::-1])
        #    #flow1= self.flow[i:i+1]
        #    #flow2 = self.flow[i+local_batch_size//2:i+local_batch_size//2+1]
        #    #flow1 = torch.cat([flow1, torch.zeros_like(flow1)[:,:,:,:1]],-1)
        #    #flow2 = torch.cat([flow2, torch.zeros_like(flow2)[:,:,:,:1]],-1)
        #    self.flow[:,2] = 0
        #    ddlib.write_pfm( 'tmp/%d-3.pfm'%(i),np.asarray(self.flow[i].permute(1,2,0).cpu()))
        #    #ddlib.write_pfm( 'tmp/%d-4.pfm'%(i),np.asarray(flow1[0].detach().cpu()))
        #    ddlib.write_pfm( 'tmp/%d-5.pfm'%(i),np.asarray(self.flow[local_batch_size//2+i].permute(1,2,0).cpu()))
        #    #ddlib.write_pfm( 'tmp/%d-6.pfm'%(i),np.asarray(flow2[0].detach().cpu()))

#        torch.cuda.synchronize()
#        print('before rend + flow time:%.2f'%(time.time()-start_time))
        if opts.debug:
            torch.cuda.synchronize()
            print('after flow rend time:%.2f'%(time.time()-start_time))
              
        # 2) silhouette
        Rmat_mask = Rmat.clone().view(-1,opts.n_mesh,3,3)
        Rmat_mask = torch.cat([Rmat_mask[:,:1], Rmat_mask[:,1:]],1).view(-1,3,3)
        if not opts.finetune:
            Rmat_mask = Rmat_mask.detach()
        #if (not opts.finetune) and self.epoch<5: Rmat_mask = Rmat_mask.detach()
        #Rmat_mask = torch.cat([Rmat_mask[:,:1].detach(), Rmat_mask[:,1:]],1).view(-1,3,3)
        verts_mask = obj_to_cam(pred_v, Rmat_mask, Tmat[:,np.newaxis,:],opts.n_mesh, opts.n_hypo,skin)
        verts_mask = torch.cat([verts_mask,torch.ones_like(verts_mask[:, :, 0:1])], dim=-1)
        verts_mask = pinhole_cam(verts_mask, ppoint, scale)

        # softras
        offset = torch.Tensor( self.renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_mask[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        self.mask_pred = self.renderer_soft.render_mesh(sr.Mesh(verts_pre,faces))[:,-1]

        if opts.emd:
            tmp_render = self.renderer_softtex.render_mesh(sr.Mesh(verts_pre,faces, textures=verts_pre,  texture_type=self.texture_type))
            self.mask_pred = tmp_render[:,-1]
            grid2d_render = tmp_render[:,:2].view(local_batch_size,opts.n_hypo,2,opts.img_size, opts.img_size)
            


        if opts.debug:
            torch.cuda.synchronize()
            print('after sil rend time:%.2f'%(time.time()-start_time))

        if opts.opt_tex=='yes':
            #if self.iters==0:print(self.light_params)
            #self.renderer_softtex.lighting.directionals[0].light_intensity = self.light_params[0]
            #self.renderer_softtex.lighting.directionals[0].light_direction = F.normalize(self.light_params[1:4], p=2,dim=0)
            # 3) texture rendering
            #Rmat_tex = Rmat.clone().view(local_batch_size,opts.n_hypo,opts.n_mesh,3,3)[:,optim_idx].view(-1,3,3)
            #scale_tex = scale[:,optim_idx:optim_idx+1]
            if not opts.finetune and self.epoch<5: 
                Rmat_tex = Rmat.clone().view(-1,opts.n_mesh,3,3)
                Rmat_tex = torch.cat([Rmat_tex[:,:1].detach(), Rmat_tex[:,1:]],1).view(-1,3,3)
            else:
                Rmat_tex = Rmat.clone().view(-1,3,3)
            verts_tex = obj_to_cam(pred_v, Rmat_tex, Tmat[:,np.newaxis,:],opts.n_mesh, opts.n_hypo,skin)
            verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
            verts_tex = pinhole_cam(verts_tex, ppoint, scale)
            #verts_tex = pinhole_cam(verts_tex, ppoint, scale_tex)
            offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
            verts_pre = verts_tex[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
            self.renderer_softtex.rasterizer.background_color = [1,1,1]
            #if not opts.finetune and self.epoch<5: 
            #    verts_pre = verts_pre.detach()
            self.texture_render = self.renderer_softtex.render_mesh(sr.Mesh(verts_pre, faces, textures=tex,  texture_type=self.texture_type)).clone()
            #self.mask_pred = self.texture_render[:,-1]
            self.fgmask_tex = self.texture_render[:,-1]
            self.texture_render = self.texture_render[:,:3]
            img_obs = self.imgs[:]*(self.masks[:]>0).float()[:,None]
            img_rnd = self.texture_render*(self.fgmask_tex)[:,None]
            img_white = 1-(self.masks[:]>0).float()[:,None] + img_obs

            if opts.hrtex>0:
                self.renderer_softtexhr.rasterizer.near=dmin-(dmax-dmin)/2
                self.renderer_softtexhr.rasterizer.far= dmax+(dmax-dmin)/2
                self.renderer_softtexhr.lighting.directionals[0].light_intensity = self.light_params[0]
                self.renderer_softtexhr.lighting.directionals[0].light_direction = F.normalize(self.light_params[1:4], p=2,dim=0)
                texhr = self.texhr[None].repeat(local_batch_size,1,1,1).sigmoid()
                self.texture_render_hr = self.renderer_softtexhr.render_mesh(sr.Mesh(verts_pre.detach(), self.faces[None].repeat(local_batch_size,1,1), textures=texhr,texture_type='surface'))[:,:3].clone()

            #cv2.imwrite('tmp/00.png',255*np.asarray(img_obs[0].permute(1,2,0).detach().cpu())[:,:,::-1])
            #cv2.imwrite('tmp/01.png',255*np.asarray(img_rnd[0].permute(1,2,0).detach().cpu())[:,:,::-1])
         #   verts_pre = verts_tex[:,:,:3].clone(); verts_pre[:,:,2] += 2; verts_pre[:,:,:2] = -1*verts_pre[:,:,:2]
         #   mesh = Meshes(verts=verts_pre, faces=faces,textures= TexturesAtlas(atlas = self.tex.view(1+local_batch_size,-1,self.tex_res,self.tex_res,3)))
         #   #mesh = Meshes(verts=verts_pre, faces=faces,textures=TexturesVertex(verts_features=self.tex))
         #   self.texture_render = self.renderer_softtex(mesh)[:,:,:,:3].permute(0,3,1,2)
            #mesh = sr.Mesh.from_obj('/home/gengshany_google_com/code/SoftRas/data/obj/spot/spot_triangulated.obj',
            #                    load_texture=True, texture_res=5, texture_type='surface')
            #self.texture_render = self.renderer_softtex.render_mesh(mesh)[:,:3].clone()
          #  id_cam = self.cams.clone(); id_cam[:,0]=1; id_cam[:,1:3]=0; id_cam[:,3]=1; id_cam[:,4:]=0
          #  self.texture_render = self.renderer_softtex(verts_tex, faces, id_cam, textures=self.tex.view(17,1280,6,6,1,3).repeat(1,1,1,1,6,1))
            #cv2.imwrite('tmp/00.png',255*np.asarray(self.imgs[0].permute(1,2,0).detach().cpu())[:,:,::-1])
            #cv2.imwrite('tmp/01.png',255*np.asarray(self.texture_render[0].permute(1,2,0).detach().cpu())[:,:,::-1])
            #error = (self.imgs - self.texture_render).norm(2,1)[:-1]
            #cv2.imwrite('tmp/02.png',255*np.asarray(error[0].detach().cpu()))

#        torch.cuda.synchronize()
#        print('before rend + flow + sil + tex time:%.2f'%(time.time()-start_time))              

#        if self.frameid[0]==24:
#            self.part_renderx = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.view(local_batch_size,opts.n_hypo,-1,3)[:1,self.optim_idx].detach(), faces[:1], textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()
        if opts.n_mesh>1 and self.iters==0:
            # part rendering
            if opts.testins:
                self.optim_idx = self.optim_idx[self.dataid[0]]
            colormap = torch.Tensor(citylabs[:opts.n_mesh-1]).cuda() # 5x3
            skin_colors = (skin[self.optim_idx] * colormap[:,None]).sum(0)/256.
            self.part_render = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.view(local_batch_size,opts.n_hypo,-1,3)[:1,self.optim_idx].detach(), faces[:1], textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()

        # depth rendering
        verts_depth = obj_to_cam(pred_v, Rmat.detach(), Tmat[:,np.newaxis,:].detach(),opts.n_mesh, opts.n_hypo,skin.detach())
        verts_depth = 1./verts_depth
        self.depth_render = self.renderer_softdepth.render_mesh(sr.Mesh(verts_pre, faces, textures=verts_depth,  texture_type=self.texture_type)).clone()
        depth_mask = self.depth_render[:,3]
        self.depth_render = self.depth_render[:,2]
        try:
            self.depth_render[depth_mask==0] = self.depth_render[depth_mask>0].min()
        except:
            pass#pdb.set_trace()
        
        
        if opts.debug:
            torch.cuda.synchronize()
            print('after tex+part render time:%.2f'%(time.time()-start_time))

        
        # losses
        ## randomly set a rect as not observed
        #for i in range(local_batch_size):
        #    mask_xy = np.random.randint(50,opts.img_size-50, size=2)
        #    mask_hw =   np.random.randint(10, 100, size=2)
        #    self.occ[i][mask_xy[0] - mask_hw[0]:mask_xy[0] + mask_hw[0], mask_xy[1] - mask_hw[1]:mask_xy[1] + mask_hw[1]] = 0

        # pixel weights
        weight_mask = torch.ones(local_batch_size, opts.img_size, opts.img_size).cuda()
        weight_mask[self.occ==0] = 0

        # 1) mask loss
        mask_pred = self.mask_pred.view(local_batch_size,-1,opts.img_size, opts.img_size)
        self.mask_loss_sub = (mask_pred - self.masks[:,None]).pow(2)

        if not opts.finetune:
            self.mask_loss_sub = 0
            for i in range (5): # 256,128,64,32,16
               # if i==0:ksize=1
               # else:   ksize = 2**i+1  # 1,3,5,9,17
               # diff_img = (kornia.box_blur(mask_pred         , kernel_size=(ksize,ksize),border_type='constant')
               #           - kornia.box_blur(self.masks[:,None], kernel_size=(ksize,ksize),border_type='constant')).pow(2)
        #        cv2.imwrite('/data/gengshay/0.png', np.asarray(F.interpolate(mask_pred         , scale_factor=(0.5)**i,mode='area')[0,0].detach().cpu()*255))
        #        cv2.imwrite('/data/gengshay/1.png', np.asarray(F.interpolate(self.masks[:,None], scale_factor=(0.5)**i,mode='area')[0,0].detach().cpu()*255))
                diff_img = (F.interpolate(mask_pred         , scale_factor=(0.5)**i,mode='area')
                          - F.interpolate(self.masks[:,None], scale_factor=(0.5)**i,mode='area')).pow(2)
                self.mask_loss_sub += F.interpolate(diff_img, mask_pred.shape[2:4])
            self.mask_loss_sub *= 0.2
        
        tmplist = torch.zeros(local_batch_size, opts.n_hypo).cuda()
        for i in range(local_batch_size):
            for j in range(opts.n_hypo):
                #tmplist[i,j] = self.mask_loss_sub[i,j][self.occ[i]!=0].mean()
                tmplist[i,j] = (self.mask_loss_sub[i,j]*weight_mask).mean()
                if opts.emd:
                    from geomloss import SamplesLoss
                    samp_loss = SamplesLoss("sinkhorn", p=2, blur=0.1, reach=0.3)
                    meshgrid = torch.Tensor(np.meshgrid(range(opts.img_size), range(opts.img_size))).cuda()
                    idx_sample = self.masks[i]>0
                    gt_samplex = torch.masked_select(meshgrid[0,:,:], idx_sample)
                    gt_sampley = torch.masked_select(meshgrid[1,:,:], idx_sample)
                    gt_sample = torch.stack([gt_samplex, gt_sampley], -1) / (opts.img_size/2.) - 1 # range -1 to 1

                    idx_sample = torch.bernoulli(mask_pred[i,j]*(self.occ[i]!=0).float()).bool()
                    render_samplex = torch.masked_select(grid2d_render[i,j,0,:,:], idx_sample)
                    render_sampley =-torch.masked_select(grid2d_render[i,j,1,:,:], idx_sample)
                    render_sample = torch.stack([render_samplex, render_sampley], -1)
                    render_sample = render_sample[torch.randperm(render_sample.shape[0])[:gt_sample.shape[0]]]
                    tmplist[i,j] = samp_loss(render_sample, gt_sample) * 20
        self.mask_loss_sub = opts.mask_loss_wt * tmplist
        self.mask_loss = self.mask_loss_sub.mean()  # get rid of invalid pixels (out of border)
        self.total_loss = self.mask_loss.clone()

        # 2) flow loss
        #import utils.dydepth as ddlib
        #for i in range(local_batch_size):
        #    cv2.imwrite('tmp/%d-0.png'%i,255*np.asarray(self.masks[i].detach().cpu()))
        #    cv2.imwrite('tmp/%d-1.png'%i,255*np.asarray(self.fgmask_flow[i].detach().cpu()))
        #    flow1= self.flow_rd[i:i+1]
        #    flow1 = torch.cat([flow1, torch.zeros_like(flow1)[:,:,:,:1]],-1)
        #    flow2 = self.flow[i].permute(1,2,0).clone()
        #    flow2 = torch.cat([flow2[:,:,:2],torch.zeros_like(flow2)[:,:,:1]],-1) 
        #    ddlib.write_pfm( 'tmp/%d-2.pfm'%(i),np.asarray(flow2.cpu()))
        #    ddlib.write_pfm( 'tmp/%d-3.pfm'%(i),np.asarray(flow1[0].detach().cpu()))
        flow_rd = self.flow_rd.view(local_batch_size,-1,opts.img_size, opts.img_size,2)
        # overlapped flow    
        mask = (~self.bgmask).view(local_batch_size,-1,opts.img_size, opts.img_size) & ((self.occ!=0)[:,None] &  (self.masks[:]>0) [:,None]).repeat(1,opts.n_hypo,1,1)
        self.flow_rd_map = torch.norm((flow_rd-self.flow[:,None,:2].permute(0,1,3,4,2)),2,-1)
        ## all obj pixels
        #mask = ((self.occ!=0)[:,None] & (self.flow[:,-1]==1) [:,None]).repeat(1,opts.n_hypo,1,1)
        #self.flow_rd_map = 20 * torch.norm( flow_rd * self.fgmask_flow.view(local_batch_size,-1,opts.img_size, opts.img_size,1)-
        #                             self.flow[:,None,:2].permute(0,1,3,4,2) * (self.masks[:]>0).float()[:,None,:,:,None]   ,2,-1)

        #if not opts.finetune:
        #    self.flow_rd_map = 0
        #    for i in range (5): # 256,128,64,32,16
        #       # if i==0:ksize=1
        #       # else:   ksize = 2**i+1  # 1,3,5,9,17
        #       # diff_imgx = kornia.box_blur(flow_rd[:,:,:,:,0],  kernel_size=(ksize,ksize),border_type='constant')-\
        #       #             kornia.box_blur(self.flow[:,None,0], kernel_size=(ksize,ksize),border_type='constant')
        #       # diff_imgy = kornia.box_blur(flow_rd[:,:,:,:,1],  kernel_size=(ksize,ksize),border_type='constant')-\
        #       #             kornia.box_blur(self.flow[:,None,1], kernel_size=(ksize,ksize),border_type='constant')
        #       # diff_img = torch.stack([diff_imgx, diff_imgy],-1).norm(2,-1)
        #        diff_img = torch.norm((F.interpolate(flow_rd,                                 scale_factor=((0.5)**i,(0.5)**i,1),mode='area')-
        #                               F.interpolate(self.flow[:,None,:2].permute(0,1,3,4,2), scale_factor=((0.5)**i,(0.5)**i,1),mode='area')),2,-1)
        #       # diff_img = diff_img + 0.1*(1 - F.cosine_similarity(F.interpolate(flow_rd,                                 scale_factor=((0.5)**i,(0.5)**i,1),mode='area'),
        #       #                                     F.interpolate(self.flow[:,None,:2].permute(0,1,3,4,2), scale_factor=((0.5)**i,(0.5)**i,1),mode='area'),-1))
        #        self.flow_rd_map += F.interpolate(diff_img, flow_rd.shape[2:4])
        #    self.flow_rd_map *= 0.2

        self.vis_mask = mask.clone()
        weights_flow = (-self.occ).sigmoid()[:,None].repeat(1,opts.n_hypo,1,1)
        weights_flow = weights_flow / weights_flow[mask].mean()
        self.flow_rd_map = self.flow_rd_map * weights_flow
    
        tmplist = torch.zeros(local_batch_size, opts.n_hypo).cuda()
        for i in range(local_batch_size):
            for j in range(opts.n_hypo):
                tmplist[i,j] = self.flow_rd_map[i,j][mask[i,j]].mean()
                if mask[i,j].sum()==0: tmplist[i,j]=0
        #if opts.n_hypo>1:
        self.flow_rd_loss_sub = 0.5*tmplist
        #else:
        #    self.flow_rd_loss_sub = 10*tmplist
    
        #self.flow_rd_loss_sub = torch.stack([self.flow_rd_map[:,i][mask[:,i]].mean() for i in range(opts.n_hypo)])
    
        self.flow_rd_loss = self.flow_rd_loss_sub.mean()
        self.total_loss += self.flow_rd_loss
        #if (opts.finetune) and self.epoch<3:
        #    self.total_loss += self.flow_rd_loss * 10
    
        # exp
        flowrdmask = (~self.bgmask).view(local_batch_size,-1,opts.img_size, opts.img_size)
        flowobmask = ((self.occ!=0)[:,None] &  (self.masks[:]>0) [:,None]).repeat(1,opts.n_hypo,1,1)
        exp_rd , expmask_rd  = affine(self.flow_rd.view(-1,opts.img_size, opts.img_size, 2).permute(0,3,1,2),flowrdmask,5) # bsxhp, 1, h, w
        exp_obs, expmask_obs = affine(self.flow[:,:2],flowobmask,5) # bs, 1, h,w
        expmask = mask & expmask_obs[:,None] * expmask_rd[:,None]
        exp_obs_vis = exp_obs.clone(); exp_obs_vis[~expmask]=1 # for vis
        exp_rd_vis = exp_rd.clone(); exp_rd_vis[~expmask]=1 # for vis
        self.exp_rd_map = (exp_rd.view(-1,opts.n_hypo,opts.img_size, opts.img_size).log() - exp_obs.log()).abs()
        self.exp_rd_map  = self.exp_rd_map * weights_flow
        tmplist = torch.zeros(local_batch_size, opts.n_hypo).cuda()
        for i in range(local_batch_size):
            for j in range(opts.n_hypo):
                tmplist[i,j] += self.exp_rd_map[i,j][expmask[i,j]].mean()
                if expmask[i,j].sum()==0: tmplist[i,j]=0
        self.exp_rd_loss_sub = 0.5*tmplist
        self.exp_rd_loss = self.exp_rd_loss_sub.mean()
        if not opts.finetune:
            self.total_loss += 0.2*self.exp_rd_loss


        
        # 3) texture loss
        if opts.opt_tex=='yes':
            imgobs_rep = img_obs[:,None].repeat(1,opts.n_hypo,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            imgwhite_rep = img_white[:,None].repeat(1,opts.n_hypo,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            obspair = torch.cat([imgobs_rep, imgwhite_rep],0) 
            rndpair = torch.cat([img_rnd, self.texture_render],0) 
            #self.texture_loss = 2*(img_obs[:,None] - img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)).abs().mean(2)[self.occ[:,None].repeat(1,opts.n_hypo,1,1)!=0].mean()  
            #self.texture_loss +=2*(img_white[:,None] - self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)).abs().mean(2)[self.occ[:,None].repeat(1,opts.n_hypo,1,1)!=0].mean()  
            #self.texture_loss += 0.005*self.ptex_loss.forward_pair(2*imgobs_rep-1, 2*img_rnd-1).mean()  
            #self.texture_loss += 0.005*self.ptex_loss.forward_pair(2*imgwhite_rep-1, 2*self.texture_render-1).mean() 
    
            tmplist = torch.zeros(local_batch_size, opts.n_hypo).cuda()
            pyr_img = img_obs
            for i in range(local_batch_size):
                for j in range(opts.n_hypo):
                    #tmplist[i,j] += (img_obs[i] - img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)[self.occ[i]!=0].mean()  
                    #tmplist[i,j] += (img_white[i] - self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)[self.occ[i]!=0].mean()  
                    tmplist[i,j] += ((img_obs[i] - img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)*weight_mask[i]).mean()  
                    tmplist[i,j] += ((img_white[i] - self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)*weight_mask[i]).mean()  

                    if False:# not opts.finetune:
                        tmplist[i,j] = 0
                        for k in range(5):
                           # if i==0:ksize=1
                           # else:   ksize = 2**i+1  # 1,3,5,9,17
                           # diff_img = (kornia.box_blur(img_obs[i:i+1]                                                             ,kernel_size=(ksize,ksize),border_type='constant')- 
                           #             kornia.box_blur(img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i:i+1,j],kernel_size=(ksize,ksize),border_type='constant')
                           #             ).abs()
                           # tmplist[i,j] += diff_img[0].mean(0)[self.occ[i]!=0].mean()
                           # diff_img = (kornia.box_blur(img_white[i:i+1]                                                                       ,kernel_size=(ksize,ksize),border_type='constant')- 
                           #             kornia.box_blur(self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i:i+1,j],kernel_size=(ksize,ksize),border_type='constant')
                           #             ).abs()
                           # tmplist[i,j] += diff_img[0].mean(0)[self.occ[i]!=0].mean()
                            diff_img = (F.interpolate(img_obs[i]                                                             ,scale_factor=(0.5)**k,mode='area')- 
                                    F.interpolate(img_rnd.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j],scale_factor=(0.5)**k,mode='area')
                                    ).abs()
                            tmplist[i,j] += F.interpolate(diff_img[None], img_obs.shape[2:4])[0].mean(0)[self.occ[i]!=0].mean()
                            diff_img = (F.interpolate(img_white[i]                                                             ,scale_factor=(0.5)**k,mode='area')- 
                                    F.interpolate(self.texture_render.view(local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j],scale_factor=(0.5)**k,mode='area')
                                    ).abs()
                            tmplist[i,j] += F.interpolate(diff_img[None], img_obs.shape[2:4])[0].mean(0)[self.occ[i]!=0].mean()
                        tmplist[i,j] *= 0.2*2*opts.l1tex_wt
    
            if (opts.finetune) and self.epoch<3 and self.reinit_bones:
                percept_loss = self.ptex_loss.forward_pair(2*obspair-1, 2*rndpair-1, reweight=True)
            else:
                percept_loss = self.ptex_loss.forward_pair(2*obspair-1, 2*rndpair-1)
            #tmplist =  0.005*percept_loss.view(2,-1).sum(0).view(local_batch_size,opts.n_hypo)
            tmplist +=  0.005*percept_loss.view(2,-1).sum(0).view(local_batch_size,opts.n_hypo)
            self.texture_loss_sub = 0.25*tmplist
            self.texture_loss = self.texture_loss_sub.mean()
    
            self.total_loss += self.texture_loss

        if opts.hrtex>0:
            self.total_loss += 2*(img_obs - self.texture_render_hr).abs().mean(1)[self.occ!=0].mean()
            self.total_loss += 0.005*self.ptex_loss.forward_pair(2*img_obs-1, 2*self.texture_render_hr-1).mean()  
        
        # 4) shape smoothness/symmetry
        factor=int(opts.n_faces)/1280
        if not opts.finetune:
            #factor = 1 # possibly related to symmetry loss?
            factor = reg_decay(self.epoch, opts.num_epochs, 0.1, 1)
        else: 
            factor = reg_decay(self.epoch, opts.num_epochs, 0.05, 0.5)
   
        self.triangle_loss_sub = torch.zeros(2*opts.n_hypo*local_batch_size//2).cuda()
        for idx in range(local_batch_size): 
            predv_batch = self.deform_v[idx*opts.n_hypo:(idx+1)*opts.n_hypo]
            self.triangle_loss_sub[idx*opts.n_hypo:(idx+1)*opts.n_hypo] = factor*opts.triangle_reg_wt*self.triangle_loss_fn_sr[self.dataid[idx]](predv_batch)*(4**opts.subdivide)/64.
            self.triangle_loss_sub[idx*opts.n_hypo:(idx+1)*opts.n_hypo] +=factor*0.1*opts.triangle_reg_wt*self.flatten_loss[self.dataid[idx]](predv_batch)*(2**opts.subdivide/8.0)
        self.triangle_loss_sub = self.triangle_loss_sub.view(local_batch_size,opts.n_hypo)
        self.triangle_loss = self.triangle_loss_sub.mean()
        self.total_loss += self.triangle_loss
                
        if opts.debug:
            torch.cuda.synchronize()
            print('after sym loss time:%.2f'%(time.time()-start_time))

        # 5) shape deformation loss
        if opts.n_mesh>1:
            # bones
            self.bone_rot_l1 =  compute_geodesic_distance_from_two_matrices(
                        quat.view(-1,opts.n_hypo,opts.n_mesh,9)[:,:,1:].reshape(-1,3,3), 
             torch.eye(3).cuda().repeat(local_batch_size*opts.n_hypo*(opts.n_mesh-1),1,1)).mean() # small rotation
            self.bone_trans_l1 = torch.cat([trans,depth],1).view(-1,opts.n_hypo,opts.n_mesh,3)[:,:,1:].abs().mean()
            if not opts.finetune:
                #factor=1
                #factor=0.01
                factor = reg_decay(self.epoch, opts.num_epochs, 0.1, 1.0)
            else: 
                factor=0.001
                #factor=0.1
                #factor = reg_decay(self.epoch, opts.num_epochs, 0.01, 0.1)
            #if opts.n_hypo==1 or self.epoch>10: factor = 0.1
            self.lmotion_loss_sub = factor*(self.deform_v - pred_v).norm(2,-1).mean(-1).view(local_batch_size,opts.n_hypo)
            self.lmotion_loss = self.lmotion_loss_sub.mean()
            #self.lmotion_loss = 1*(self.deform_v.detach().mean(0) - pred_v[0]).norm(2,-1).mean() # mean of deformed vertices is mean shape
            self.total_loss += self.lmotion_loss * opts.l1_wt
    
            # skins
            shape_samp = pytorch3d.ops.sample_points_from_meshes(pytorch3d.structures.meshes.Meshes(verts=pred_v, faces=faces), 1000, return_normals=False).detach()
            from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
            samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            self.skin_ent_loss = samploss(ctl_ts[:,:,:,0], shape_samp).mean()  # By default, use constant weights = 1/number of samples
            #self.skin_ent_loss = self.entropy_loss(skin)
            if opts.finetune:
                factor = 0.1
            else:
                factor = 0.
            self.total_loss += self.skin_ent_loss * factor
            #self.arap_loss = torch.zeros(local_batch_size//2).cuda()
            #for i in range(local_batch_size//2):
            #    predv_batch1 = self.deform_v[:local_batch_size//2*opts.n_hypo][i*opts.n_hypo:(i+1)*opts.n_hypo]
            #    predv_batch2 = self.deform_v[local_batch_size//2*opts.n_hypo:][i*opts.n_hypo:(i+1)*opts.n_hypo]
            #    self.arap_loss[i] = self.arap_loss_fn[self.dataid[i]](predv_batch1, predv_batch2).mean()*(4**opts.subdivide)/64.
            self.arap_loss = torch.zeros(local_batch_size).cuda()
            from nnutils.loss_utils import mesh_area
            for i in range(local_batch_size):
                self.arap_loss[i] = self.arap_loss_fn[self.dataid[i]](self.deform_v[i:i+1], pred_v[i:i+1]).mean()*(4**opts.subdivide)/64.
                self.arap_loss[i] += (mesh_area(self.deform_v[i], faces[i]) - mesh_area(pred_v[i], faces[i])).abs().mean()
            self.arap_loss = self.arap_loss.mean()
            self.total_loss += opts.arap_wt * self.arap_loss * 10

            # self-intersection
            if opts.si_wt>0:
                triangles = [] 
                for i in range(local_batch_size):
                    triangles.append( self.deform_v[i].view([-1, 3])[faces[i].long()][None] )
                triangles=torch.cat(triangles,0)
                collision_idxs = self.search_tree(triangles)
                self.pen_loss = opts.si_wt *self.pen_distance(triangles, collision_idxs).mean()
                self.total_loss +=  self.pen_loss

        ##### common losses

        # 7) camera loss
        if opts.use_gtpose:
            self.cam_loss = compute_geodesic_distance_from_two_matrices(quat.view(-1,3,3), quat_pred.view(-1,3,3)).mean()
            self.cam_loss += (scale_pred - scale).abs().mean()
            self.cam_loss += (trans_pred - trans).abs().mean()
            self.cam_loss += (depth_pred - depth).abs().mean()
            self.cam_loss += (ppoint_pred - ppoint).abs().mean()
            self.cam_loss = opts.cam_loss_wt * self.cam_loss
        else:
            self.rotg_sm_sub = compute_geodesic_distance_from_two_matrices(quat.view(-1,opts.n_hypo,opts.n_mesh,9)[:local_batch_size//2,:].view(-1,3,3),
                                                                            quat.view(-1,opts.n_hypo,opts.n_mesh,9)[local_batch_size//2:,:].view(-1,3,3)).view(-1,opts.n_hypo,opts.n_mesh)
            self.cam_loss =  0.01*self.rotg_sm_sub.mean()
            #self.cam_loss =  0.001*self.rotg_sm_sub.mean()
            #self.cam_loss += 0.01* (scale[:scale.shape[0]//2] - scale[scale.shape[0]//2:]).abs().mean()
            #self.cam_loss += 0.01*(Tmat.view(-1,opts.n_hypo,opts.n_mesh,3)[:local_batch_size//2,:,:1] - 
            #                 Tmat.view(-1,opts.n_hypo,opts.n_mesh,3)[local_batch_size//2:,:,:1]).norm(2,-1).mean()
            if opts.n_mesh>1:
                self.cam_loss += 0.01*(trans.view(-1,opts.n_hypo,opts.n_mesh,2)[:local_batch_size//2,:,1:] - 
                              trans.view(-1,opts.n_hypo,opts.n_mesh,2)[local_batch_size//2:,:,1:]).abs().mean()
                self.cam_loss += 0.01*(depth.view(-1,opts.n_hypo,opts.n_mesh,1)[:local_batch_size//2,:,1:] - 
                              depth.view(-1,opts.n_hypo,opts.n_mesh,1)[local_batch_size//2:,:,1:]).abs().mean()
            ## noise loss
            #noisediff = compute_geodesic_distance_from_two_matrices(quatpre.view(-1,opts.n_hypo,opts.n_mesh,9).view(-1,3,3),
            #                                                  quat.detach().view(-1,opts.n_hypo,opts.n_mesh,9).view(-1,3,3)).view(-1,opts.n_hypo,opts.n_mesh).mean(-1)
            #correctess = self.flow_rd_loss_sub
            #self.cam_loss += 1 * (noisediff * (-10*correctess.detach()).exp()).mean()
            if opts.ibraug and opts.local_rank==0:
                import cv2
                for i in range(self.texture_render.shape[0]):
                    #cv2.imwrite('./tmp/%03d.png'%i, np.asarray(img_rnd_input[i,:3].permute(1,2,0).cpu())[:,:,::-1]*255)
                    #cv2.imwrite('./tmp/%03d-img.png'%i, np.asarray(img[i,:3].permute(1,2,0).cpu())[:,:,::-1]*255)
                    try:
                        vis = torch.cat([mvrnd[i,:3], mvrnd_pred[i,:3], self.texture_render[i,:3].detach(), self.imgs[i,:3]],2)
                        vis = np.asarray(vis.permute(1,2,0).cpu())[:,:,::-1].copy()*255
                        vis = vis.astype(np.uint8)
                        cv2.putText(vis, '%d:%.5f'%(self.iters, geo_aug_loss[i]), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                        cv2.imwrite('./tmp/%03d.png'%i, vis)
                    except: pass

            # rotg loss
            rotg_gt = self.rotgf
            fac = 0.02
#            if opts.debug: fac = 0.
            self.cam_loss += fac*loss_utils.quat_loss_geodesic(rotg.view(-1,opts.n_mesh,4)[:,:1].view(-1,4), rotg_gt[:,:1].view(-1,4),).mean()
            #self.cam_loss += 0.2*loss_utils.quat_loss_geodesic(rotg.view(-1,opts.n_mesh,4)[:,:1].view(-1,4), rotg_gt[:,:1].view(-1,4),).mean()
            #self.cam_loss += 2*loss_utils.quat_loss_geodesic(rotg.view(-1,opts.n_mesh,4)[:,:1].view(-1,4), rotg_gt[:,:,:1].view(-1,4),).mean()

        #if opts.debug:
        #    view_v = obj_to_cam(pred_v.detach(), Rmat.view(-1,3,3), Tmat[:,np.newaxis,:],\
        #                        opts.n_mesh, opts.n_hypo,skin.detach(),tocam=False)
        #    view_v = view_v.view(local_batch_size,3*self.num_verts).clone()
        #    self.trajs[self.frameid] = view_v.detach()
        #    ppoint_ori = ((ppoint_raw.view(-1,2) + self.cams[:,:1]  * self.pp /(opts.img_size/2.) + 1) / self.cams[:,:1]).view(-1,2)
        #    self.trajs_pp[self.frameid] = ppoint_ori.detach()
        #    self.confs[self.frameid] = (-100*self.flow_rd_loss_sub).exp().view(-1).detach()
        #    #if opts.local_rank==0:pdb.set_trace()
        #    
        #    # TxN3->TxN3
        #    self.velo_loss = 0

        #    dframe=4
        #    ftraj = torch.zeros_like(self.trajs)
        #    fconf = torch.zeros_like(self.confs)
        #    btraj = torch.zeros_like(self.trajs)
        #    bconf = torch.zeros_like(self.confs)

        #    # forward backward
        #    ftraj[2*dframe:] = self.trajs[dframe:-dframe]*2  - self.trajs[:-2*dframe] # 0,0,xxx
        #    fconf[2*dframe:] = self.confs[dframe:-dframe]**2 * self.confs[:-2*dframe]
        #    btraj[:-2*dframe] = self.trajs[dframe:-dframe]*2  - self.trajs[2*dframe:] # xxx,0,0
        #    bconf[:-2*dframe] = self.confs[dframe:-dframe]**2 * self.confs[2*dframe:]
        #    self.velo_loss +=( (ftraj[self.frameid] - view_v).view(local_batch_size,-1,3).norm(2,-1)*fconf[self.frameid,None]).mean()*0 +\
        #                     ( (btraj[self.frameid] - view_v).view(local_batch_size,-1,3).norm(2,-1)*bconf[self.frameid,None]).mean()

        #    #exp_ppoint = (self.trajs_pp * self.confs[:,None]/self.confs.sum()).sum(0)[None]
        #    #if opts.local_rank==0: 
        #    #    print(exp_ppoint)
        #    #    print(ppoint_ori.view(-1,2))
        #    #self.velo_loss = (ppoint_ori - exp_ppoint).abs().mean() * 0.05

        #    decay_factor = reg_decay(self.epoch, opts.num_epochs, 0.1, 1)
        #    self.cam_loss += self.velo_loss * 10 * decay_factor
        #    #self.cam_loss += self.velo_loss * 0.1
            

        self.total_loss += self.cam_loss
        try:        
            self.total_loss += 0.1*ent_loss
            print(ent_loss)
        except: pass
        try:        
            geo_aug_loss = F.relu(geo_aug_loss-0.5).mean()
            if self.epoch>5:
                self.total_loss += 0.1*geo_aug_loss
        except: pass

        
        # 9) aux losses
        # pull far away from the camera center
        self.total_loss += 0.02*F.relu(2-Tmat.view(-1, 1, opts.n_mesh, 3)[:,:,:1,-1]).mean()
        #self.total_loss += 1./ctl_ts[0,:,:,0].svd()[1].sum()
        

        # texture and shape volume losses
        if opts.ptex:
            self.nerf_tex_loss = (texture_sampled[:local_batch_size//2*opts.n_hypo] - texture_sampled[local_batch_size//2*opts.n_hypo:])[:,:,:3].norm(1,-1).mean()
            #self.total_loss += 0.2*self.nerf_tex_loss * 0.1
            self.total_loss += 0.2*self.nerf_tex_loss * 0.01

        if opts.debug:
            torch.cuda.synchronize()
            print('forward time:%.2f'%(time.time()-start_time))

        aux_output={}
        aux_output['flow_rd_map'] = self.flow_rd_map
        aux_output['flow_rd'] = self.flow_rd
        aux_output['vis_mask'] = self.vis_mask
        aux_output['mask_pred'] = self.mask_pred
        aux_output['total_loss'] = self.total_loss
        aux_output['mask_loss'] = self.mask_loss
        aux_output['texture_loss'] = self.texture_loss
        aux_output['flow_rd_loss'] = self.flow_rd_loss
        aux_output['skin_ent_loss'] = self.skin_ent_loss
        aux_output['arap_loss'] = self.arap_loss
        aux_output['rotg_sm_sub_loss'] = self.rotg_sm_sub.mean()
#        if opts.debug:
#            aux_output['velo_loss'] = self.velo_loss

        try:
            aux_output['match_loss'] = self.match_loss
        except: pass
        
        try:
            aux_output['csm_loss'] = self.csm_loss
        except: pass
        
        try:
            aux_output['flow_s_obs'] = flow_obs
            aux_output['flow_s'] = flow_s
            aux_output['mask_cost'] = mask_cost
            aux_output['score_s'] = score_s
        except: pass
        try:
            aux_output['flow_s_loss'] = self.flow_s_loss
        except: pass
        try:
            aux_output['rank_loss'] = self.rank_loss
        except: pass
        aux_output['triangle_loss'] = self.triangle_loss
        if opts.n_mesh>1:
            aux_output['lmotion_loss'] = self.lmotion_loss
        if opts.si_wt>0:
            aux_output['pen_loss'] = self.pen_loss
        if opts.ptex:
            aux_output['nerf_tex_loss'] = self.nerf_tex_loss
        try: aux_output['l1_deform_loss'] = self.l1_deform_loss
        except: pass
        try: aux_output['geo_aug_loss'] = geo_aug_loss
        except: pass
        
        if opts.testins:
            aux_output['current_nscore'] = torch.zeros(self.shape_code_fix.shape[0], opts.n_hypo).cuda()
            for i in range(local_batch_size):
                aux_output['current_nscore'][self.dataid[i]] += self.texture_loss_sub[i] + self.flow_rd_loss_sub[i] + self.mask_loss_sub[i]
        else:
            aux_output['current_nscore'] = self.texture_loss_sub.mean(0) + self.flow_rd_loss_sub.mean(0) + self.mask_loss_sub.mean(0)
        aux_output['exp_rd']  = exp_rd_vis
        aux_output['exp_obs'] = exp_obs_vis
        aux_output['exp_rd_loss'] = self.exp_rd_loss
        if opts.n_hypo > 1:
            for ihp in range(opts.n_hypo):
                aux_output['mask_hypo_%d'%(ihp)] = self.mask_loss_sub[:,ihp].mean()
                aux_output['flow_hypo_%d'%(ihp)] = self.flow_rd_loss_sub[:,ihp].mean()
                aux_output['tex_hypo_%d'%(ihp)] = self.texture_loss_sub[:,ihp].mean()
        try:
            aux_output['texture_render'] = self.texture_render
            aux_output['ctl_proj'] = self.ctl_proj
            aux_output['joint_proj'] = self.joint_proj
            aux_output['part_render'] = self.part_render
            aux_output['cost_rd'] = self.cost_rd
            aux_output['texture_render_hr'] = self.texture_render_hr
            aux_output['orth_loss'] = self.orth_loss
        except:pass
        try:
            aux_output['nerf_shape_loss'] = self.nerf_shape_loss
        except: pass
        return self.total_loss, aux_output

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip.cuda() * V[-self.num_sym:]
                verts=torch.cat([V, V_left], 0)
                #if self.training:
                #    verts[:self.num_indept] = torch.cat((verts[:self.num_indept,0:1].detach(),verts[:self.num_indept,1:]),1) 
                #if not self.training:
                verts[:self.num_indept,self.opts.symidx]=0
                return verts 
            else:
                pdb.set_trace()
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                verts = torch.cat([V, V_left], 1)
                verts[:,:self.num_indept, 0] = 0
                return verts
        else:
            return V
    
    def symmetrize_color(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        # No batch
        if self.symmetric:
            V_left = V[-self.num_sym:]
            verts=torch.cat([V, V_left], 0)
        else: verts = V
        return verts 

    def symmetrize_color_faces(self, tex_pred):
        if self.symmetric:
            tex_left = tex_pred[-self.num_sym_faces:]
            tex = torch.cat([tex_pred, tex_left], 0)
        else: tex = tex_pred
        return tex
    
    def get_mean_shape(self,local_batch_size):
        mean_v = torch.cat([self.symmetrize(i)[None] for i in self.mean_v],0)
        tex = torch.cat([self.symmetrize_color(i)[None] for i in self.tex],0)
        faces=self.faces

        mean_v =mean_v[None].repeat(local_batch_size,1,1,1).view(local_batch_size*mean_v.shape[0],-1,3)
        faces =  faces[None].repeat(local_batch_size,1,1,1).view(local_batch_size*faces.shape[0],-1,3)
        tex =      tex[None].repeat(local_batch_size,1,1,1).sigmoid().view(local_batch_size*tex.shape[0],-1,3)
        return mean_v, tex, faces

