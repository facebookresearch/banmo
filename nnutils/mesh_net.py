"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from collections import defaultdict
import os
import os.path as osp
import sys
sys.path.insert(0, 'third_party')
import cv2, numpy as np, time, torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh, pytorch3d, pytorch3d.loss, pdb

from ext_utils import mesh
from ext_utils import geometry as geom_utils
from ext_nnutils.nerf import Embedding, NeRF
import kornia, configparser, soft_renderer as sr
from nnutils.geom_utils import K2mat, Kmatinv, K2inv, raycast, sample_xy
from nnutils.rendering import render_rays

flags.DEFINE_string('rtk_path', '', 'path to rtk files')
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')
flags.DEFINE_float('random_geo', 1, 'Random geometric augmentation')
flags.DEFINE_string('config_name', 'template', 'name of the test data config file')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_float('padding_frac', 0.05, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_integer('num_kps', 15, 'The dataloader should override these.')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_string('logname', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_boolean('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentu')
flags.DEFINE_integer('batch_size', 1, 'size of minibatches')
flags.DEFINE_string('checkpoint_dir', 'logdir/', 'Root directory for output files')
flags.DEFINE_string('model_path', '', 'load model path')
flags.DEFINE_boolean('freeze_shape', False, 'whether to load an initial shape and freeze it')
flags.DEFINE_boolean('finetune', False, 'whether to load the full model and finetune it')
flags.DEFINE_boolean('upgrade_mesh', False, 'whether to subdivide mesh')
flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 10000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 1, 'save model every k epochs')
flags.DEFINE_string('n_faces', '1280', 'number of faces for remeshing')
flags.DEFINE_integer('display_freq', 100, 'visuals logging frequency')
flags.DEFINE_boolean('display_visuals', False, 'whether to display images')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
flags.DEFINE_boolean('plot_scalars', False, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_integer('display_id', 1, 'Display Id')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_integer('display_port', 8097, 'Display port')
flags.DEFINE_integer('display_single_pane_ncols', 0, 'if positive, display all images in a single visdom web panel with certain number of images per row.')
flags.DEFINE_boolean('reg3d', False, 'register in 3d')
flags.DEFINE_boolean('emd', False, 'deubg')
flags.DEFINE_boolean('debug', False, 'deubg')
flags.DEFINE_boolean('usekp', False, 'deubg: use kp')
flags.DEFINE_boolean('ibraug', False, 'deubg: use image based rendering augmentation')
flags.DEFINE_boolean('freeze_weights', False, 'Freeze network weights')
flags.DEFINE_boolean('noise', True, 'Add random noise to pose')
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_boolean('symmetric_loss', True, 'Use symmetric loss or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')
flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')
flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')
flags.DEFINE_integer('symidx', 0, 'symmetry index: 0-x 1-y 2-z')
flags.DEFINE_integer('n_mesh', 1, 'num of meshes')
flags.DEFINE_integer('n_hypo', 1, 'num of hypothesis cameras')
flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')
flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')
flags.DEFINE_boolean('volume', False, 'If true, use implicit func + volumetric rendering')
flags.DEFINE_boolean('ptex', False, 'If true, use implicit func for volume texture')
flags.DEFINE_boolean('pshape', False, 'If true, use implicit func for shape variation volume')
flags.DEFINE_string('load_mesh', '', 'load a tempalte mesh')
flags.DEFINE_string('opt_tex', 'no', 'optimize texture')
flags.DEFINE_float('rscale', 1.0, 'scale random variance')
flags.DEFINE_float('l1tex_wt', 1.0, 'weight of l1 texture')
flags.DEFINE_float('arap_wt', 1.0, 'weight of arap loss')
flags.DEFINE_float('l1_wt', 1.0, 'weight of arap loss')
flags.DEFINE_float('si_wt', 0.0, 'weight of self intersection loss')
flags.DEFINE_float('hrtex', 0.0, 'high-res texture')
flags.DEFINE_bool('self_augment', False, 'whether to self-augment the model')
flags.DEFINE_bool('testins', False, 'whether to test multi instance hypothesis')
flags.DEFINE_bool('catemodel', False, 'learn a category model')
flags.DEFINE_bool('ptime', False, 'positional encoding for time dimension')
flags.DEFINE_bool('cnnpp', False, 'cnn principle points')
flags.DEFINE_bool('stop_csm', False, 'stop using csm loss')
flags.DEFINE_bool('nothuman', False, 'using animal model')
# nerf
flags.DEFINE_integer('chunk', 32*1024, 'chunk size to split the input to avoid OOM')
flags.DEFINE_integer('N_importance', 0, 'number of additional fine samples')
flags.DEFINE_float('perturb',   1.0, 'factor to perturb depth sampling points')
flags.DEFINE_float('noise_std', 1.0, 'std dev of noise added to regularize sigma')
flags.DEFINE_bool('flowbw', False, 'use backward warping 3d flow')
flags.DEFINE_bool('lbs', False, 'use lbs for backward warping 3d flow')
flags.DEFINE_integer('sample_grid3d', 128, 'resolution for mesh extraction from nerf')

class v2s_net(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        super(v2s_net, self).__init__()
        self.opts = opts
        self.device = torch.device("cuda:%d"%opts.local_rank)

        # set nerf model
        self.nerf_coarse = NeRF()
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number

        # set dnerf model
        num_bones = 35
        self.num_t_feat = 7*num_bones
        self.embedding_time = nn.Embedding(100, self.num_t_feat) ##TODO change 15
        self.nerf_flowbw = NeRF(in_channels_xyz=63+self.num_t_feat, 
                                in_channels_dir=0)
        
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}
        self.nerf_models= {'coarse':self.nerf_coarse}
        if opts.flowbw:
            self.nerf_models['flowbw'] = self.nerf_flowbw
            self.embeddings['time'] = self.embedding_time
        elif opts.lbs:
            center =  torch.rand(num_bones,3).to(self.device)
            center = center-0.5
            orient =  torch.Tensor([[1,0,0,0]]).to(self.device)
            orient = orient.repeat(num_bones,1)
            scale = torch.zeros(num_bones,3).to(self.device)

            self.bones = nn.Parameter(torch.cat([center, orient, scale],-1))
            self.nerf_models['bones'] = self.bones
            self.embeddings['time'] = self.embedding_time

        if opts.N_importance>0:
            self.nerf_fine = NeRF()
            self.nerf_models['fine'] = self.nerf_fine
        
        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        
    def nerf_render(self, rtk, kaug, frameid, img_size, nsample=256, ndepth=128):
        opts=self.opts
        Rmat = rtk[:,:3,:3]
        Tmat = rtk[:,:3,3]
        Kmat = K2mat(rtk[:,3,:])
        Kaug = K2inv(kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))
        bs = Kinv.shape[0]

        rand_inds, xys = sample_xy(img_size, bs, nsample, self.device, 
                                   return_all= not(self.training))

        rays = raycast(xys, Rmat, Tmat, Kinv, bound=1.5)

        # render rays
        frameid = frameid[:,None,None].repeat(1,rays.shape[1],1).to(self.device)
        rays = torch.cat([rays, frameid],-1)
        rays = rays.view(-1,9)  # origin, unnormalized direction, near, far, frameid
        bs_rays = rays.shape[0]
        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays[i:i+opts.chunk],
                        N_samples = ndepth,
                        use_disp=False,
                        perturb=opts.perturb,
                        noise_std=opts.noise_std,
                        N_importance=opts.N_importance,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        white_back=False) # never turn on white_back
            for k, v in rendered_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            v = torch.cat(v, 0)
            if self.training:
                v = v.view(bs,nsample,-1)
            else:
                v = v.view(bs,img_size, img_size, -1)
            results[k] = v
        
        return results, rand_inds

    def set_input(self, batch):
        if len(batch['img'].shape)==5:
            bs,_,_,h,w = batch['img'].shape
        else:
            bs=1
            _,_,h,w = batch['img'].shape

        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()
            if not self.training:
                batch[k] = batch[k][:,:1]

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs   = input_img_tensor.cuda()
        self.imgs         = img_tensor.cuda()
        self.flow         = batch['flow']        .view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w).to(self.device)    
        self.masks        = batch['mask']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)    
        self.depth        = batch['depth']       .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)    
        self.occ          = batch['occ']         .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)    
        self.cams         = batch['cam']         .view(bs,-1,7).permute(1,0,2).reshape(-1,7)          .to(self.device)    
        self.pp           = batch['pps']         .view(bs,-1,2).permute(1,0,2).reshape(-1,2)          .to(self.device)    
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)      .to(self.device)    
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)            .to(self.device)    
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1)
        self.is_canonical = batch['is_canonical'].view(bs,-1).permute(1,0).reshape(-1)
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1)
        
        bs = 2*bs
        return bs

    def forward(self, batch):
        opts = self.opts
        bs = self.set_input(batch)
        rtk = self.rtk
        kaug= self.kaug
        frameid=self.frameid
    
        # Render
        rendered, rand_inds = self.nerf_render(rtk, kaug, frameid, opts.img_size)
        rendered_img = rendered['img_coarse']
        rendered_sil = rendered['sil_coarse']
        img_at_samp = torch.stack([self.imgs[i].view(3,-1).T[rand_inds[i]] for i in range(bs)],0) # bs,ns,3
        sil_at_samp = torch.stack([self.masks[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
           
        # loss
        img_loss = (rendered_img - img_at_samp).pow(2)
        img_loss = img_loss[sil_at_samp[...,0]>0].mean() # eval on valid pts
        sil_loss = F.mse_loss(rendered_sil, sil_at_samp)
        total_loss = sil_loss+img_loss
        
        aux_out={}
        aux_out['total_loss'] = total_loss
        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        
        return total_loss, aux_out

