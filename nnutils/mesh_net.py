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
from pytorch3d import transforms

from ext_utils import mesh
from ext_utils import geometry as geom_utils
from ext_nnutils.net_blocks import Encoder, CodePredictor
from nnutils.nerf import Embedding, NeRF, RTHead
import kornia, configparser, soft_renderer as sr
from nnutils.geom_utils import K2mat, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones
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
flags.DEFINE_bool('use_cam', True, 'whether to use camera pose')
flags.DEFINE_bool('root_opt', False, 'whether to optimize root body poses')
flags.DEFINE_bool('use_corresp', False, 'whether to render and compare correspondence')
flags.DEFINE_bool('cnn_root', False, 'whether to use cnn encoder for root pose')
flags.DEFINE_integer('sample_grid3d', 128, 'resolution for mesh extraction from nerf')

#viser
flags.DEFINE_bool('use_viser', False, 'whether to use viser')
flags.DEFINE_integer('cnn_shape', 256, 'image size as input to cnn')

class v2s_net(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        super(v2s_net, self).__init__()
        self.opts = opts
        self.cnn_shape = (opts.cnn_shape,opts.cnn_shape)
        self.device = torch.device("cuda:%d"%opts.local_rank)

        # set nerf model
        self.nerf_coarse = NeRF()
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}
        self.nerf_models= {'coarse':self.nerf_coarse}

        # set dnerf model
        max_t=100  ##TODO change 15
        self.embedding_time = nn.Embedding(max_t, max_t)
        if opts.flowbw:
            self.nerf_flowbw = NeRF(in_channels_xyz=63+max_t,
                                in_channels_dir=0, raw_feat=True)
            self.nerf_flowfw = NeRF(in_channels_xyz=63+max_t,
                                in_channels_dir=0, raw_feat=True)
            self.nerf_models['flowbw'] = self.nerf_flowbw
            self.nerf_models['flowfw'] = self.nerf_flowfw
                
        elif opts.lbs:
            num_bones_x = 4
            bound = 0.5
            bones, num_bones = generate_bones(num_bones_x, bound, self.device)
            self.bones = nn.Parameter(bones)
            self.nerf_models['bones'] = self.bones

            self.nerf_bone_rts = nn.Sequential(self.embedding_time,
                                RTHead(is_bone=True, in_channels_xyz=max_t, D=4,
                                in_channels_dir=0,
                                out_channels=7*num_bones, raw_feat=True))

        # optimize camera
        if opts.root_opt:
            if opts.cnn_root:
                self.nerf_root_rts = nn.Sequential(Encoder(self.cnn_shape, n_blocks=4),
                                                   CodePredictor(n_bones=1, n_hypo=1))
            else:
                self.nerf_root_rts = nn.Sequential(self.embedding_time,
                                RTHead(is_bone=False, in_channels_xyz=max_t, D=4,
                                in_channels_dir=0,
                                out_channels=7, raw_feat=True))

        if opts.N_importance>0:
            self.nerf_fine = NeRF()
            self.nerf_models['fine'] = self.nerf_fine

        if opts.use_viser:
            from ext_nnutils.viser_net import MeshNet
            self.viser = MeshNet(self.cnn_shape, opts)

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
        frameid = frameid.long().to(self.device)[:,None]

        rand_inds, xys = sample_xy(img_size, bs, nsample, self.device, 
                                   return_all= not(self.training))
        rays = raycast(xys, Rmat, Tmat, Kinv, bound=1.5)

        # update rays
        if opts.use_corresp and bs>1:
            rtk_vec = rays['rtk_vec']
            rtk_vec_target = rtk_vec.view(2,-1).flip(0)
            rays['rtk_vec_target'] = rtk_vec_target.reshape(rays['rtk_vec'].shape)
            
            frameid_target = frameid.view(2,-1).flip(0).reshape(-1,1)
            if opts.flowbw:
                time_embedded_target = self.embedding_time(frameid_target)
                rays['time_embedded_target'] = time_embedded_target.repeat(1,
                                                            rays['nsample'],1)
            elif opts.lbs:
                bone_rts_target = self.nerf_bone_rts(frameid_target)
                rays['bone_rts_target'] = bone_rts_target.repeat(1,rays['nsample'],1)

        if opts.flowbw:
            time_embedded = self.embedding_time(frameid)
            rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        elif opts.lbs:
            bone_rts = self.nerf_bone_rts(frameid)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

        # render rays
        bs_rays = rays['bs']
        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
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
       
        # render flow 
        # bs, nsamp, -1, x
        weights_coarse = results['weights_coarse']
        # renormalize
        weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])
        weights_shape = weights_coarse.shape
        xyz_coarse_target = results['xyz_coarse_target']
        xyz_coarse_target = xyz_coarse_target.view(weights_shape+(3,))
        xy_coarse_target = xyz_coarse_target[...,:2]

        # candidate motion vector
        xys_unsq = xys.view(weights_shape[:-1]+(1,2))
        flo_coarse = xy_coarse_target - xys_unsq
        flo_coarse =  weights_coarse[...,None] * flo_coarse
        flo_coarse = flo_coarse.sum(-2)

        ## candidate target point
        #xys_unsq = xys.view(weights_shape[:-1]+(2,))
        #xy_coarse_target = weights_coarse[...,None] * xy_coarse_target
        #xy_coarse_target = xy_coarse_target.sum(-2)
        #flo_coarse = xy_coarse_target - xys_unsq

        results['flo_coarse'] = flo_coarse/img_size * 2
        del results['weights_coarse']
        del results['xyz_coarse_target']
        
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
            #if not self.training:
            #    batch[k] = batch[k][:,:1]

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs   = input_img_tensor.cuda()
        self.imgs         = img_tensor.cuda()
        self.flow         = batch['flow']        .view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w).to(self.device)
        self.flow = self.flow[:,:2]
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

        bs = self.imgs.shape[0]
        if not self.opts.use_cam:
            self.rtk[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(self.device)
            self.rtk[:,:2,3] = 0.
        
        if self.opts.root_opt:
            frameid = self.frameid.long().to(self.device)
            if self.opts.cnn_root:
                input_imgs = F.interpolate(self.input_imgs, self.cnn_shape, mode='bilinear')
                _, root_tmat1, root_rmat, root_tmat2, _ = \
                    self.nerf_root_rts(input_imgs)
                root_tmat = torch.cat([root_tmat1, root_tmat2],-1)
                root_tmat = root_tmat
                root_rmat = root_rmat.view(bs,3,3)
            else:
                root_rts = self.nerf_root_rts(frameid)
                root_quat = root_rts[:,:4]
                root_rmat = transforms.quaternion_to_matrix(root_quat)
                root_tmat = root_rts[:,4:7]
            
            rmat = self.rtk[:,:3,:3]
            tmat = self.rtk[:,:3,3]
            rmat = rmat.matmul(root_rmat)
            tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
            self.rtk[:,:3,:3] = rmat
            self.rtk[:,:3,3] = tmat
        return bs

    def forward(self, batch):
        opts = self.opts
        bs = self.set_input(batch)
        rtk = self.rtk
        kaug= self.kaug
        frameid=self.frameid
    
        
        # Render viser
        if opts.use_viser:
            pdb.set_trace()
            viser_loss = self.viser(self.input_imgs, 
                                    self.imgs, 
                                    self.masks, 
                                    self.cams, 
                                    self.flow, 
                                    self.pp, 
                                    self.occ, 
                                    self.frameid, 
                                    self.dataid)

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
        
        # regularization 
        cyc_loss = rendered['frame_cyc_dis'].mean()

        total_loss = sil_loss+img_loss+cyc_loss
        
        aux_out={}
        aux_out['total_loss'] = total_loss
        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        aux_out['cyc_loss'] = cyc_loss
        
        # flow loss
        if opts.use_corresp:
            rendered_flo = rendered['flo_coarse']
            flo_at_samp = torch.stack([self.flow[i].view(2,-1).T[rand_inds[i]] for i in range(bs)],0) # bs,ns,2
            flo_loss = (rendered_flo - flo_at_samp).pow(2).sum(-1)
            sil_at_samp_flo = (sil_at_samp>0)
            flo_loss = flo_loss[sil_at_samp_flo[...,0]].mean() # eval on valid pts

            #warmup_fac = min(1,max(0,(self.epoch-5)*0.1))
            #total_loss = total_loss*warmup_fac + flo_loss

            total_loss = total_loss + flo_loss

            aux_out['flo_loss'] = flo_loss
        
        return total_loss, aux_out

