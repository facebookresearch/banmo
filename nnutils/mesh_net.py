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
from ext_nnutils.rendering import render_rays
import kornia, configparser, soft_renderer as sr
from nnutils.geom_utils import K2mat, Kmatinv, K2inv, raycast, sample_xy

flags.DEFINE_float('random_geo', 1, 'Random geometric augmentation')
flags.DEFINE_string('config_name', 'template', 'name of the test data config file')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_integer('img_size', 256, 'image size')
flags.DEFINE_float('padding_frac', 0.05, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('jitter_frac', 0.05, 'bbox is jittered by this fraction of max_dim')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_integer('num_kps', 15, 'The dataloader should override these.')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_boolean('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentu')
flags.DEFINE_integer('batch_size', 8, 'Size of minibatches')
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

class v2s_net(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        super(v2s_net, self).__init__()
        self.opts = opts
        self.device = torch.device("cuda:%d"%opts.local_rank)

        # set nerf model
        self.nerf_coarse = NeRF()
        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.nerf_models = [self.nerf_coarse]
        
        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        
    def nerf_render(self, nsample=256, ndepth=64, is_eval=False):
        opts=self.opts
        Rmat = self.rtk[:,:3,:3]
        Tmat = self.rtk[:,:3,3]
        Kmat = K2mat(self.rtk[:,3,:])
        Kaug = K2inv(self.kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))
        bs = Kinv.shape[0]

        rand_inds, xys = sample_xy(opts.img_size, bs, nsample, self.device, return_all=is_eval)

        rays = raycast(xys, Rmat, Tmat, Kinv)

        # render rays
        rays = rays.view(-1,8)
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
                        white_back=False)
            for k, v in rendered_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            v = torch.cat(v, 0)
            if is_eval:
                v = v.view(bs,opts.img_size, opts.img_size, -1)
            else:
                v = v.view(bs,nsample,-1)
            results[k] = v
        
        return results, rand_inds

    def set_input(self, batch):
        opts = self.opts
        batch_size,_,_,h,w = batch['img'].shape
        bs = 2*batch_size

        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()

        img_tensor = batch['img'].view(batch_size,2, 3, h,w).permute(1,0,2,3,4).reshape(batch_size*2,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        self.input_imgs   = input_img_tensor.cuda()
        self.imgs         = img_tensor.cuda()
        self.flow         = batch['flow']        .view(batch_size,2,3,h,w).permute(1,0,2,3,4).reshape(batch_size*2,3,h,w).cuda()
        self.masks        = batch['mask']        .view(batch_size,2,h,w).permute(1,0,2,3).reshape(batch_size*2,h,w).cuda()
        self.depth        = batch['depth']       .view(batch_size,2,h,w).permute(1,0,2,3).reshape(batch_size*2,h,w).cuda()
        self.occ          = batch['occ']         .view(batch_size,2,h,w).permute(1,0,2,3).reshape(batch_size*2,h,w).cuda()
        self.cams         = batch['cam']         .view(batch_size,2,-1).permute(1,0,2).reshape(batch_size*2,-1).cuda()
        self.pp           = batch['pps']         .view(batch_size,2,-1).permute(1,0,2).reshape(batch_size*2,-1).cuda()
        self.rtk          = batch['rtk']         .view(batch_size,2,4,4).permute(1,0,2,3).reshape(batch_size*2,4,4).cuda()
        self.kaug         = batch['kaug']        .view(batch_size,2,4).permute(1,0,2).reshape(batch_size*2,4).cuda()
        self.frameid      = batch['frameid']     .view(batch_size,2).permute(1,0).reshape(-1)
        self.is_canonical = batch['is_canonical'].view(batch_size,2).permute(1,0).reshape(-1)
        self.dataid       = torch.cat([    batch['dataid'][:batch_size], batch['dataid'][:batch_size]],0)
        return bs

    def forward(self, batch_input):
        opts = self.opts
        if self.training:
            bs = self.set_input(batch_input)
        else:
            bs = len(batch_input)
            self.input_imgs = batch_input
    
        # Render
        rendered, rand_inds = self.nerf_render()
        rendered_img = rendered['rgb_coarse']
        rendered_sil = rendered['opacity_coarse']
        img_at_samp = torch.stack([self.imgs[i].view(3,-1).T[rand_inds[i]] for i in range(bs)],0) # bs,ns,3
        sil_at_samp = torch.stack([self.masks[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
           
        # loss
        img_loss = F.mse_loss(rendered_img, img_at_samp)
        sil_loss = F.mse_loss(rendered_sil, sil_at_samp)
        #total_loss = img_loss + sil_loss
        total_loss = sil_loss
        
        aux_out = {}
        aux_out['total_loss'] = total_loss
        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        
        # evaluation
        if self.iters==0:
            with torch.no_grad():
                rendered, rand_inds = self.nerf_render(is_eval=True)
                aux_out['rendered_img'] = rendered['rgb_coarse']
                aux_out['rendered_sil'] = rendered['opacity_coarse']
                
                # run marching cubes
                import mcubes
                grid_size = 100
                threshold = 0.2
                pts = np.linspace(-1.2, 1.2, grid_size).astype(np.float32)
                query_xyz = np.stack(np.meshgrid(pts, pts, pts), -1)
                query_xyz = torch.Tensor(query_xyz).to(self.device).view(-1, 3)
                query_dir = torch.zeros_like(query_xyz)

                bs_pts = query_xyz.shape[0]
                out_chunks = []
                for i in range(0, bs_pts, opts.chunk):
                    xyz_embedded = self.embedding_xyz(query_xyz[i:i+opts.chunk]) # (N, embed_xyz_channels)
                    dir_embedded = self.embedding_dir(query_dir[i:i+opts.chunk]) # (N, embed_dir_channels)
                    xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
                    out_chunks += [self.nerf_coarse(xyzdir_embedded)]
                vol_rgbo = torch.cat(out_chunks, 0)

                vol_o = vol_rgbo[...,-1].view(grid_size, grid_size, grid_size)
                print('fraction occupied:', (vol_o > threshold).float().mean())
                vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
                vertices = (vertices - grid_size/2)/grid_size*2
                mesh = trimesh.Trimesh(vertices, triangles)
                mesh.export('/private/home/gengshany/dropbox/output/0.obj')
        return total_loss, aux_out
