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
import pickle
import sys
sys.path.insert(0, 'third_party')
import cv2, numpy as np, time, torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh, pytorch3d, pytorch3d.loss, pdb
from pytorch3d import transforms
import configparser

from ext_utils import mesh
from ext_utils import geometry as geom_utils
from ext_nnutils.net_blocks import Encoder, CodePredictor
from nnutils.nerf import Embedding, NeRF, RTHead
import kornia, configparser, soft_renderer as sr
from nnutils.geom_utils import K2mat, mat2K, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones,\
                                canonical2ndc, obj_to_cam, vec_to_sim3, \
                                near_far_to_bound
from nnutils.rendering import render_rays
from nnutils.loss_utils import eikonal_loss
from ext_utils.flowlib import cat_imgflo 
from ext_utils.flowlib import warp_flow

flags.DEFINE_string('rtk_path', '', 'path to rtk files')
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')
flags.DEFINE_float('random_geo', 1, 'Random geometric augmentation')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_integer('img_size', 512, 'image size')
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
flags.DEFINE_float('noise_std', 0., 'std dev of noise added to regularize sigma')
flags.DEFINE_bool('queryfw', False, 'use forward warping to query deformed shape')
flags.DEFINE_bool('flowbw', False, 'use backward warping 3d flow')
flags.DEFINE_bool('lbs', False, 'use lbs for backward warping 3d flow')
flags.DEFINE_bool('use_cam', True, 'whether to use camera pose')
flags.DEFINE_bool('root_opt', False, 'whether to optimize root body poses')
flags.DEFINE_bool('ks_opt', False,   'whether to optimize camera intrinsics')
flags.DEFINE_bool('bg', False, 'whether to optimize background')
flags.DEFINE_bool('use_corresp', False, 'whether to render and compare correspondence')
flags.DEFINE_bool('cnn_root', False, 'whether to use cnn encoder for root pose')
flags.DEFINE_integer('sample_grid3d', 64, 'resolution for mesh extraction from nerf')
flags.DEFINE_integer('num_test_views', 0, 'number of test views, 0: use all viewsf')
flags.DEFINE_bool('use_dp', False, 'whether to use densepose')
flags.DEFINE_bool('flow_dp', False, 'replace flow with densepose flow')
flags.DEFINE_bool('anneal_freq', True, 'whether to use frequency annealing')
flags.DEFINE_integer('alpha', 10, 'maximum frequency for fourier features')
flags.DEFINE_bool('eikonal_loss', False, 'whether to use eikonal loss')
flags.DEFINE_float('rot_angle', 0.0, 'angle of initial rotation * pi')

#viser
flags.DEFINE_bool('use_viser', False, 'whether to use viser')
flags.DEFINE_integer('cnn_shape', 256, 'image size as input to cnn')

class v2s_net(nn.Module):
    def __init__(self, input_shape, opts, data_info, half_bones):
        super(v2s_net, self).__init__()
        self.opts = opts
        self.cnn_shape = (opts.cnn_shape,opts.cnn_shape)
        self.device = torch.device("cuda:%d"%opts.local_rank)
        self.config = configparser.RawConfigParser()
        self.config.read('configs/%s.config'%opts.seqname)
        self.alpha=torch.Tensor([opts.alpha])
        self.alpha=nn.Parameter(self.alpha)

        # multi-video mode
        self.num_vid =  len(self.config.sections())-1
        self.data_offset = data_info['offset']
        self.impath      = data_info['impath']
        self.latest_vars = {}
        self.latest_vars['rtk'] = np.zeros((self.data_offset[-1], 4,4))
        self.latest_vars['j2c'] = np.zeros((self.data_offset[-1], 10))
        self.latest_vars['idk'] = np.zeros((self.data_offset[-1],))
        self.latest_vars['vis'] = np.zeros((self.data_offset[-1],
                                 opts.img_size,opts.img_size)).astype(bool)

        # get near-far plane
        try:
            self.near_far = np.zeros((self.data_offset[-1],2)).astype(np.float32)
            for nvid in range(self.num_vid):
                self.near_far[self.data_offset[nvid]:self.data_offset[nvid+1]]=\
        [float(i) for i in self.config.get('data_%d'%nvid, 'near_far').split(',')]
            self.near_far = torch.Tensor(self.near_far).to(self.device)
            self.near_far = nn.Parameter(self.near_far)
        except: self.near_far = None
    
        # object bound
        self.latest_vars['obj_bound'] = near_far_to_bound(self.near_far)

        self.vis_min=np.asarray([[0,0,0]])
        self.vis_max=np.asarray([[1,1,1]])
        
        # video specific sim3: from video to joint canonical space
        self.sim3_j2c= generate_bones(self.num_vid, self.num_vid, 0, self.device)
        if self.num_vid>1:
            angle=opts.rot_angle*np.pi
            init_rot = transforms.axis_angle_to_quaternion(torch.Tensor([0,angle,0]))
            self.sim3_j2c.data[1,3:7] = init_rot.to(self.device) #TODO
        self.sim3_j2c = nn.Parameter(self.sim3_j2c)

        # set nerf model
        self.num_freqs = 10
        in_channels_xyz=3+3*self.num_freqs*2
        self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz)
        self.embedding_xyz = Embedding(3,self.num_freqs,alpha=self.alpha.data[0])
        self.embedding_dir = Embedding(3,4,             alpha=self.alpha.data[0])
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}
        self.nerf_models= {'coarse':self.nerf_coarse}

        # set dnerf model
        max_t=self.data_offset[-1]  
        t_embed_dim = 128
        self.embedding_time = nn.Embedding(max_t, t_embed_dim)
        if opts.flowbw:
            self.nerf_flowbw = NeRF(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                in_channels_dir=0, raw_feat=True)
            self.nerf_flowfw = NeRF(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                in_channels_dir=0, raw_feat=True)
            self.nerf_models['flowbw'] = self.nerf_flowbw
            self.nerf_models['flowfw'] = self.nerf_flowfw
                
        elif opts.lbs:
            num_bones_x = 3 # TODO change to # of cat bones
            if opts.bg: num_bones_x=5
            num_bones = num_bones_x**3
            if half_bones: num_bones = num_bones // 2
            self.num_bones = num_bones
            bones= generate_bones(num_bones_x, num_bones, 0, self.device)
            self.bones = nn.Parameter(bones)
            self.nerf_models['bones'] = self.bones

            self.nerf_bone_rts = nn.Sequential(self.embedding_time,
                                RTHead(is_bone=True, use_cam=opts.use_cam, 
                                in_channels_xyz=t_embed_dim, D=4, in_channels_dir=0,
                                out_channels=7*self.num_bones, raw_feat=True))

        # optimize camera
        if opts.root_opt:
            if opts.cnn_root:
                self.nerf_root_rts = nn.Sequential(Encoder(self.cnn_shape, n_blocks=4),
                                                   CodePredictor(n_bones=1, n_hypo=1))
            else:
                self.nerf_root_rts = nn.Sequential(self.embedding_time,
                                RTHead(is_bone=False, use_cam=opts.use_cam, 
                                in_channels_xyz=t_embed_dim, D=4, in_channels_dir=0,
                                out_channels=7, raw_feat=True))

        if opts.ks_opt:
            # TODO: change according to multiple video
            fx,fy,px,py=[int(float(i)) for i in \
                        self.config.get('data_0', 'ks').split(',')]
            self.ks = torch.Tensor([fx,fy,px,py]).to(self.device)
            self.ks_param = nn.Parameter(self.ks)
            

        # densepose
        if opts.flow_dp:
            with open('mesh_material/geodists_sheep_5004.pkl', 'rb') as f: 
                geodists=pickle.load(f)
                geodists = torch.Tensor(geodists).cuda(self.device)
                geodists[0,:] = np.inf
                geodists[:,0] = np.inf
                self.geodists = geodists
            self.dp_thrd = 0.1 # threshold to fb error of dp

        with open('mesh_material/sheep_5004.pkl', 'rb') as f:
            dp = pickle.load(f)
            self.dp_verts = dp['vertices']
            self.dp_faces = dp['faces']
            self.dp_verts = torch.Tensor(self.dp_verts).cuda(self.device)
            
            self.dp_verts -= self.dp_verts.mean(0)[None]
            self.dp_verts /= self.dp_verts.abs().max()
            self.dp_verts *= (self.near_far[:,1] - self.near_far[:,0]).mean()/2
            
            # visualize
            self.dp_vis = self.dp_verts
            self.dp_vmin = self.dp_vis.min(0)[0][None]
            self.dp_vis = self.dp_vis - self.dp_vmin
            self.dp_vmax = self.dp_vis.max(0)[0][None]
            self.dp_vis = self.dp_vis / self.dp_vmax

        if opts.use_dp:
            self.dp_verts = nn.Parameter(self.dp_verts)

            # deformation from joint canonincal space to dp space
            self.nerf_dp = NeRF(in_channels_xyz=in_channels_xyz,
                                in_channels_dir=0, raw_feat=True)
            self.nerf_models['nerf_dp'] = self.nerf_dp

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
        # don't update the canonical frame sim3
        sim3_j2c = torch.cat([self.sim3_j2c[:1].detach(),  
                              self.sim3_j2c[1:]],0)

        rand_inds, xys = sample_xy(img_size, bs, nsample, self.device, 
                                   return_all= not(self.training))
        near_far = self.near_far[self.frameid.long()]
        rays = raycast(xys, Rmat, Tmat, Kinv, near_far)

        # update rays
        if bs>1:
            rtk_vec = rays['rtk_vec'] # bs, N, 21
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

            if opts.flow_dp:
                # randomly choose 1 target image
                rays['rtk_vec_dentrg'] = rtk_vec[self.rand_dentrg] # bs,N,21
                frameid_dentrg = frameid.view(-1,1)[self.rand_dentrg]
                if opts.flowbw:
                    print('Error: not implemented')
                    exit()
                elif opts.lbs:
                    bone_rts_dentrg = self.nerf_bone_rts(frameid_dentrg) #bsxbs,x 
                    rays['bone_rts_dentrg'] = bone_rts_dentrg.repeat(1,rays['nsample'],1)

                dataid_dentrg = self.dataid[self.rand_dentrg]
                rays['sim3_j2c_dentrg'] = sim3_j2c[dataid_dentrg.long()]
                rays['sim3_j2c_dentrg'] = rays['sim3_j2c_dentrg'][:,None].repeat(1,rays['nsample'],1)
                 

        if opts.flowbw:
            time_embedded = self.embedding_time(frameid)
            rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        elif opts.lbs:
            bone_rts = self.nerf_bone_rts(frameid)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

        # pass the canonical to joint space transforms
        rays['sim3_j2c'] = sim3_j2c[self.dataid.long()]
        rays['sim3_j2c'] = rays['sim3_j2c'][:,None].repeat(1,rays['nsample'],1)

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
        weights_coarse = results['weights_coarse'].clone()
        weights_shape = weights_coarse.shape
        xyz_coarse_target = results['xyz_coarse_target']
        xyz_coarse_target = xyz_coarse_target.view(weights_shape+(3,))
        xy_coarse_target = xyz_coarse_target[...,:2]

        # deal with negative z
        invalid_ind = torch.logical_or(xyz_coarse_target[...,-1]<1e-5,
                               xy_coarse_target.norm(2,-1).abs()>2*img_size)
        weights_coarse[invalid_ind] = 0.
        xy_coarse_target[invalid_ind] = 0.

        # renormalize
        weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])

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
        del results['xyz_coarse_target']

        if opts.flow_dp:
            weights_coarse = results['weights_coarse'].clone()
            xyz_coarse_dentrg = results['xyz_coarse_dentrg']
            xyz_coarse_dentrg = xyz_coarse_dentrg.view(weights_shape+(3,))
            xy_coarse_dentrg = xyz_coarse_dentrg[...,:2]

            # deal with negative z
            invalid_ind = torch.logical_or(xyz_coarse_dentrg[...,-1]<1e-5,
                                   xy_coarse_dentrg.norm(2,-1).abs()>2*img_size)
            weights_coarse[invalid_ind] = 0.
            xy_coarse_dentrg[invalid_ind] = 0.

            # renormalize
            weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])

            # candidate motion vector
            xys_unsq = xys.view(weights_shape[:-1]+(1,2))
            fdp = xy_coarse_dentrg - xys_unsq
            fdp =  weights_coarse[...,None] * fdp
            fdp = fdp.sum(-2)

            results['fdp_coarse'] = fdp/img_size * 2
            del results['xyz_coarse_dentrg']
        del results['weights_coarse']

        if opts.use_dp:
            # visualize cse predictions
            dp_pts = results['joint_render']
            dp_vis_gt = self.dp_vis[self.dps.long()] # bs, h, w, 3
            dp_vis_pred = (dp_pts - self.dp_vmin)/self.dp_vmax
            dp_vis_pred = dp_vis_pred.clamp(0,1)
            results['dp_vis_gt'] = dp_vis_gt
            results['dp_vis_pred'] = dp_vis_pred
    
        results['joint_render_vis'] = (results['joint_render']-\
                       torch.Tensor(self.vis_min[None,None]).to(self.device))/\
                       torch.Tensor(self.vis_max[None,None]).to(self.device)
        results['joint_render_vis'] = results['joint_render_vis'].clamp(0,1)
        #    pdb.set_trace() 
        #    trimesh.Trimesh(self.dp_verts.cpu(), self.dp_faces,
        #            vertex_colors=self.dp_vis.cpu()).export('0.obj')
        #    trimesh.Trimesh(dp_pts[0].view(-1,3).cpu(),
        # vertex_colors=dp_vis_pred[0].view(-1,3).cpu()).export('1.obj')
               
        
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

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs   = input_img_tensor.cuda()
        self.imgs         = img_tensor.cuda()
        self.flow         = batch['flow']        .view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w).to(self.device)
        self.flow = self.flow[:,:2]
        self.masks        = batch['mask']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)
        self.vis2d        = batch['vis2d']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)
        self.dps          = batch['dp']          .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)
        self.depth        = batch['depth']       .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)
        self.occ          = batch['occ']         .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(self.device)
        self.cams         = batch['cam']         .view(bs,-1,7).permute(1,0,2).reshape(-1,7)          .to(self.device)  
        self.pp           = batch['pps']         .view(bs,-1,2).permute(1,0,2).reshape(-1,2)          .to(self.device)  
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)      .to(self.device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)            .to(self.device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1)
        self.is_canonical = batch['is_canonical'].view(bs,-1).permute(1,0).reshape(-1)
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1)

        if self.opts.flow_dp:
            # densepose to correspondence
            # randomly choose 1 target image
            if self.training:
                order1 = np.asarray(range(2*bs))
                is_degenerate_pair = len(set((self.frameid.cpu().numpy())))==2
                while True:
                    rand_dentrg = np.random.randint(0,2*bs,2*bs)
                    if is_degenerate_pair or \
            ((self.frameid[rand_dentrg]-self.frameid[order1])==0).sum()==0:
                        break
            else: rand_dentrg = np.asarray([1,0])
            self.rand_dentrg = rand_dentrg
            # downsample
            h_rszd,w_rszd=h//4,w//4
            self.dp_flow = torch.zeros(2*bs,2,h_rszd,w_rszd).to(self.device)
            self.dp_conf = torch.zeros(2*bs,h_rszd,w_rszd).to(self.device)
            hw_rszd = h_rszd*w_rszd
            dps = F.interpolate(self.dps[:,None], (h_rszd,w_rszd), mode='nearest')[:,0]
            for idx in range(2*bs):
                jdx = self.rand_dentrg[idx]
                def compute_flow_geodist(dps,idx,jdx):
                    h_rszd,w_rszd = dps.shape[1:]
                    hw_rszd = h_rszd*w_rszd
                    device = dps.device
                    dps = dps.view(2*bs,-1).long()
                    dp_refr = dps[idx].view(-1,1).repeat(1,hw_rszd).view(-1,1)
                    dp_targ = dps[jdx].view(1,-1).repeat(hw_rszd,1).view(-1,1)
                    match = self.geodists[dp_refr, dp_targ]
                    dis_geo,match = match.view(hw_rszd, hw_rszd).min(1)
                    #match[dis_geo>0.1] = 0

                    # cx,cy
                    tar_coord = torch.cat([match[:,None]%w_rszd, match[:,None]//w_rszd],-1)
                    tar_coord = tar_coord.view(h_rszd, w_rszd, 2).float()
                    ref_coord = torch.Tensor(np.meshgrid(range(w_rszd), range(h_rszd)))
                    ref_coord = ref_coord.to(device).permute(1,2,0).view(-1,2)
                    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
                    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
                    match = match.view(h_rszd, w_rszd)
                    flo_dp[match==0] = 0
                    flo_dp = flo_dp.permute(2,0,1)
                    return flo_dp
                flo_refr = compute_flow_geodist(dps,idx,jdx)
                flo_targ = compute_flow_geodist(dps,jdx,idx)
                self.dp_flow[idx] = flo_refr

                # clean up flow
                flo_refr = flo_refr.permute(1,2,0).cpu().numpy()
                flo_targ = flo_targ.permute(1,2,0).cpu().numpy()
                flo_refr_mask = np.linalg.norm(flo_refr,2,-1)>0
                flo_targ_mask = np.linalg.norm(flo_targ,2,-1)>0
                flo_refr_px = flo_refr * w_rszd / 2
                flo_targ_px = flo_targ * w_rszd / 2

                #fb check
                x0,y0  =np.meshgrid(range(w_rszd),range(h_rszd))
                hp0 = np.stack([x0,y0],-1) # screen coord

                flo_fb = warp_flow(hp0 + flo_targ_px, flo_refr_px) - hp0
                flo_fb = 2*flo_fb/w_rszd
                fberr_fw = np.linalg.norm(flo_fb, 2,-1)
                fberr_fw[~flo_refr_mask] = 0
                self.dp_conf[idx] = torch.Tensor(fberr_fw)

                flo_bf = warp_flow(hp0 + flo_refr_px, flo_targ_px) - hp0
                flo_bf = 2*flo_bf/w_rszd
                fberr_bw = np.linalg.norm(flo_bf, 2,-1)
                fberr_bw[~flo_targ_mask] = 0

                ## vis
                #thrd_vis = 0.01
                #img_refr = F.interpolate(self.imgs[idx:idx+1], (h_rszd, w_rszd), mode='bilinear')[0]
                #img_refr = img_refr.permute(1,2,0).cpu().numpy()[:,:,::-1]
                #img_targ = F.interpolate(self.imgs[jdx:jdx+1], (h_rszd, w_rszd), mode='bilinear')[0]
                #img_targ = img_targ.permute(1,2,0).cpu().numpy()[:,:,::-1]
                #flo_refr[:,:,0] = (flo_refr[:,:,0] + 2)/2
                #flo_targ[:,:,0] = (flo_targ[:,:,0] - 2)/2
                #flo_refr[fberr_fw>thrd_vis]=0.
                #flo_targ[fberr_bw>thrd_vis]=0.
                #flo_refr[~flo_refr_mask]=0.
                #flo_targ[~flo_targ_mask]=0.
                #img = np.concatenate([img_refr, img_targ], 1)
                #flo = np.concatenate([flo_refr, flo_targ], 1)
                #imgflo = cat_imgflo(img, flo)
                #imgcnf = np.concatenate([fberr_fw, fberr_bw],1)
                #imgcnf = np.clip(imgcnf, 0, self.dp_thrd)*(255/self.dp_thrd)
                #imgcnf = np.repeat(imgcnf[...,None],3,-1)
                #imgcnf = cv2.resize(imgcnf, imgflo.shape[::-1][1:])
                #imgflo_cnf = np.concatenate([imgflo, imgcnf],0)
                #cv2.imwrite('tmp/img-%05d-%05d.jpg'%(idx,jdx), imgflo_cnf)
            self.dp_conf[self.dp_conf>self.dp_thrd] = self.dp_thrd
 
        ## TODO
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]
        self.sils = self.masks.clone()
        if self.opts.bg: self.masks[:] = 1
        self.masks = (self.masks*self.vis2d)>0
        self.masks = self.masks.float()
        self.sils = (self.sils*self.vis2d)>0
        self.sils =  self.sils.float()

        bs = self.imgs.shape[0]
        if not self.opts.use_cam:
            self.rtk[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(self.device)
            self.rtk[:,:2,3] = 0.
            if self.near_far is None:
                self.rtk[:,2,3] = 3. # TODO heuristics of depth=3xobject size
            else:
                self.rtk[:,2,3] = self.near_far.mean() # TODO need to deal with multi-videl
       
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
            tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
            rmat = rmat.matmul(root_rmat)
            self.rtk[:,:3,:3] = rmat
            self.rtk[:,:3,3] = tmat

        if self.opts.ks_opt:
            self.rtk[:,3,:] = self.ks_param #TODO kmat

        # save latest variables
        rtk = self.rtk.clone().detach()
        Kmat = K2mat(rtk[:,3])
        Kaug = K2inv(self.kaug) # p = Kaug Kmat P
        rtk[:,3] = mat2K(Kaug.matmul(Kmat))
        self.latest_vars['rtk'][self.frameid.long()] = rtk.cpu().numpy()
        self.latest_vars['j2c'][self.frameid.long()] = self.sim3_j2c.detach().cpu().numpy()\
                                                        [self.dataid.long()]
        self.latest_vars['idk'][self.frameid.long()] = 1
        self.latest_vars['vis'][self.frameid.long()] = self.vis2d.cpu().numpy()
        
        if self.training and self.opts.anneal_freq:
            alpha = self.num_freqs * self.total_steps / (self.final_steps/2)
            if alpha>self.alpha.data[0]:
                self.alpha.data[0] = min(max(3, alpha),self.num_freqs) # alpha from 3 to 10
            self.embedding_xyz.alpha = self.alpha.data[0]
            self.embedding_dir.alpha = self.alpha.data[0]

        return bs

    def forward(self, batch):
        opts = self.opts
        bs = self.set_input(batch)
        rtk = self.rtk
        kaug= self.kaug
        frameid=self.frameid
        aux_out={}
        
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
        total_loss = img_loss
        if not opts.bg: total_loss = total_loss + sil_loss 

        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        
        if opts.use_dp:
            rendered_dp = rendered['joint_render']
            dp_at_samp = torch.stack([self.dps[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
            dp_at_samp = self.dp_verts[dp_at_samp[...,0].long()]
            dp_loss = (rendered_dp - dp_at_samp).pow(2)
            dp_loss = dp_loss[sil_at_samp[...,0]>0].mean() # eval on valid pts
            total_loss = total_loss + dp_loss
            aux_out['dp_loss'] = dp_loss
            
        
        # flow loss
        if opts.use_corresp:
            rendered_flo = rendered['flo_coarse']

            flo_at_samp = torch.stack([self.flow[i].view(2,-1).T[rand_inds[i]] for i in range(bs)],0) # bs,ns,2
            flo_loss = (rendered_flo - flo_at_samp).pow(2).sum(-1)
            sil_at_samp_flo = (sil_at_samp>0)

            # confidence weighting: 30x normalized distance
            cfd_at_samp = torch.stack([self.occ[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
            cfd_at_samp = (-cfd_at_samp).sigmoid()
            cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss = flo_loss * cfd_at_samp[...,0]
            
            flo_loss = flo_loss[sil_at_samp_flo[...,0]].mean() # eval on valid pts

            if opts.root_opt and (not opts.use_cam):
                warmup_fac = min(1,max(0,(self.epoch-5)*0.1))
                total_loss = total_loss*warmup_fac + flo_loss
            else:
                total_loss = total_loss + flo_loss

            aux_out['flo_loss'] = flo_loss
        
        # flow densepose loss
        if opts.flow_dp:
            rendered_fdp = rendered['fdp_coarse'] # bs,N,2
            fdp_at_samp = []
            dcf_at_samp = []
            for i in range(bs):
                # 1/4 resolution
                dp_flow = F.interpolate(self.dp_flow[i][None], 
                             (opts.img_size,opts.img_size), mode='bilinear')[0]
                dp_conf = F.interpolate(self.dp_conf[i][None,None], 
                             (opts.img_size,opts.img_size), mode='bilinear')[0]
                fdp_at_samp.append(dp_flow.view(2,-1).T[rand_inds[i]])
                dcf_at_samp.append(dp_conf.view(-1,1)[rand_inds[i]])
            fdp_at_samp = torch.stack(fdp_at_samp,0)
            dcf_at_samp = torch.stack(dcf_at_samp,0)
            fdp_loss = (rendered_fdp - fdp_at_samp).pow(2).sum(-1)
            fdp_loss = (fdp_loss-0.02).relu() # ignore error < 1/20 unit

            # TODO confidence weighting
            sil_at_samp_fdp = (sil_at_samp>0) & (dcf_at_samp<self.dp_thrd-1e-3)
            dcf_at_samp = (-30*dcf_at_samp).sigmoid()
            dcf_at_samp = dcf_at_samp / dcf_at_samp[sil_at_samp_fdp].mean()
            fdp_loss = fdp_loss * dcf_at_samp[...,0]
            
            fdp_loss = 0.1*fdp_loss[sil_at_samp_fdp[...,0]].mean() # eval on valid pts
            total_loss = total_loss + fdp_loss
            aux_out['fdp_loss'] = fdp_loss
        
        # regularization 
        if opts.lbs or opts.flowbw:
            # cycle loss
            cyc_loss = rendered['frame_cyc_dis'].mean()
            total_loss = total_loss + cyc_loss
            aux_out['cyc_loss'] = cyc_loss

            # globally rigid prior
            rig_loss = 0.0001*rendered['frame_rigloss'].mean()
            total_loss = total_loss + rig_loss
            aux_out['rig_loss'] = rig_loss

        if opts.eikonal_loss:
            ekl_loss = 0.01*eikonal_loss(self.nerf_coarse, self.embedding_xyz, 
                                         self.latest_vars['obj_bound'])
            total_loss = total_loss + ekl_loss
            aux_out['ekl_loss'] = ekl_loss

        aux_out['total_loss'] = total_loss
        return total_loss, aux_out

