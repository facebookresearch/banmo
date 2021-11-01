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
#from pytorch3d import structures
import configparser
#from geomloss import SamplesLoss

from ext_utils import mesh
from ext_utils import geometry as geom_utils
from nnutils.nerf import Embedding, NeRF, RTHead, SE3head, RTExplicit, Encoder,\
                    ScoreHead, evaluate_mlp, Transhead, NeRFUnc
import kornia, configparser, soft_renderer as sr
from nnutils.geom_utils import K2mat, mat2K, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones,\
                                canonical2ndc, obj_to_cam, vec_to_sim3, \
                                near_far_to_bound, compute_flow_geodist, \
                                compute_flow_cse, fb_flow_check, pinhole_cam, \
                                render_color, mask_aug, bbox_dp2rnd, resample_dp, \
                                vrender_flo
from nnutils.rendering import render_rays
from nnutils.loss_utils import eikonal_loss, nerf_gradient, rtk_loss, rtk_cls_loss,\
                            feat_match_loss, kp_reproj_loss, grad_update_bone
from utils.io import vis_viser, draw_pts

flags.DEFINE_string('rtk_path', '', 'path to rtk files')
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')
flags.DEFINE_float('random_geo', 1, 'Random geometric augmentation')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_integer('img_size', 512, 'image size')
flags.DEFINE_integer('render_size', 64, 'size used for eval visualizations')
flags.DEFINE_enum('split', 'train', ['train', 'val', 'all', 'test'], 'eval split')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_string('logname', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_boolean('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentu')
flags.DEFINE_integer('batch_size', 1, 'size of minibatches')
flags.DEFINE_string('checkpoint_dir', 'logdir/', 'Root directory for output files')
flags.DEFINE_string('model_path', '', 'load model path')
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
flags.DEFINE_integer('rnd_frame_chunk', 3, 'chunk size to render eval images')
flags.DEFINE_integer('frame_chunk', 20, 'chunk size to split the input frames')
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
flags.DEFINE_string('root_basis', 'mlp', 'which root pose basis to use {mlp, cnn, exp}')
flags.DEFINE_integer('sample_grid3d', 64, 'resolution for mesh extraction from nerf')
flags.DEFINE_string('test_frames', '9', 'a list of video index or num of frames, {0,1,2}, 30')
flags.DEFINE_bool('flow_dp', False, 'replace flow with densepose flow')
flags.DEFINE_bool('anneal_freq', True, 'whether to use frequency annealing')
flags.DEFINE_integer('alpha', 10, 'maximum frequency for fourier features')
flags.DEFINE_bool('eikonal_loss', False, 'whether to use eikonal loss')
flags.DEFINE_bool('use_sim3', False, 'whether to use sim3 transformation')
flags.DEFINE_float('rot_angle', 0.0, 'angle of initial rotation * pi')
flags.DEFINE_integer('num_bones', 25, 'maximum number of bones')
flags.DEFINE_float('warmup_init_steps', 0.2, 'steps before using sil loss')
flags.DEFINE_float('warmup_steps', 0.2, 'steps used to increase sil loss')
flags.DEFINE_integer('lbs_reinit_epochs', -1, 'epochs to initialize bones')
#flags.DEFINE_float('reinit_bone_steps', 0, 'steps to initialize bones')
flags.DEFINE_float('reinit_bone_steps', 0.667, 'steps to initialize bones')
flags.DEFINE_float('proj_start', 0.0, 'steps to strat projection opt')
flags.DEFINE_float('proj_end', 0.5,  'steps to end projection opt')
flags.DEFINE_float('dskin_steps', 0.8, 'steps to add delta skinning weights')
flags.DEFINE_integer('lbs_all_epochs', 10, 'epochs used to add all bones')
flags.DEFINE_bool('se3_flow', False, 'whether to use se3 field for 3d flow')
flags.DEFINE_bool('nerf_vis', True, 'use visibility volume')
flags.DEFINE_bool('nerf_skin', True, 'use mlp skinning function')
flags.DEFINE_float('init_beta', 1., 'initial value for transparency beta')
flags.DEFINE_float('sil_wt', 0.1, 'weight for silhouette loss')
flags.DEFINE_bool('bone_loc_reg', True, 'use bone location regularization')
flags.DEFINE_integer('nsample', 256, 'num of samples per image at optimization time')
flags.DEFINE_integer('ndepth', 128, 'num of depth samples per px at optimization time')
flags.DEFINE_bool('vis_dpflow', False, 'whether to visualize densepose flow')
flags.DEFINE_bool('env_code', True, 'whether to use environment code for each video')
flags.DEFINE_integer('warmup_pose_ep', 0, 'epochs to pre-train cnn pose predictor')
flags.DEFINE_string('nf_path', '', 'a array of near far planes, Nx2')
flags.DEFINE_string('pose_cnn_path', '', 'path to pre-trained pose cnn')
flags.DEFINE_string('cnn_feature', 'embed', 'input to pose cnn')
flags.DEFINE_string('cnn_type', 'reg', 'output of pose cnn')
flags.DEFINE_bool('sfm_init', True, 'whether to maintain sfm relative trajectory')
flags.DEFINE_bool('unit_nf', True, 'whether to set near-far plane as unit value (0-6)')

#viser
flags.DEFINE_bool('use_viser', False, 'whether to use viser')
flags.DEFINE_bool('use_proj', False, 'whether to use reprojection loss')
flags.DEFINE_bool('freeze_proj', False, 'whether to freeze some params w/ proj loss')
flags.DEFINE_bool('freeze_cvf',  False, 'whether to freeze canonical features')
flags.DEFINE_bool('freeze_shape',False, 'whether to freeze canonical shape')
flags.DEFINE_bool('freeze_root',False, 'whether to freeze root pose')
flags.DEFINE_integer('cnn_shape', 256, 'image size as input to cnn')
flags.DEFINE_float('fine_steps', 0.8, 'by default, not using fine samples')
flags.DEFINE_float('nf_reset', 0.5, 'by default, start reseting near-far plane at 50%')
flags.DEFINE_bool('use_resize',True, 'whether to use cycle resize')
flags.DEFINE_bool('use_unc',True, 'whether to use uncertainty sampling')

# for match
flags.DEFINE_string('match_frames', '0 1', 'a list of frame index')

class v2s_net(nn.Module):
    def __init__(self, opts, data_info):
        super(v2s_net, self).__init__()
        self.opts = opts
        self.cnn_shape = (opts.cnn_shape,opts.cnn_shape)
        self.device = torch.device("cuda:%d"%opts.local_rank)
        self.config = configparser.RawConfigParser()
        self.config.read('configs/%s.config'%opts.seqname)
        self.alpha=torch.Tensor([opts.alpha])
        self.alpha=nn.Parameter(self.alpha)
        self.pose_update = 2 # by default, update all, use all losses
        self.shape_update = 0 # by default, update all
        self.cvf_update = 0 # by default, update all
        self.progress = 0. # also reseted in optimizer
        self.use_fine = False # by default not using fine samples
        self.root_basis = opts.root_basis
        self.use_cam = opts.use_cam
        self.img_size = opts.img_size # current rendering size, 
                                      # have to be consistent with dataloader, 
                                      # eval/train has different size
        
        # multi-video mode
        self.num_vid =  len(self.config.sections())-1
        self.data_offset = data_info['offset']
        self.max_ts = (self.data_offset[1:] - self.data_offset[:-1]).max()
        self.impath      = data_info['impath']
        self.latest_vars = {}
        self.latest_vars['rtk'] = np.zeros((self.data_offset[-1], 4,4))
        if opts.use_sim3:
            self.latest_vars['j2c'] = np.zeros((self.data_offset[-1], 10))
        self.latest_vars['idk'] = np.zeros((self.data_offset[-1],))
        self.latest_vars['vis'] = np.zeros((self.data_offset[-1],
                                 self.img_size,self.img_size)).astype(bool)
        self.latest_vars['mesh_rest'] = trimesh.Trimesh()

        # get near-far plane
        if opts.unit_nf:
            self.near_far = np.zeros((self.data_offset[-1],2))
            self.near_far[...,1] = 6.
            self.near_far = self.near_far.astype(np.float32)
        elif opts.nf_path=='':
            try:
                self.near_far = self.near_far_from_config(self.config, 
                                         self.data_offset, self.num_vid)
            except:
                print('near_far plane not defined')
                exit()
        else:
            print('near_far plane not defined')
            exit()
        self.near_far = torch.Tensor(self.near_far).to(self.device)
        self.obj_scale = float(near_far_to_bound(self.near_far)) / 0.3 # to 0.3
        self.near_far = self.near_far / self.obj_scale
        self.near_far = nn.Parameter(self.near_far)
    
        # object bound
        self.latest_vars['obj_bound'] = near_far_to_bound(self.near_far)

        self.vis_min=np.asarray([[0,0,0]])
        self.vis_len=np.asarray([[1,1,1]])*self.latest_vars['obj_bound']/2
        
        if opts.use_sim3:
            # video specific sim3: from video to joint canonical space
            self.sim3_j2c= generate_bones(self.num_vid, self.num_vid, 0, self.device)
            if self.num_vid>1:
                angle=opts.rot_angle*np.pi
                init_rot = transforms.axis_angle_to_quaternion(torch.Tensor([0,angle,0]))
                self.sim3_j2c.data[1,3:7] = init_rot.to(self.device)
            self.sim3_j2c = nn.Parameter(self.sim3_j2c)

        if opts.env_code:
            # add video-speficit environment lighting embedding
            env_code_dim = 64
            self.env_code = nn.Embedding(self.num_vid, env_code_dim)
        else:
            env_code_dim = 0

        # set nerf model
        self.num_freqs = 10
        in_channels_xyz=3+3*self.num_freqs*2
        in_channels_dir=27
        self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz, 
                                in_channels_dir=in_channels_dir+env_code_dim,
                                init_beta=opts.init_beta)
        self.embedding_xyz = Embedding(3,self.num_freqs,alpha=self.alpha.data[0])
        self.embedding_dir = Embedding(3,4,             alpha=self.alpha.data[0])
        self.embeddings = {'xyz':self.embedding_xyz, 'dir':self.embedding_dir}
        self.nerf_models= {'coarse':self.nerf_coarse}

        # set dnerf model
        max_t=self.data_offset[-1]  
        t_embed_dim = 128
        self.pose_code = nn.Embedding(max_t, t_embed_dim)
        if opts.flowbw:
            if opts.se3_flow:
                flow3d_arch = SE3head
                out_channels=9
            else:
                flow3d_arch = Transhead
                out_channels=3
            self.nerf_flowbw = flow3d_arch(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                D=5, W=128,
                    out_channels=out_channels,in_channels_dir=0, raw_feat=True)
            self.nerf_flowfw = flow3d_arch(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                D=5, W=128,
                    out_channels=out_channels,in_channels_dir=0, raw_feat=True)
            self.nerf_models['flowbw'] = self.nerf_flowbw
            self.nerf_models['flowfw'] = self.nerf_flowfw
                
        elif opts.lbs:
            self.num_bones = opts.num_bones
            bones= generate_bones(self.num_bones, self.num_bones, 0, self.device)
            self.bones = nn.Parameter(bones)
            self.nerf_models['bones'] = self.bones
            self.num_bone_used = self.num_bones # bones used in the model

            self.nerf_bone_rts = nn.Sequential(self.pose_code,
                                RTHead(use_quat=False, 
                                #D=5,W=128,
                                in_channels_xyz=t_embed_dim,in_channels_dir=0,
                                out_channels=6*self.num_bones, raw_feat=True))
            #TODO scale+constant parameters
            skin_aux = torch.Tensor([0,self.obj_scale]) 
            self.skin_aux = nn.Parameter(skin_aux)
            self.nerf_models['skin_aux'] = self.skin_aux

            if opts.nerf_skin:
                self.nerf_skin = NeRF(in_channels_xyz=in_channels_xyz+t_embed_dim,
                                    D=5,W=128,
                     in_channels_dir=0, out_channels=self.num_bones, raw_feat=True)
                self.rest_pose_code = nn.Embedding(1, t_embed_dim)
                self.nerf_models['nerf_skin'] = self.nerf_skin
                self.nerf_models['rest_pose_code'] = self.rest_pose_code

        # set visibility nerf
        if opts.nerf_vis:
            self.nerf_vis = NeRF(in_channels_xyz=in_channels_xyz, D=5, W=64, 
                                    out_channels=1, in_channels_dir=0,
                                    raw_feat=True)
            self.nerf_models['nerf_vis'] = self.nerf_vis
        
        # optimize camera
        if opts.root_opt:
            if self.use_cam: 
                use_quat=False
                out_channels=6
            else:
                use_quat=True
                out_channels=7
            # train a cnn pose predictor for warmup
            if opts.cnn_feature=='embed':
                cnn_in_channels = 16
            elif opts.cnn_feature=='pts':
                cnn_in_channels = 3

            if opts.cnn_type == 'reg':
                cnn_head = RTHead(use_quat=True, D=1,
                            in_channels_xyz=t_embed_dim,in_channels_dir=0,
                            out_channels=7, raw_feat=True)
            elif opts.cnn_type == 'cls':
                recursion_level = 1
                cnn_head = ScoreHead(recursion_level=recursion_level, D=1,
                            in_channels_xyz=t_embed_dim,in_channels_dir=0,
                        out_channels=3+72*8**recursion_level, raw_feat=True)
            self.dp_root_rts = nn.Sequential(
                            Encoder((112,112), in_channels=cnn_in_channels,
                                out_channels=128), cnn_head)
            if self.root_basis == 'cnn':
                self.nerf_root_rts = nn.Sequential(
                                Encoder((112,112), in_channels=cnn_in_channels,
                                out_channels=128),
                                RTHead(use_quat=use_quat, D=1,
                                in_channels_xyz=t_embed_dim,in_channels_dir=0,
                                out_channels=out_channels, raw_feat=True))
            elif self.root_basis == 'exp':
                self.nerf_root_rts = RTExplicit(max_t, delta=self.use_cam)
            elif self.root_basis == 'mlp':
                self.root_code = nn.Embedding(max_t, t_embed_dim)
                self.nerf_root_rts = nn.Sequential(self.root_code,
                                RTHead(use_quat=use_quat, 
                                #D=5,W=128,
                                #activation=nn.Tanh(),
                                in_channels_xyz=t_embed_dim,in_channels_dir=0,
                                out_channels=out_channels, raw_feat=True))
            else: print('error'); exit()

        # TODO: change according to multiple video
        ks_list = []
        for i in range(self.num_vid):
            fx,fy,px,py=[float(i) for i in \
                    self.config.get('data_%d'%i, 'ks').split(' ')]
            ks_list.append([fx,fy,px,py])
        self.ks_param = torch.Tensor(ks_list).to(self.device)
        if opts.ks_opt:
            self.ks_param = nn.Parameter(self.ks_param)
            

        # densepose
        if opts.flow_dp:
            with open('mesh_material/geodists_sheep_5004.pkl', 'rb') as f: 
                geodists=pickle.load(f)
                geodists = torch.Tensor(geodists).cuda(self.device)
                geodists[0,:] = np.inf
                geodists[:,0] = np.inf
                self.geodists = geodists
            self.dp_thrd = 0.1 # threshold to fb error of dp
            self.is_flow_dp=True
        else: self.is_flow_dp=False

        with open('mesh_material/sheep_5004.pkl', 'rb') as f:
            dp = pickle.load(f)
            self.dp_verts = dp['vertices']
            self.dp_faces = dp['faces']
            self.dp_verts = torch.Tensor(self.dp_verts).cuda(self.device)
            self.dp_faces = torch.Tensor(self.dp_faces).cuda(self.device).long()
            
            self.dp_verts -= self.dp_verts.mean(0)[None]
            self.dp_verts /= self.dp_verts.abs().max()
            self.dp_verts_unit = self.dp_verts.clone()
            self.dp_verts *= (self.near_far[:,1] - self.near_far[:,0]).mean()/2
            
            # visualize
            self.dp_vis = self.dp_verts
            self.dp_vmin = self.dp_vis.min(0)[0][None]
            self.dp_vis = self.dp_vis - self.dp_vmin
            self.dp_vmax = self.dp_vis.max(0)[0][None]
            self.dp_vis = self.dp_vis / self.dp_vmax
            
            # load surface embedding
            from utils.cselib import create_cse
            detbase='../detectron2/'
            config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml'%(detbase)
            weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl'
            _, _, mesh_vertex_embeddings = create_cse(config_path,
                                                            weight_path)
            self.dp_embed = mesh_vertex_embeddings['sheep_5004']

        if opts.N_importance>0:
            self.nerf_fine = NeRF()
            self.nerf_models['fine'] = self.nerf_fine

        # add densepose mlp
        if opts.use_viser:
            self.num_feat = 16
            # TODO change this to D-8
            self.nerf_feat = NeRF(in_channels_xyz=in_channels_xyz, D=5, W=128,
     out_channels=self.num_feat,in_channels_dir=0, raw_feat=True, init_beta=1.)
            self.nerf_models['nerf_feat'] = self.nerf_feat

        # add uncertainty MLP
        if opts.use_unc:
            # input, (x,y,t)+code, output, (1)
            vid_code_dim=32  # add video-specific code
            self.vid_code = nn.Embedding(self.num_vid, vid_code_dim)
            self.nerf_unc = NeRFUnc(in_channels_xyz=in_channels_xyz, D=5, W=128,
         out_channels=1,in_channels_dir=vid_code_dim, raw_feat=True, init_beta=1.)
            self.nerf_models['nerf_unc'] = self.nerf_unc

        # load densepose surface features
        if opts.warmup_pose_ep>0:
            # soft renderer
            self.mesh_renderer = sr.SoftRenderer(image_size=256, sigma_val=1e-12, 
                           camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                           light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)


        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

    @staticmethod
    def near_far_from_config(config, data_offset, num_vid):
        """
        near_far: N, 2
        """
        near_far = np.zeros((data_offset[-1],2)).astype(np.float32)
        for nvid in range(num_vid):
            near_far[data_offset[nvid]:data_offset[nvid+1]]=\
            [float(i) for i in config.get('data_%d'%nvid, 'near_far').split(',')]
        return near_far

    def nerf_render(self, rtk, kaug, embedid, nsample=256, ndepth=128):
        opts=self.opts
        # render rays
        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()
        Rmat = rtk[:,:3,:3]
        Tmat = rtk[:,:3,3]
        Kmat = K2mat(rtk[:,3,:])
        Kaug = K2inv(kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))
        bs = Kinv.shape[0]
        embedid = embedid.long().to(self.device)[:,None]
        if opts.use_sim3:
            # don't update the canonical frame sim3
            sim3_j2c = torch.cat([self.sim3_j2c[:1].detach(),  
                                  self.sim3_j2c[1:]],0)

        # sample 1x points, sample 4x points for further selection
        nsample_a = 4*nsample
        rand_inds, xys = sample_xy(self.img_size, bs, nsample+nsample_a, self.device, 
                               return_all= not(self.training))
        if self.training:
            rand_inds_a, xys_a = rand_inds[:,nsample:].clone(), xys[:,nsample:].clone()
            rand_inds, xys     = rand_inds[:,:nsample].clone(), xys[:,:nsample].clone()
        
        if opts.debug:
            torch.cuda.synchronize()
            print('initial xy sample time: %.2f'%(time.time()-start_time))

        # importance sampling
        if self.training and opts.use_unc and \
                self.progress > (opts.warmup_init_steps + opts.warmup_steps):
            with torch.no_grad():
                # select .2x points
                nsample_s = nsample//5
                # run uncertainty estimation
                ts = self.frameid_sub.to(self.device) / self.max_ts * 2 -1
                ts = ts[:,None,None].repeat(1,nsample_a,1)
                dataid = self.dataid.long().to(self.device)
                vid_code = self.vid_code(dataid)[:,None].repeat(1,nsample_a,1)
                
                # convert to normalized coords
                xysn = torch.cat([xys_a, torch.ones_like(xys_a[...,:1])],2)
                xysn = xysn.matmul(Kinv.permute(0,2,1))[...,:2]

                xyt = torch.cat([xysn, ts],-1)
                xyt_embedded = self.embedding_xyz(xyt)
                xyt_code = torch.cat([xyt_embedded, vid_code],-1)
                unc_pred = self.nerf_unc(xyt_code)[...,0]
            
                # merge top nsamples
                topk_samp = unc_pred.topk(nsample_s,dim=-1)[1] # bs,nsamp
                xys_a =       torch.stack(      [xys_a[i][topk_samp[i]] for i in range(bs)],0)
                rand_inds_a = torch.stack([rand_inds_a[i][topk_samp[i]] for i in range(bs)],0)
                xys = torch.cat([xys,xys_a],1)
                rand_inds = torch.cat([rand_inds,rand_inds_a],1)
                nsample = nsample + nsample_s

            # TODO visualize samples
            #pdb.set_trace()
            #self.imgs_samp = []
            #for i in range(bs):
            #    self.imgs_samp.append(draw_pts(self.imgs[i], xys_a[i]))
            #self.imgs_samp = torch.stack(self.imgs_samp,0)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('importance sampling time: %.2f'%(time.time()-start_time))
        
        near_far = self.near_far[self.frameid.long()]
        rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
       
        # update rays
        # rays: input to renderer
        rays['img_at_samp'] = torch.stack([self.imgs[i].view(3,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,3
        rays['sil_at_samp'] = torch.stack([self.masks[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        rays['flo_at_samp'] = torch.stack([self.flow[i].view(2,-1).T[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,2
        rays['cfd_at_samp'] = torch.stack([self.occ[i].view(-1,1)[rand_inds[i]]\
                                for i in range(bs)],0) # bs,ns,1
        if opts.use_viser:
            dp_feats_rsmp = resample_dp(self.dp_feats, 
                    self.dp_bbox, kaug, self.img_size)
            feats_at_samp = [dp_feats_rsmp[i].view(self.num_feat,-1).T\
                             [rand_inds[i].long()] for i in range(bs)]
            feats_at_samp = torch.stack(feats_at_samp,0) # bs,ns,num_feat
            rays['feats_at_samp'] = feats_at_samp

        # update rays
        if bs>1:
            rtk_vec = rays['rtk_vec'] # bs, N, 21
            rtk_vec_target = rtk_vec.view(2,-1).flip(0)
            rays['rtk_vec_target'] = rtk_vec_target.reshape(rays['rtk_vec'].shape)
            
            embedid_target = embedid.view(2,-1).flip(0).reshape(-1,1)
            if opts.flowbw:
                time_embedded_target = self.pose_code(embedid_target)
                rays['time_embedded_target'] = time_embedded_target.repeat(1,
                                                            rays['nsample'],1)
            elif opts.lbs and self.num_bone_used>0:
                bone_rts_target = self.nerf_bone_rts(embedid_target)
                rays['bone_rts_target'] = bone_rts_target.repeat(1,rays['nsample'],1)

            if self.is_flow_dp:
                # randomly choose 1 target image
                rays['rtk_vec_dentrg'] = rtk_vec[self.rand_dentrg] # bs,N,21
                embedid_dentrg = embedid.view(-1,1)[self.rand_dentrg]
                if opts.flowbw:
                    time_embedded_dentrg = self.pose_code(embedid_dentrg)
                    rays['time_embedded_dentrg'] = time_embedded_dentrg.repeat(1,
                                                            rays['nsample'],1)
                elif opts.lbs and self.num_bone_used>0:
                    bone_rts_dentrg = self.nerf_bone_rts(embedid_dentrg) #bsxbs,x 
                    rays['bone_rts_dentrg'] = bone_rts_dentrg.repeat(1,rays['nsample'],1)

                if opts.use_sim3:
                    dataid_dentrg = self.dataid[self.rand_dentrg]
                    rays['sim3_j2c_dentrg'] = sim3_j2c[dataid_dentrg.long()]
                    rays['sim3_j2c_dentrg'] = rays['sim3_j2c_dentrg'][:,None].repeat(1,rays['nsample'],1)
                 
        # pass time-dependent inputs
        time_embedded = self.pose_code(embedid)
        rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        if opts.lbs and self.num_bone_used>0:
            bone_rts = self.nerf_bone_rts(embedid)
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

        if opts.use_sim3:
            # pass the canonical to joint space transforms
            rays['sim3_j2c'] = sim3_j2c[self.dataid.long()]
            rays['sim3_j2c'] = rays['sim3_j2c'][:,None].repeat(1,rays['nsample'],1)

        if opts.env_code:
            rays['env_code'] = self.env_code(self.dataid.long().to(self.device))
            rays['env_code'] = rays['env_code'][:,None].repeat(1,rays['nsample'],1)

        if opts.use_unc:
            ts = self.frameid_sub.to(self.device) / self.max_ts * 2 -1
            ts = ts[:,None,None].repeat(1,rays['nsample'],1)
            rays['ts'] = ts
        
            dataid = self.dataid.long().to(self.device)
            vid_code = self.vid_code(dataid)[:,None].repeat(1,rays['nsample'],1)
            rays['vid_code'] = vid_code
            
            xysn = torch.cat([xys, torch.ones_like(xys[...,:1])],2)
            xysn = xysn.matmul(Kinv.permute(0,2,1))[...,:2]
            rays['xysn'] = xysn
        
        if opts.debug:
            torch.cuda.synchronize()
            print('prepare rays time: %.2f'%(time.time()-start_time))

        bs_rays = rays['bs'] * rays['nsample'] # over pixels
        results=defaultdict(list)
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            # decide whether to use fine samples 
            if self.progress > opts.fine_steps:
                self.use_fine = True
            else:
                self.use_fine = False
            rendered_chunks = render_rays(self.nerf_models,
                        self.embeddings,
                        rays_chunk,
                        N_samples = ndepth,
                        use_disp=False,
                        perturb=opts.perturb,
                        noise_std=opts.noise_std,
                        N_importance=opts.N_importance,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        obj_bound=self.latest_vars['obj_bound'],
                        use_fine=self.use_fine,
                        img_size=self.img_size,
                        progress=self.progress,
                        opts=opts,
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v]
        
        for k, v in results.items():
            if v[0].dim()==0: # loss
                v = torch.stack(v).mean()
            else:
                v = torch.cat(v, 0)
                if self.training:
                    v = v.view(bs,nsample,-1)
                else:
                    v = v.view(bs,self.img_size, self.img_size, -1)
            results[k] = v
        if opts.debug:
            torch.cuda.synchronize()
            print('rendering time: %.2f'%(time.time()-start_time))
        
        # viser feature matching
        if opts.use_viser:
            # visualization
            #vis_viser(results, self.masks, self.imgs, 
            #            bs,self.img_size, ndepth)
            #pdb.set_trace()
            
            results['pts_pred'] = (results['pts_pred'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_exp']  = (results['pts_exp'] - torch.Tensor(self.vis_min[None]).\
                    to(self.device)) / torch.Tensor(self.vis_len[None]).to(self.device)
            results['pts_pred'] = results['pts_pred'].clamp(0,1)
            results['pts_exp']  = results['pts_exp'].clamp(0,1)
        del results['xyz_coarse_frame']

        if opts.debug:
            torch.cuda.synchronize()
            print('feature mtaching time: %.2f'%(time.time()-start_time))

        results['joint_render_vis'] = (results['joint_render']-\
                       torch.Tensor(self.vis_min[None,None]).to(self.device))/\
                       torch.Tensor(self.vis_len[None,None]).to(self.device)
        results['joint_render_vis'] = results['joint_render_vis'].clamp(0,1)
        #    pdb.set_trace() 
        #    trimesh.Trimesh(self.dp_verts.cpu(), self.dp_faces,
        #            vertex_colors=self.dp_vis.cpu()).export('0.obj')
        #    trimesh.Trimesh(dp_pts[0].view(-1,3).cpu(),
        # vertex_colors=dp_vis_pred[0].view(-1,3).cpu()).export('1.obj')
        if opts.debug:
            torch.cuda.synchronize()
            print('compute flow time: %.2f'%(time.time()-start_time))
               
        
        return results, rand_inds

    def compute_dp_flow(self, bs):
        """
        assuming the following variable exists in self.
        {frameid, dps , dp_feats, dp_bbox, kaug}
        will generate the following to self.
        {rand_dentrg, dp_flow, dp_conf}
        bs, batch of pairs
        """
        device = self.device
        opts = self.opts
        h = self.img_size
        w = self.img_size
        # choose a forward-backward consistent pair
        is_degenerate_pair = len(set((self.frameid.numpy())))==2
        if is_degenerate_pair:
            rand_dentrg = np.asarray(range(bs))
            rand_dentrg = np.flip(rand_dentrg.reshape((2,-1)),0).flatten()
        else:
            rand_dentrg = -1 * np.ones(bs)
            for idx in range(bs):
                if rand_dentrg[idx] > -1: continue # already assigned
                while True:
                    tidx = np.random.randint(0,bs)
                    if idx!=tidx and rand_dentrg[tidx]==-1: break
                rand_dentrg[idx]  = tidx
                rand_dentrg[tidx] = idx
        self.rand_dentrg = rand_dentrg.astype(int)

        # densepose to correspondence
        geodesic=True
        #geodesic=False
        if geodesic: 
            # densepose geodesic
            # downsample
            h_rszd,w_rszd=h//4,w//4
            dps = F.interpolate(self.dps[:,None], (h_rszd,w_rszd), 
                                    mode='nearest')[:,0]
            dps = dps.long()
        else:
            h_rszd,w_rszd = 112,112
            # densepose-cropped to dataloader cropped transformation
            cropa2b = bbox_dp2rnd(self.dp_bbox, self.kaug)
        
        hw_rszd = h_rszd*w_rszd
        self.dp_flow   = torch.zeros(bs,2,h_rszd,w_rszd).to(device)
        self.dp_conf = torch.zeros(bs,h_rszd,w_rszd).to(device)

        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()

        for idx in range(bs):
            jdx = self.rand_dentrg[idx]
            if self.dp_flow[idx].abs().sum()!=0: continue # already computed 
            if geodesic:
                flo_refr = compute_flow_geodist(dps[idx], dps[jdx], 
                                                self.geodists)
                flo_targ = compute_flow_geodist(dps[jdx], dps[idx], 
                                                self.geodists)
            else:
                flo_refr, flo_targ = compute_flow_cse(self.dp_feats[idx],
                                            self.dp_feats[jdx],
                                            cropa2b[idx], cropa2b[jdx], 
                                            self.img_size)
            self.dp_flow[idx] = flo_refr
            self.dp_flow[jdx] = flo_targ
        if opts.debug:
            torch.cuda.synchronize()
            print('compute dp flow:%.2f'%(time.time()-start_time))

        for idx in range(bs):
            jdx = self.rand_dentrg[idx]
            img_refr = self.imgs[idx:idx+1]
            img_targ = self.imgs[jdx:jdx+1]
            flo_refr = self.dp_flow[idx]
            flo_targ = self.dp_flow[jdx]
            if opts.vis_dpflow:
                save_path = 'tmp/img-%05d-%05d.jpg'%(idx,jdx)
            else: save_path = None
            fberr_fw, fberr_bw = fb_flow_check(flo_refr, flo_targ,
                                               img_refr, img_targ, 
                                                self.dp_thrd,
                                                save_path = save_path)

            self.dp_conf[idx] = torch.Tensor(fberr_fw)
            if self.dataid[idx]==self.dataid[jdx]: # remove flow in same vid
                self.dp_conf[idx] = self.dp_thrd
        self.dp_conf[self.dp_conf>self.dp_thrd] = self.dp_thrd

    def convert_batch_input(self, batch):
        device = self.device
        if batch['img'].dim()==4:
            bs,_,h,w = batch['img'].shape
        else:
            bs,_,_,h,w = batch['img'].shape
        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        
        self.input_imgs   = input_img_tensor.to(device)
        self.imgs         = img_tensor.to(device)
        self.masks        = batch['mask']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        self.vis2d        = batch['vis2d']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
        self.dps          = batch['dp']          .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        dpfd = 16
        dpfs = 112
        self.dp_feats     = batch['dp_feat']     .view(bs,-1,dpfd,dpfs,dpfs).permute(1,0,2,3,4).reshape(-1,dpfd,dpfs,dpfs).to(device)
        self.dp_feats     = F.normalize(self.dp_feats, 2,1)
        self.dp_bbox      = batch['dp_bbox']     .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)    .to(device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.is_canonical = batch['is_canonical'].view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
      
        self.frameid_sub = self.frameid.clone()
        self.embedid = self.frameid + self.data_offset[self.dataid.long()]
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]

        # process silhouette
        self.sils = self.masks.clone()
        if self.opts.bg: self.masks[:] = 1
        self.masks = (self.masks*self.vis2d)>0
        self.masks = self.masks.float()
        self.sils = (self.sils*self.vis2d)>0
        self.sils =  self.sils.float()
        
        self.flow = batch['flow'].view(bs,-1,2,h,w).permute(1,0,2,3,4).reshape(-1,2,h,w).to(device)
#        self.flow = batch['flow'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w).to(device)
#        self.flow = self.flow[:,:2]
        self.occ  = batch['occ'].view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
    
    def convert_root_pose_mhp(self):
        """
        assumes has self.
        {rtk, frameid, dp_feats, dps, masks, kaug }
        produces self.
        {rtk_raw}
        """
        opts = self.opts
        bs = self.rtk.shape[0]
        device = self.device
        
        self.rtk[:,:3] = self.create_base_se3(self.near_far, bs, device)
        frame_code = self.dp_feats
        rts_mhp = self.nerf_root_rts(frame_code) # bs, N, 13

        num_scores = rts_mhp.shape[1]
        rtk_raw = self.rtk[:,None].repeat(1,num_scores, 1,1)
        rtk_raw = rtk_raw.view(bs, num_scores,-1)
        rts_mhp = torch.cat([rts_mhp, rtk_raw],-1)
        return rts_mhp
        
    def convert_root_pose(self):
        """
        assumes has self.
        {rtk, frameid, dp_feats, dps, masks, kaug }
        produces self.
        {rtk_raw}
        """
        opts = self.opts
        bs = self.rtk.shape[0]
        device = self.device
        #TODO change scale of input cameras
        self.rtk[:,:3,3] = self.rtk[:,:3,3] / self.obj_scale
        self.rtk_raw = self.rtk.clone()
        if not self.use_cam:
            self.rtk[:,:3] = self.create_base_se3(self.near_far, bs, device)

        if self.opts.root_opt:
            frameid = self.frameid.long().to(device)
            if self.root_basis == 'cnn':
                if opts.cnn_feature=='embed':
                    frame_code = self.dp_feats
                elif opts.cnn_feature=='pts':
                    # bs, h, w, 3
                    frame_code = self.dp_verts_unit[self.dps.long()]
                    frame_code = frame_code * self.masks[...,None]
                    frame_code = frame_code.permute(0,3,1,2)
                    frame_code = F.interpolate(frame_code, (112,112), mode='bilinear')
                root_rts = self.nerf_root_rts(frame_code)
            elif self.root_basis == 'mlp' or self.root_basis == 'exp':
                root_rts = self.nerf_root_rts(frameid)
            else: print('error'); exit()
            root_rmat = root_rts[:,0,:9].view(-1,3,3)
            root_tmat = root_rts[:,0,9:12]
    
            rmat = self.rtk[:,:3,:3]
            tmat = self.rtk[:,:3,3]
            tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
            rmat = rmat.matmul(root_rmat)
            self.rtk[:,:3,:3] = rmat
            self.rtk[:,:3,3] = tmat

        self.rtk[:,3,:] = self.ks_param[self.dataid.long()] #TODO kmat
       
    def save_latest_vars(self):
        """
        in: self.
        {rtk, kaug, frameid, vis2d}
        out: self.
        {latest_vars}
        """
        rtk = self.rtk.clone().detach()
        Kmat = K2mat(rtk[:,3])
        Kaug = K2inv(self.kaug) # p = Kaug Kmat P
        rtk[:,3] = mat2K(Kaug.matmul(Kmat))

        self.latest_vars['rtk'][self.frameid.long()] = rtk.cpu().numpy()
        if self.opts.use_sim3:
            self.latest_vars['j2c'][self.frameid.long()] = \
                    self.sim3_j2c.detach().cpu().numpy()[self.dataid.long()]
        self.latest_vars['idk'][self.frameid.long()] = 1
        if self.training:
            self.latest_vars['vis'][self.frameid.long()] = self.vis2d.cpu().numpy()

    def set_input(self, batch):
        device = self.device
        opts = self.opts

        self.convert_batch_input(batch)
        bs = self.imgs.shape[0]
        
        if self.is_flow_dp:
            self.compute_dp_flow(bs)
 
        self.convert_root_pose()
       
        self.save_latest_vars()
        
        if self.training and self.opts.anneal_freq:
            alpha = self.num_freqs * \
                self.progress / (opts.warmup_init_steps+opts.warmup_steps)
            if alpha>self.alpha.data[0]:
                self.alpha.data[0] = min(max(3, alpha),self.num_freqs) # alpha from 3 to 10
            self.embedding_xyz.alpha = self.alpha.data[0]
            self.embedding_dir.alpha = self.alpha.data[0]

        return bs

    def forward_default(self, batch):
        opts = self.opts
        if opts.debug:
            torch.cuda.synchronize()
            start_time = time.time()
        bs = self.set_input(batch)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input time:%.2f'%(time.time()-start_time))
        rtk = self.rtk
        kaug= self.kaug
        embedid=self.embedid
        aux_out={}
        
        # Render
        rendered, rand_inds = self.nerf_render(rtk, kaug, embedid, 
                nsample=opts.nsample, ndepth=opts.ndepth)
        
        if opts.debug:
            torch.cuda.synchronize()
            print('set input + render time:%.2f'%(time.time()-start_time))
           
        # image and silhouette loss
        sil_at_samp = rendered['sil_at_samp']
        sil_at_samp_flo = rendered['sil_at_samp_flo']
        img_loss_samp = rendered['img_loss_samp']
        img_loss = img_loss_samp[sil_at_samp[...,0]>0].mean() # eval on valid pts
        sil_loss_samp = rendered['sil_loss_samp']
        sil_loss = opts.sil_wt*sil_loss_samp.mean()
        aux_out['sil_loss'] = sil_loss
        aux_out['img_loss'] = img_loss
        total_loss = img_loss
        if not opts.bg: total_loss = total_loss + sil_loss 

        # flow loss
        if opts.use_corresp:
            flo_loss_samp = rendered['flo_loss_samp']
            flo_loss = flo_loss_samp[sil_at_samp_flo[...,0]].mean() # eval on valid pts
    
            # warm up by only using flow loss to optimize root pose
            if self.pose_update == 0:
                total_loss = total_loss*0. + flo_loss
            else:
                total_loss = total_loss + flo_loss
            aux_out['flo_loss'] = flo_loss
        
        # flow densepose loss
        if self.is_flow_dp:
            rendered_fdp = rendered['fdp_coarse'] # bs,N,2
            fdp_at_samp = []
            dcf_at_samp = []
            for i in range(bs):
                # upsample to same resolution
                dp_flow = F.interpolate(self.dp_flow[i][None], 
                             (self.img_size,self.img_size), mode='bilinear')[0]
                dp_conf = F.interpolate(self.dp_conf[i][None,None], 
                             (self.img_size,self.img_size), mode='bilinear')[0]
                fdp_at_samp.append(dp_flow.view(2,-1).T[rand_inds[i]])
                dcf_at_samp.append(dp_conf.view(-1,1)[rand_inds[i]])
            fdp_at_samp = torch.stack(fdp_at_samp,0)
            dcf_at_samp = torch.stack(dcf_at_samp,0)
            fdp_loss = (rendered_fdp - fdp_at_samp).pow(2).sum(-1)
            fdp_loss = (fdp_loss-0.02).relu() # ignore error < 1/20 unit

            # TODO confidence weighting
            sil_at_samp_fdp = (sil_at_samp>0) & (dcf_at_samp<self.dp_thrd-1e-3)\
#                                & (rendered['fdp_valid']==1)
            dcf_at_samp = (-30*dcf_at_samp).sigmoid()
            if sil_at_samp_fdp.sum()>0:
                dcf_at_samp = dcf_at_samp / dcf_at_samp[sil_at_samp_fdp].mean()
                fdp_loss = fdp_loss * dcf_at_samp[...,0]
                
                fdp_loss = 0.001*fdp_loss[sil_at_samp_fdp[...,0]].mean() # eval on valid pts
                total_loss = total_loss + fdp_loss
                aux_out['fdp_loss'] = fdp_loss
        
        # viser loss
        if opts.use_viser:
            feat_loss = rendered['feat_err'][sil_at_samp>0].mean()*0.1
            #feat_loss = rendered['feat_err'][sil_at_samp>0].mean()*0.02
            total_loss = total_loss + feat_loss
            aux_out['feat_loss'] = feat_loss
            aux_out['beta_feat'] = self.nerf_feat.beta.clone().detach()[0]
        
        if opts.use_proj:
            proj_loss = rendered['proj_err'][sil_at_samp>0].mean()*0.1
            aux_out['proj_loss'] = proj_loss
            if opts.freeze_proj:
                # warm up by only using projection loss to optimize bones
                warmup_weight = (self.progress - opts.proj_start)/(opts.proj_end-opts.proj_start)
                warmup_weight = (warmup_weight - 0.5) * 2
                warmup_weight = np.clip(warmup_weight, 0,1)
                if (self.progress > opts.proj_start and \
                    self.progress < opts.proj_end):
                    total_loss = total_loss*warmup_weight + \
                               2*proj_loss*(1-warmup_weight)
            elif self.progress > (opts.warmup_init_steps + opts.warmup_steps) and\
                 self.progress < 0.8: #TODO change this to a arg
                # only add it after feature volume is trained well
                total_loss = total_loss + proj_loss
        
        # regularization 
        if 'frame_cyc_dis' in rendered.keys():
            # cycle loss
            cyc_loss = rendered['frame_cyc_dis'].mean()
            total_loss = total_loss + cyc_loss
            aux_out['cyc_loss'] = cyc_loss

            # globally rigid prior
            rig_loss = 0.0001*rendered['frame_rigloss'].mean()
            total_loss = total_loss + rig_loss
            aux_out['rig_loss'] = rig_loss

            ## TODO enforcing bone distribution to be close to points within surface
            #if opts.lbs:
            #    bone_density_loss = density_loss(self.nerf_coarse, 
            #                                    self.embedding_xyz, self.bones)
            #    total_loss = total_loss + bone_density_loss
            #    aux_out['bone_density_loss'] = bone_density_loss

            # elastic energy for se3 field / translation field
            if 'elastic_loss' in rendered.keys():
                elastic_loss = rendered['elastic_loss'].mean() * 1e-3
                total_loss = total_loss + elastic_loss
                aux_out['elastic_loss'] = elastic_loss

        if opts.eikonal_loss and self.progress> opts.fine_steps:
            ekl_loss = 1e-6*eikonal_loss(self.nerf_coarse, self.embedding_xyz, 
                                         self.latest_vars['obj_bound'])
            total_loss = total_loss + ekl_loss
            aux_out['ekl_loss'] = ekl_loss

        # bone location regularization: pull bones away from empth space (low sdf)
        if opts.lbs and opts.bone_loc_reg:
            bone_xyz_embed = self.embedding_xyz(self.bones[:,None,:3])
            sdf_at_bone = evaluate_mlp(self.nerf_coarse, bone_xyz_embed,
                                            sigma_only=True)
            #sdf_at_bone = evaluate_mlp(self.nerf_vis, bone_xyz_embed)
            bone_loc_loss = 0.01*F.relu(-sdf_at_bone).mean()
            total_loss = total_loss + bone_loc_loss
            #bone_loc_loss = grad_update_bone(self.bones, self.embedding_xyz,
            #                        self.nerf_vis, opts.learning_rate)
            aux_out['bone_loc_loss'] = bone_loc_loss
            #mesh_rest = self.latest_vars['mesh_rest']
            #if len(mesh_rest.vertices)>100: # not a degenerate mesh
            #    mesh_rest = structures.meshes.Meshes(
            #            verts=torch.Tensor(mesh_rest.vertices[None]),
            #            faces=torch.Tensor(mesh_rest.faces[None]))
            #    shape_samp = pytorch3d.ops.sample_points_from_meshes(mesh_rest,
            #                            1000, return_normals=False)
            #    shape_samp = shape_samp[0].to(self.device)
            #    samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            #    bone_loc_loss = 0.1*samploss(self.bones[:,:3], shape_samp).mean()
            #    total_loss = total_loss + bone_loc_loss
            #    aux_out['bone_loc_loss'] = bone_loc_loss
            
        # visibility loss
        if 'vis_loss' in rendered.keys():
            vis_loss = 0.01*rendered['vis_loss'].mean()
            total_loss = total_loss + vis_loss
            aux_out['visibility_loss'] = vis_loss

        #TODO regularize nerf-skin

        # uncertainty MLP inference
        if opts.use_unc:
            # add uncertainty MLP loss, loss = | |img-img_r|*sil - unc_pred |
            unc_pred = rendered['unc_pred']
            unc_loss = ((sil_at_samp[...,0]*img_loss_samp.sum(-1)).detach() -\
                                unc_pred[...,0]).pow(2)
            unc_loss = unc_loss.mean()
            aux_out['unc_loss'] = unc_loss
            total_loss = total_loss + unc_loss

        # save some variables
        if opts.lbs:
            aux_out['skin_scale'] = self.skin_aux[0].clone().detach()
            aux_out['skin_const'] = self.skin_aux[1].clone().detach()
        aux_out['total_loss'] = total_loss
        aux_out['beta'] = self.nerf_coarse.beta.clone().detach()[0]
        return total_loss, aux_out

    def forward_warmup(self, batch):
        """
        batch variable is not never being used here
        """
        # render ground-truth data
        opts = self.opts
        bs_rd = 16
        with torch.no_grad():
            if self.opts.cnn_feature=='pts':
                vertex_color = self.dp_verts_unit
            elif self.opts.cnn_feature=='embed':
                vertex_color = self.dp_embed
            dp_feats_rd, rtk_raw = self.render_dp(self.dp_verts_unit, 
                    self.dp_faces, vertex_color, self.near_far, self.device, 
                    self.mesh_renderer, bs_rd)

        aux_out={}
        if opts.cnn_type=='reg':
            # predict delta se3
            root_rts = self.nerf_root_rts(dp_feats_rd)
            root_rmat = root_rts[:,0,:9].view(-1,3,3)
            root_tmat = root_rts[:,0,9:12]    

            # construct base se3
            rtk = torch.zeros(bs_rd, 4,4).to(self.device)
            rtk[:,:3] = self.create_base_se3(self.near_far, bs_rd, self.device)

            # compose se3
            rmat = rtk[:,:3,:3]
            tmat = rtk[:,:3,3]
            tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
            rmat = rmat.matmul(root_rmat)
            rtk[:,:3,:3] = rmat
            rtk[:,:3,3] = tmat.detach() # do not train translation
        
            # loss
            total_loss = rtk_loss(rtk, rtk_raw, aux_out)

        elif opts.cnn_type=='cls':
            scores, grid = self.nerf_root_rts(dp_feats_rd)
            total_loss = rtk_cls_loss(scores, grid, rtk_raw, aux_out)
        
        aux_out['total_loss'] = total_loss

        return total_loss, aux_out
    
    @staticmethod
    def render_dp(dp_verts_unit, dp_faces, dp_embed, near_far, device, 
                  mesh_renderer, bs):
        """
        render a pair of (densepose feature bsx16x112x112, se3)
        input is densepose surface model and near-far plane
        """
        verts = dp_verts_unit
        faces = dp_faces
        dp_embed = dp_embed
        num_verts, embed_dim = dp_embed.shape
        img_size = 256
        crop_size = 112
        focal = 2
        std_rot = 6.28 # rotation std
        std_dep = 0.5 # depth std


        # scale geometry and translation based on near-far plane
        d_mean = near_far.mean()
        verts = verts / 3 * d_mean # scale based on mean depth
        dep_rand = 1 + np.random.normal(0,std_dep,bs)
        dep_rand = torch.Tensor(dep_rand).to(device)
        d_obj = d_mean * dep_rand
        d_obj = torch.max(d_obj, 1.2*1/3 * d_mean)
        
        # set cameras
        rot_rand = np.random.normal(0,std_rot,(bs,3))
        rot_rand = torch.Tensor(rot_rand).to(device)
        Rmat = transforms.axis_angle_to_matrix(rot_rand)
        Tmat = torch.cat([torch.zeros(bs, 2).to(device), d_obj[:,None]],-1)
        K =    torch.Tensor([[focal,focal,0,0]]).to(device).repeat(bs,1)
        
        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        Kimg = torch.Tensor([[focal*img_size/2.,focal*img_size/2.,img_size/2.,
                            img_size/2.]]).to(device).repeat(bs,1)
        rtk = torch.zeros(bs,4,4).to(device)
        rtk[:,:3,:3] = Rmat
        rtk[:,:3, 3] = Tmat
        rtk[:,3, :]  = Kimg

        # repeat mesh
        verts = verts[None].repeat(bs,1,1)
        faces = faces[None].repeat(bs,1,1)
        dp_embed = dp_embed[None].repeat(bs,1,1)

        # obj-cam transform 
        verts = obj_to_cam(verts, Rmat, Tmat)
        
        # pespective projection
        verts = pinhole_cam(verts, K)
        
        # render sil+rgb
        rendered = []
        for i in range(0,embed_dim,3):
            dp_chunk = dp_embed[...,i:i+3]
            dp_chunk_size = dp_chunk.shape[-1]
            if dp_chunk_size<3:
                dp_chunk = torch.cat([dp_chunk,
                    dp_embed[...,:(3-dp_chunk_size)]],-1)
            rendered_chunk = render_color(mesh_renderer, verts, faces, 
                    dp_chunk,  texture_type='vertex')
            rendered_chunk = rendered_chunk[:,:3]
            rendered.append(rendered_chunk)
        rendered = torch.cat(rendered, 1)
        rendered = rendered[:,:embed_dim]

        # resize to bounding box
        rendered_crops = []
        for i in range(bs):
            mask = rendered[i].max(0)[0]>0
            mask = mask.cpu().numpy()
            indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
            center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
            length = ( int((xid.max()-xid.min())*1.//2 ), 
                      int((yid.max()-yid.min())*1.//2  ))
            left,top,w,h = [center[0]-length[0], center[1]-length[1],
                    length[0]*2, length[1]*2]
            rendered_crop = torchvision.transforms.functional.resized_crop(\
                    rendered[i], top,left,h,w,(50,50))
            # mask augmentation
            rendered_crop = mask_aug(rendered_crop)

            rendered_crops.append( rendered_crop)
            #cv2.imwrite('%d.png'%i, rendered_crop.std(0).cpu().numpy()*1000)

        rendered_crops = torch.stack(rendered_crops,0)
        rendered_crops = F.interpolate(rendered_crops, (crop_size, crop_size), 
                mode='bilinear')
        rendered_crops = F.normalize(rendered_crops, 2,1)
        return rendered_crops, rtk

    @staticmethod
    def create_base_se3(near_far, bs, device):
        """
        create a base se3 based on near-far plane
        """
        rt = torch.zeros(bs,3,4).to(device)
        rt[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(device)
        rt[:,:2,3] = 0.
        if near_far is None:
            rt[:,2,3] = 3. # TODO heuristics of depth=3xobject size
        else:
            rt[:,2,3] = near_far.mean() # TODO need to deal with multi-videl
        return rt

