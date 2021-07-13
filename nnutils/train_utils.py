"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags
import cv2

import mcubes
import soft_renderer as sr
from nnutils import mesh_net
import subprocess
from torch.utils.tensorboard import SummaryWriter
from kmeans_pytorch import kmeans
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
import chamfer3D.dist_chamfer_3D
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms

from nnutils.geom_utils import blend_skinning_bw
from ext_nnutils.train_utils import Trainer
from nnutils.vis_utils import image_grid
from dataloader import frameloader


class v2s_trainer(Trainer):
    def __init__(self, opts):
        self.opts = opts
        self.local_rank = opts.local_rank
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        # write logs
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            with open(log_file, 'w') as f:
                for k in dir(opts): f.write('{}: {}\n'.format(k, opts.__getattr__(k)))

    def define_model(self, no_ddp=False):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = mesh_net.v2s_net(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=None)

        if opts.model_path!='':
            self.load_network(opts.model_path)

        if no_ddp:
            self.model = self.model.to(self.device)
        else:
            # ddp
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = self.model.to(self.device)

            self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[opts.local_rank],
                    output_device=opts.local_rank,
                    find_unused_parameters=True,
            )
            self.model = self.model.module
        return
    
    def init_dataset(self):
        self.dataloader = frameloader.data_loader(self.opts)
        self.evalloader = frameloader.eval_loader(self.opts)
    
    def init_training(self):
        opts = self.opts
        params_nerf=[]
        params_embed=[]
        params_bones=[]
        for name,p in self.model.named_parameters():
            if 'nerf' in name:
                params_nerf.append(p)
            elif 'embedding_time' in name:
                params_embed.append(p)
            elif 'bones' == name:
                params_bones.append(p)
            else: continue
            print(name)
                
        self.optimizer = torch.optim.AdamW(
            [{'params': params_nerf},
             {'params': params_embed},
             {'params': params_bones},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
           [opts.learning_rate, # params_nerf
            opts.learning_rate*10, # params_embed
            opts.learning_rate*10, # params_bones
            ],
            200*len(self.dataloader), pct_start=0.01, 
            cycle_momentum=False, anneal_strategy='linear',
            final_div_factor=1./25, div_factor = 25,
            )
    
    def save_network(self, epoch_label):
        if self.opts.local_rank==0:
            save_filename = 'params_{}.pth'.format(epoch_label)
            save_path = os.path.join(self.save_dir, save_filename)
            save_dict = self.model.state_dict()
            torch.save(save_dict, save_path)
            return
    
    def load_network(self,model_path=None):
        states = torch.load(model_path,map_location='cpu')
        self.model.load_state_dict(states, strict=False)
        return
   
    def eval(self, num_view=9, dynamic_mesh=False): 
        """
        num_view: number of views to render
        dynamic_mesh: whether to extract canonical shape, or dynamic shape
        """
        with torch.no_grad():
            self.model.eval()

            # run marching cubes
            mesh_dict = self.extract_mesh(self.model, self.opts.chunk, \
                                                    self.opts.sample_grid3d)

            # render a grid image or the whold video
            if num_view>0:
                idx_render = np.linspace(0,len(self.evalloader)-1,num_view, dtype=int)
            else:
                idx_render = np.asarray(range(len(self.evalloader)))

            # render and save intermediate outputs
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh':[],
                       'rtk':[],
                       'idx':[],
                       'bone':[],}
            for i,batch in enumerate(self.evalloader):
                if i in idx_render:
                    rendered = self.render_vid(self.model, batch)
                    for k, v in rendered.items():
                        rendered_seq[k] += [v]

                    # save images
                    rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)]
                    rendered_seq['sil'] += [self.model.masks[...,None]]

                    # run marching cubes
                    if dynamic_mesh:
                        mesh_dict = self.extract_mesh(self.model,self.opts.chunk,
                                            self.opts.sample_grid3d, frameid=i)
                    aux_seq['mesh'].append(mesh_dict['mesh'])

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[0].cpu().numpy())
                    
                    # save image list
                    aux_seq['idx'].append(self.model.frameid)

                    # save bones
                    if 'bones' in mesh_dict.keys():
                        aux_seq['bone'].append(mesh_dict['bones'][0].cpu().numpy())

            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)

        return rendered_seq, aux_seq
    
    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.logname), comment=opts.logname)
        total_steps = 0
        dataset_size = len(self.dataloader)
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)
        self.save_network('0')

        # start training
        for epoch in range(0, opts.num_epochs):
            epoch_iter = 0
            self.model.epoch = epoch
            self.model.ep_iters = len(self.dataloader)
            
            # evaluation
            rendered_seq, aux_seq = self.eval()                
            mesh_file = os.path.join(self.save_dir, '%s.obj'%opts.logname)
            aux_seq['mesh'][0].export(mesh_file)
            for k,v in rendered_seq.items():
                grid_img = image_grid(rendered_seq[k],3,3)
                self.add_image(log, k, grid_img, epoch, scale=False)
                
#            rgb_coarse = image_grid(rendered_seq['rgb_coarse'], 3,3)
#            sil_coarse = image_grid(rendered_seq['opacity_coarse'], 3,3)
#            self.add_image(log, 'rendered_img', rgb_coarse, epoch, scale=False)
#            self.add_image(log, 'rendered_sil', sil_coarse, epoch, scale=False)

            self.model.train()
            for i, batch in enumerate(self.dataloader):
                self.model.iters=i
                self.model.total_steps = total_steps

                if self.opts.debug:
                    torch.cuda.synchronize()
                    start_time = time.time()

                self.optimizer.zero_grad()
                total_loss,aux_out = self.model(batch)
                total_loss.mean().backward()
                
                if self.opts.debug:
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

                ## gradient clipping

                self.optimizer.step()
                self.scheduler.step()

                #for param_group in self.optimizer.param_groups:
                #    print(param_group['lr'])

                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0: 
                    self.save_logs(log, aux_out, total_steps, epoch)

            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save_network('latest')
                self.save_network(epoch+1)
   
    @staticmethod 
    def render_vid(model, batch):
        opts=model.opts
        model.set_input(batch)
        rtk = model.rtk
        kaug=model.kaug
        frameid=model.frameid
        render_size=64
        kaug[:,:2] *= opts.img_size/render_size

        rendered, _ = model.nerf_render(rtk, kaug, frameid, render_size)
        return rendered  

    @staticmethod
    def extract_mesh(model,chunk,grid_size,frameid=None,threshold=0.5,bound=1.2):
        rt_dict = {}
        pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
        query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
        query_yxz = torch.Tensor(query_yxz).to(model.device).view(-1, 3)
        query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)
        query_dir = torch.zeros_like(query_xyz)

        bs_pts = query_xyz.shape[0]
        out_chunks = []
        for i in range(0, bs_pts, chunk):
            query_xyz_chunk = query_xyz[i:i+chunk]
            query_dir_chunk = query_dir[i:i+chunk]
           
            if frameid is not None:
                if model.opts.flowbw:
                    # flowbw
                    query_time = torch.ones(chunk,1).long().to(model.device)*frameid
                    xyz_embedded = model.embedding_xyz(query_xyz_chunk)
                    time_embedded = model.embedding_time(query_time)[:,0]
                    xyztime_embedded = torch.cat([xyz_embedded, time_embedded],1)
                    flowbw_chunk = model.nerf_flowbw(xyztime_embedded)
                    flowbw_chunk =  flowbw_chunk[:,:3]
                    query_xyz_chunk += flowbw_chunk
                elif model.opts.lbs:
                    # backward skinning
                    bones = model.bones
                    B = bones.shape[-2]
                    embedding_time = model.embedding_time
                    query_time = torch.ones(chunk,1).long().to(model.device)*frameid
                    time_embedded = embedding_time(query_time) 
                    time_embedded = time_embedded.view(-1,B,7)# B,7            
                    rquat=time_embedded[:,:,:4]
                    tmat= time_embedded[:,:,4:7]*0.1

                    rquat[:,:,0]+=10
                    rquat=F.normalize(rquat,2,2)
                    rmat=transforms.quaternion_to_matrix(rquat) 

                    #bones=torch.cat([bones[:,:4], 
                    #                torch.zeros(B,6).to(bones.device)],-1)
                    #rmat=torch.eye(3).to(rquat.device).view(1,1,3,3).repeat(rquat.shape[0],B,1,1)
                    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    
                    query_xyz_chunk = query_xyz_chunk[:,None]
                    query_xyz_chunk,skin,bones_dfm = blend_skinning_bw(bones, rts_fw, query_xyz_chunk)
                    query_xyz_chunk = query_xyz_chunk[:,0]
                    
                    rt_dict['bones'] = bones_dfm 
                
            xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
            dir_embedded = model.embedding_dir(query_dir_chunk) # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [model.nerf_coarse(xyzdir_embedded)]
        vol_rgbo = torch.cat(out_chunks, 0)

        vol_o = vol_rgbo[...,-1].view(grid_size, grid_size, grid_size)
        print('fraction occupied:', (vol_o > threshold).float().mean())
        vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
        vertices = (vertices - grid_size/2)/grid_size*2

        mesh = trimesh.Trimesh(vertices, triangles)
        if len(mesh.vertices)>0:
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        rt_dict['mesh'] = mesh
        return rt_dict

    def save_logs(self, log, aux_output, total_steps, epoch):
        self.add_scalar(log,'total_loss', aux_output,total_steps)
        self.add_scalar(log,'sil_loss', aux_output,total_steps)
        self.add_scalar(log,'img_loss', aux_output,total_steps)
    
    @staticmethod
    def add_image_from_dict(log,tag,data,step,scale=True):
        if tag in data.keys():
            timg = data[tag][0].detach().cpu().numpy()
            add_image(log, tag, timg, step, scale)

    @staticmethod
    def add_image(log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """
        if scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
        else:
            formats='HWC'
        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)
