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
from torch.nn.utils import clip_grad_norm_

from nnutils.geom_utils import lbs, reinit_bones
from ext_nnutils.train_utils import Trainer
from ext_utils.flowlib import flow_to_image
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
        params_nerf_coarse=[]
        params_nerf_fine=[]
        params_nerf_flowbw=[]
        params_nerf_root_rts=[]
        params_nerf_bone_rts=[]
        params_embed=[]
        params_bones=[]
        for name,p in self.model.named_parameters():
            if 'nerf_coarse' in name:
                params_nerf_coarse.append(p)
            elif 'nerf_fine' in name:
                params_nerf_fine.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw':
                params_nerf_flowbw.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_bone_rts' in name:
                params_nerf_bone_rts.append(p)
            elif 'embedding_time' in name:
                params_embed.append(p)
            elif 'bones' == name:
                params_bones.append(p)
            else: continue
            print(name)
                
        self.optimizer = torch.optim.AdamW(
            [{'params': params_nerf_coarse},
             {'params': params_nerf_fine},
             {'params': params_nerf_flowbw},
             {'params': params_nerf_root_rts},
             {'params': params_nerf_bone_rts},
             {'params': params_embed},
             {'params': params_bones},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
           [opts.learning_rate, # params_nerf_coarse
            opts.learning_rate, # params_nerf_fine
            opts.learning_rate, # params_nerf_flowbw
            opts.learning_rate, # params_nerf_root_rts
            opts.learning_rate, # params_nerf_bone_rts
            opts.learning_rate, # params_embed
            opts.learning_rate, # params_bones
            ],
            opts.num_epochs * len(self.dataloader),
            pct_start=2./opts.num_epochs, # use 2 epochs to warm up
            cycle_momentum=False, 
            anneal_strategy='linear',
            final_div_factor=1./5, div_factor = 25,
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
            aux_seq = {'mesh_rest': mesh_dict['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'idx':[],
                       'bone':[],}
            for i,batch in enumerate(self.evalloader):
                if i in idx_render:
                    rendered = self.render_vid(self.model, batch)
                    for k, v in rendered.items():
                        rendered_seq[k] += [v]

                    # save images
                    rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:1]]
                    rendered_seq['sil'] += [self.model.masks[...,None]      [:1]]
                    rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:1]]
                    rendered_seq['flo_coarse'][-1] *= rendered_seq['sil_coarse'][-1]

                    # run marching cubes
                    if dynamic_mesh:
                        mesh_dict = self.extract_mesh(self.model,self.opts.chunk,
                                            self.opts.sample_grid3d, frameid=i)
                    aux_seq['mesh'].append(mesh_dict['mesh'])
                    # save bones
                    if 'bones' in mesh_dict.keys():
                        aux_seq['bone'].append(mesh_dict['bones'][0].cpu().numpy())

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[0].cpu().numpy())
                    
                    # save image list
                    aux_seq['idx'].append(self.model.frameid[0])


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
            mesh_rest = aux_seq['mesh'][0]
            mesh_rest.export(mesh_file)
            for k,v in rendered_seq.items():
                grid_img = image_grid(rendered_seq[k],3,3)
                self.add_image(log, k, grid_img, epoch, scale=False)
               
            # reinit bones based on extracted surface
            # TODO: reinit_bone_rts(self.model.nerf_bone_rts)
            if opts.lbs and epoch==10:
                bones_reinit = reinit_bones(self.model.num_bones, mesh_rest, 
                                        self.device)
                self.model.bones.data  = bones_reinit
 
            # training loop
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
                nerf_coarse_grad = []
                nerf_root_rts_grad = []
                for name,p in self.model.named_parameters():
                    if 'nerf_coarse' in name:
                        nerf_coarse_grad.append(p)
                    elif 'nerf_root_rts' in name:
                        nerf_root_rts_grad.append(p)
                aux_out['nerf_coarse_g'] = clip_grad_norm_(nerf_coarse_grad, .1)
                aux_out['nerf_root_rts_g'] = clip_grad_norm_(nerf_root_rts_grad, .1)
                

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
        for k,v in rendered.items():
            rendered[k] = v[:1]
        return rendered  

    @staticmethod
    def extract_mesh(model,chunk,grid_size,frameid=None,threshold=0.5,bound=1.5):
        opts = model.opts
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
          
            # backward warping 
            if frameid is not None and not opts.queryfw:
                query_time = torch.ones(chunk,1).to(model.device)*frameid
                query_time = query_time.long()
                if opts.flowbw:
                    # flowbw
                    xyz_embedded = model.embedding_xyz(query_xyz_chunk)
                    time_embedded = model.embedding_time(query_time)[:,0]
                    xyztime_embedded = torch.cat([xyz_embedded, time_embedded],1)

                    flowbw_chunk = model.nerf_flowbw(xyztime_embedded)
                    query_xyz_chunk += flowbw_chunk
                elif opts.lbs:
                    # backward skinning
                    bones = model.bones
                    query_xyz_chunk = query_xyz_chunk[:,None]
                    bone_rts_fw = model.nerf_bone_rts(query_time)

                    query_xyz_chunk,_,bones_dfm = lbs(bones, 
                                                  bone_rts_fw,
                                                  query_xyz_chunk)

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
        vertices = (vertices - grid_size/2)/grid_size*2*bound

        # forward warping
        if frameid is not None and opts.queryfw:
            num_pts = vertices.shape[0]
            query_time = torch.ones(num_pts,1).long().to(model.device)*frameid
            pts_can=torch.Tensor(vertices).to(model.device)
            if opts.flowbw:
                # forward flow
                pts_can_embedded = model.embedding_xyz(pts_can)
                time_embedded = model.embedding_time(query_time)[:,0]
                ptstime_embedded = torch.cat([pts_can_embedded, time_embedded],1)

                pts_dfm = pts_can + model.nerf_flowfw(ptstime_embedded)
            elif opts.lbs:
                # forward skinning
                bones = model.bones
                pts_can = pts_can[:,None]
                bone_rts_fw = model.nerf_bone_rts(query_time)

                pts_dfm,_,bones_dfm = lbs(bones, bone_rts_fw, pts_can,backward=False)
                pts_dfm = pts_dfm[:,0]
                rt_dict['bones'] = bones_dfm
            vertices = pts_dfm.cpu().numpy()
                

        mesh = trimesh.Trimesh(vertices, triangles)
        if len(mesh.vertices)>0:
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
        rt_dict['mesh'] = mesh
        return rt_dict

    def save_logs(self, log, aux_output, total_steps, epoch):
        for k,v in aux_output.items():
            self.add_scalar(log, k, aux_output,total_steps)
            
    @staticmethod
    def add_image(log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """
        if 'flo' in tag: 
            timg = timg.detach().cpu().numpy()
            timg = flow_to_image(timg)
        if scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
            print('error'); pdb.set_trace()
        else:
            formats='HWC'

        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)
