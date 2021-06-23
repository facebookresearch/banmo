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

from ext_nnutils.train_utils import Trainer
from nnutils.vis_utils import image_grid
from dataloader import frameloader


class v2s_trainer(Trainer):
    def define_model(self, is_eval=False):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = mesh_net.v2s_net(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=None)
        
        if opts.model_path!='':
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs,model_path = opts.model_path, freeze_shape=opts.freeze_shape, finetune=opts.finetune)

        if is_eval:
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
        params=[]
        for name,p in self.model.named_parameters():
            params.append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': params},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
           [opts.learning_rate,
            ],
            200*len(self.dataloader), pct_start=0.01, 
            cycle_momentum=False, anneal_strategy='linear',
            final_div_factor=1./25, div_factor = 25,
            )
   
    def eval(self, num_eval=9, dynamic=False): 
        """
        dynamic: whether to render canonical shape, or dynamic shape
        """
        with torch.no_grad():
            self.model.eval()

            # run marching cubes
            mesh_mc = self.extract_mesh(self.model, self.opts.chunk)

            # render a video
            skip_int = len(self.evalloader)//num_eval
            rendered_seq = defaultdict(list)
            mesh_seq=[]
            rtk_seq=[]
            for i,batch in enumerate(self.evalloader):
                if i%skip_int==0:
                    rendered = self.render_vid(self.model, batch)
                    for k, v in rendered.items():
                        rendered_seq[k] += [v]

                    # run marching cubes
                    if dynamic:
                        mesh_mc = self.extract_mesh(self.model, self.opts.chunk)
                    mesh_seq.append(mesh_mc)

                    # save cams
                    rtk_seq.append(self.model.rtk)

            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)

        return rendered_seq, mesh_seq, rtk_seq
    
    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.name), comment=opts.name)
        total_steps = 0
        dataset_size = len(self.dataloader)
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)
        #if opts.local_rank==0:        self.save('0')

        # start training
        for epoch in range(0, opts.num_epochs):
            epoch_iter = 0
            self.model.epoch = epoch
            self.model.ep_iters = len(self.dataloader)
            
            # evaluation
            rendered_seq, mesh_seq, rtk_seq = self.eval()                
            mesh_seq[0].export('/private/home/gengshany/dropbox/output/0.obj')
            rgb_coarse = image_grid(rendered_seq['rgb_coarse'], 3,3)
            sil_coarse = image_grid(rendered_seq['opacity_coarse'], 3,3)
            self.add_image(log, 'rendered_img', rgb_coarse, epoch, scale=False)
            self.add_image(log, 'rendered_sil', sil_coarse, epoch, scale=False)

            self.model.train()
            for i, batch in enumerate(self.dataloader):
                print(i)
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

                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0: 
                    self.save_logs(log, aux_out, total_steps, epoch)

            #if (epoch+1) % opts.save_epoch_freq == 0:
            #    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
            #    self.save('latest')
            #    self.save(epoch+1)
   
    @staticmethod 
    def render_vid(model, batch):
        model.set_input(batch)
        rendered, _ = model.nerf_render()
        return rendered  

    @staticmethod
    def extract_mesh(model,chunk, grid_size=256, threshold=0.5, bound=1.2):
        pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
        query_xyz = np.stack(np.meshgrid(pts, pts, pts), -1)
        query_xyz = torch.Tensor(query_xyz).to(model.device).view(-1, 3)
        query_dir = torch.zeros_like(query_xyz)

        bs_pts = query_xyz.shape[0]
        out_chunks = []
        for i in range(0, bs_pts, chunk):
            xyz_embedded = model.embedding_xyz(query_xyz[i:i+chunk]) # (N, embed_xyz_channels)
            dir_embedded = model.embedding_dir(query_dir[i:i+chunk]) # (N, embed_dir_channels)
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
        return mesh

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
