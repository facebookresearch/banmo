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
from dataloader import frameloader as mvid_data

from ext_nnutils.train_utils import Trainer


class v2s_trainer(Trainer):
    def define_model(self):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.v2s_net(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=None)
        
        if opts.model_path!='':
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs,model_path = opts.model_path, freeze_shape=opts.freeze_shape, finetune=opts.finetune)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = self.model.to(device)

        self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opts.local_rank],
                output_device=opts.local_rank,
                find_unused_parameters=True,
        )
        return
    
    def init_training(self):
        opts = self.opts
        params=[]
        for name,p in self.model.module.named_parameters():
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
    
    @staticmethod
    def add_image(log,tag,data,step,scale=True):
        if tag in data.keys():
            timg = data[tag][0].detach().cpu().numpy()
            if scale:
                timg = (timg-timg.min())/(timg.max()-timg.min())
    
            if len(timg.shape)==2:
                formats='HW'
            elif timg.shape[0]==3:
                formats='CHW'
            else:
                formats='HWC'
            log.add_image(tag,timg,step,dataformats=formats)

    
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
            self.model.module.epoch = epoch
            self.model.module.ep_iters = len(self.dataloader)

            for i, batch in enumerate(self.dataloader):
                print(i)
                self.model.module.iters=i
                self.model.module.total_steps = total_steps

                if self.opts.debug:
                    torch.cuda.synchronize()
                    start_time = time.time()

                self.optimizer.zero_grad()
                total_loss,aux_output = self.model(batch)
                total_loss.mean().backward()
                
                if self.opts.debug:
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

                ## gradient clipping

                #if opts.local_rank==0 and torch.isnan(self.model.module.total_loss):
                #    pdb.set_trace()
                self.optimizer.step()
                self.scheduler.step()

                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0:
                    if 'total_loss' in aux_output.keys():
                        log.add_scalar('train/total_loss',  aux_output['total_loss'].mean(), total_steps)
                    if 'sil_loss' in aux_output.keys():
                        log.add_scalar('train/sil_loss',  aux_output['sil_loss'].mean(), total_steps)
                    if 'img_loss' in aux_output.keys():
                        log.add_scalar('train/img_loss',  aux_output['img_loss'].mean(), total_steps)
                    if i==0:
                        self.add_image(log, 'rendered_img', aux_output, epoch,scale=True)
                        self.add_image(log, 'rendered_sil', aux_output, epoch,scale=True)

            #if (epoch+1) % opts.save_epoch_freq == 0:
            #    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
            #    self.save('latest')
            #    self.save(epoch+1)
