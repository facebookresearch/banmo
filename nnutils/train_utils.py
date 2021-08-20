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
import time

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

from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam
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

    def define_model(self, data_info, no_ddp=False, half_bones=False):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = mesh_net.v2s_net(img_size, opts, data_info,
                                     half_bones=half_bones)

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

        data_info = {}
        # compute data offset
        dataset_list = self.evalloader.dataset.datasets
        data_offset = [0]
        impath = []
        for dataset in dataset_list:
            #data_offset.append( 100 )
            impath += dataset.imglist
            data_offset.append(len(dataset.imglist))
        data_info['offset'] = np.asarray(data_offset)[:-1]
        data_info['impath'] = impath
        return data_info
    
    def init_training(self):
        opts = self.opts
        self.model.final_steps = opts.num_epochs * len(self.dataloader)

        params_nerf_coarse=[]
        params_nerf_fine=[]
        params_nerf_flowbw=[]
        params_nerf_root_rts=[]
        params_nerf_bone_rts=[]
        params_embed=[]
        params_bones=[]
        params_ks=[]
        params_nerf_dp=[]
        params_sim3_j2c=[]
        params_dp_verts=[]
        for name,p in self.model.named_parameters():
            if 'nerf_coarse' in name:
                params_nerf_coarse.append(p)
            elif 'nerf_fine' in name:
                params_nerf_fine.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                params_nerf_flowbw.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_bone_rts' in name:
                params_nerf_bone_rts.append(p)
            elif 'embedding_time' in name:
                params_embed.append(p)
            elif 'bones' == name:
                params_bones.append(p)
            elif 'ks' == name:
                params_ks.append(p)
            elif 'nerf_dp' in name:
                params_nerf_dp.append(p)
            elif 'sim3_j2c' == name:
                params_sim3_j2c.append(p)
            elif 'dp_verts' == name:
                params_dp_verts.append(p)
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
             {'params': params_ks},
             {'params': params_nerf_dp},
             {'params': params_sim3_j2c},
             {'params': params_dp_verts},
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
            opts.learning_rate, # params_ks
            opts.learning_rate, # params_nerf_dp
            opts.learning_rate, # params_sim3_j2c
          0*opts.learning_rate, # params_dp_verts
            ],
            self.model.final_steps,
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
        opts = self.opts
        with torch.no_grad():
            self.model.eval()

            # run marching cubes
            mesh_dict_rest = self.extract_mesh(self.model, opts.chunk, \
                                                    opts.sample_grid3d)

            # render a grid image or the whold video
            if num_view>0:
                idx_render = np.linspace(0,len(self.evalloader)-1,num_view, dtype=int)
            else:
                idx_render = np.asarray(range(len(self.evalloader)))

            # render and save intermediate outputs
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'sim3_j2c':[],
                       'impath':[],
                       'bone':[],}

            for i in idx_render:
                batch = self.evalloader.dataset[i]
                batch = self.evalloader.collate_fn([batch])
                print('extracting frame %d'%(i))
                rendered = self.render_vid(self.model, batch)
                for k, v in rendered.items():
                    rendered_seq[k] += [v]


                # save images
                rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:1]]
                rendered_seq['sil'] += [self.model.masks[...,None]      [:1]]
                rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:1]]
                rendered_seq['dpc'] += [self.model.dps[...,None]      [:1]]
                rendered_seq['occ'] += [self.model.occ[...,None]      [:1]]
                rendered_seq['flo_coarse'][-1] *= rendered_seq['sil_coarse'][-1]

                # run marching cubes
                if dynamic_mesh:
                    if not opts.queryfw:
                       mesh_dict_rest=None 
                    mesh_dict = self.extract_mesh(self.model,opts.chunk,
                                        opts.sample_grid3d, 
                                    frameid=i, mesh_dict_in=mesh_dict_rest)
                    mesh=mesh_dict['mesh']
                    mesh.visual.vertex_colors = mesh_dict_rest['mesh'].\
                               visual.vertex_colors # assign rest surface color

                    # save bones
                    if 'bones' in mesh_dict.keys():
                        bone = mesh_dict['bones'][0].cpu().numpy()
                        aux_seq['bone'].append(bone)
                else:
                    mesh=mesh_dict_rest['mesh']
                aux_seq['mesh'].append(mesh)

                # save cams
                aux_seq['rtk'].append(self.model.rtk[0].cpu().numpy())
                sim3_j2c = self.model.sim3_j2c[self.model.dataid[0].long()]
                aux_seq['sim3_j2c'].append(sim3_j2c.cpu().numpy())
                
                # save image list
                impath = self.model.impath[self.model.frameid[0].long()]
                aux_seq['impath'].append(impath)

            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)

        return rendered_seq, aux_seq
    
    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.logname), comment=opts.logname)
        self.model.total_steps = 0
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
            if opts.lbs and epoch==10:
                reinit_bones(self.model, mesh_rest)
                self.init_training() # add new params to optimizer
 
            # training loop
            self.model.train()
            for i, batch in enumerate(self.dataloader):
                self.model.iters=i

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
                    try: 
                        pgrad_nan = p.grad.isnan()
                        if pgrad_nan.sum()>0: pdb.set_trace()
                    except: pass
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

                self.model.total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0: 
                    self.save_logs(log, aux_out, self.model.total_steps, epoch)

            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.\
                                         format(epoch, self.model.total_steps))
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
    def extract_mesh(model,chunk,grid_size,
                      frameid=None,
                      mesh_dict_in=None):
        opts = model.opts
        mesh_dict = {}
        if model.near_far is not None: 
            bound=np.mean(model.near_far) 
        else: bound=1.5

        if mesh_dict_in is None:
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
                    query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                                   query_xyz_chunk, frameid)
                    
                xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                dir_embedded = model.embedding_dir(query_dir_chunk) # (N, embed_dir_channels)
                xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
                out_chunks += [model.nerf_coarse(xyzdir_embedded)]
            vol_rgbo = torch.cat(out_chunks, 0)

            vol_o = vol_rgbo[...,-1].view(grid_size, grid_size, grid_size)
            vol_o = F.softplus(vol_o)
            if opts.bg: 
                percentage_th = 0.4*64**2/grid_size**2
                threshold=torch.quantile(vol_o, 1-percentage_th) # empirical value
            else: threshold = 20
            print('fraction occupied:', (vol_o > threshold).float().mean())
            vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
            vertices = (vertices - grid_size/2)/grid_size*2*bound
            mesh = trimesh.Trimesh(vertices, triangles)
       
            # mesh post-processing 
            if not opts.bg and len(mesh.vertices)>0:
                mesh = [i for i in mesh.split(only_watertight=False)]
                mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                mesh = mesh[-1]
            #    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
            
                ## assign color based on dp canonical location
                #verts = torch.Tensor(mesh.vertices).to(model.device)
                #verts_embedded = model.embedding_xyz(verts)
                #dp_verts = verts + model.nerf_dp(verts_embedded)
                #dp_verts_color = (dp_verts - model.dp_vmin)/model.dp_vmax
                ##mesh.vertices = dp_verts.cpu().numpy()
                #mesh.visual.vertex_colors = dp_verts_color.clamp(0,1).cpu().numpy()

                # assign color based on canonical location
                vis = mesh.vertices
                vis = vis - vis.min(0)[None]
                vis = vis / vis.max(0)[None]
                mesh.visual.vertex_colors[:,:3] = vis*255
                
                ## save canonical mesh
                #trimesh.Trimesh(model.dp_verts.cpu(), model.dp_faces,  
                #            vertex_colors=model.dp_vis.cpu()).export('0.obj')

                #if opts.use_dp:
                #    pdb.set_trace()
                #    dp_verts = model.dp_verts
                #    dp_verts = dp_verts - dp_verts.mean(0)[None] + \
                #     torch.Tensor(mesh.vertices.mean(0)[None]).to(model.device)
                #    dp_verts = dp_verts / dp_verts.max(0)[None] * \
                #     torch.Tensor(mesh.vertices.max(0)[None]).to(model.device)
                #    model.dp_verts.data = dp_verts

        # forward warping
        if frameid is not None and opts.queryfw:
            mesh = mesh_dict_in['mesh'].copy()
            vertices = mesh.vertices
            vertices, mesh_dict = warp_fw(opts, model, mesh_dict, 
                                           vertices, frameid)
            mesh.vertices = vertices
               
        mesh_dict['mesh'] = mesh
        return mesh_dict

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
