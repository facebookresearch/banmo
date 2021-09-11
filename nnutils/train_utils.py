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
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import cm

from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam, get_near_far, near_far_to_bound, \
                               compute_point_visibility
from ext_nnutils.train_utils import Trainer
from ext_utils.flowlib import flow_to_image
from nnutils.vis_utils import image_grid
from dataloader import frameloader
from utils.io import save_vid

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

    def define_model(self, data_info, no_ddp=False):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = mesh_net.v2s_net(img_size, opts, data_info)

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
        data_info['offset'] = np.asarray(data_offset).cumsum()
        data_info['impath'] = impath
        return data_info
    
    def init_training(self):
        opts = self.opts
        self.model.final_steps = opts.num_epochs * len(self.dataloader)

        params_nerf_coarse=[]
        params_nerf_fine=[]
        params_nerf_flowbw=[]
        params_nerf_skin=[]
        params_nerf_vis=[]
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
            elif 'nerf_skin' in name:
                params_nerf_skin.append(p)
            elif 'nerf_vis' in name:
                params_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_bone_rts' in name:
                params_nerf_bone_rts.append(p)
            elif 'embedding_time' in name or 'rest_pose_code' in name:
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
             {'params': params_nerf_skin},
             {'params': params_nerf_vis},
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
            opts.learning_rate, # params_nerf_skin
            opts.learning_rate, # params_nerf_vis
        #0.2*opts.learning_rate, # params_nerf_root_rts
        #0.1*opts.learning_rate, # params_nerf_bone_rts
          2*opts.learning_rate, # params_nerf_root_rts
            opts.learning_rate, # params_nerf_bone_rts
            opts.learning_rate, # params_embed
            opts.learning_rate, # params_bones
            opts.learning_rate, # params_ks
            opts.learning_rate, # params_nerf_dp
         10*opts.learning_rate, # params_sim3_j2c
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
            param_path = '%s/params_%s.pth'%(self.save_dir, epoch_label)
            save_dict = self.model.state_dict()
            torch.save(save_dict, param_path)

            var_path = '%s/vars_%s.npy'%(self.save_dir, epoch_label)
            np.save(var_path, self.model.latest_vars)
            return
    
    def load_network(self,model_path=None):
        states = torch.load(model_path,map_location='cpu')

        # TODO: modify states to be compatible with possibly more datasets
        len_prev_fr = states['near_far'].shape[0]
        len_prev_vid= states['sim3_j2c'].shape[0]
        self.model.near_far.data[:len_prev_fr] = states['near_far']
        self.model.sim3_j2c.data[:len_prev_vid]= states['sim3_j2c']
        self.model.embedding_time.weight.data[:len_prev_fr] = \
           states['embedding_time.weight']

        self.del_key( states, 'near_far') 
        self.del_key( states, 'sim3_j2c')
        self.del_key( states, 'embedding_time.weight')
        self.del_key( states, 'nerf_bone_rts.0.weight')

        self.model.load_state_dict(states, strict=False)
        #self.model.sim3_j2c.data[1,3:7] = torch.Tensor([0,0,1,0]).cuda()
    
        # load variables
        var_path = model_path.replace('params', 'vars').replace('.pth', '.npy')
        self.model.latest_vars = np.load(var_path,allow_pickle=True)[()]
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

            # choose a grid image or the whold video
            if num_view>0:
                idx_render = np.linspace(0,len(self.evalloader)-1,num_view, dtype=int)
            else:
                idx_render = np.asarray(range(len(self.evalloader)))

            # render
            batch = []
            rendered_seq = defaultdict(list)
            for i in idx_render:
                batch.append( self.evalloader.dataset[i] )
            batch = self.evalloader.collate_fn(batch)
            rendered = self.render_vid(self.model, batch)
            for k, v in rendered.items():
                rendered_seq[k] += [v]
                
            hbs=len(idx_render)
            rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:hbs]]
            rendered_seq['sil'] += [self.model.masks[...,None]      [:hbs]]
            rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:hbs]]
            rendered_seq['dpc'] += [self.model.dp_vis[self.model.dps.long()][:hbs]]
            rendered_seq['occ'] += [self.model.occ[...,None]      [:hbs]]
            rendered_seq['flo_coarse'][0] *= rendered_seq['sil_coarse'][0]
            if opts.flow_dp:
                rendered_seq['fdp'] += [self.model.dp_flow.permute(0,2,3,1)[:hbs]]
                rendered_seq['dcf'] += [self.model.dp_conf[...,None][:hbs]/\
                                        self.model.dp_thrd]

            # save images
            for k,v in rendered_seq.items():
                #TODO save images
                print('saving %s to gif'%k)
                rendered_seq[k] = torch.cat(rendered_seq[k],0)
                if 'flo' in k or 'fdp' in k: 
                    is_flow = True
                else: is_flow = False
                upsample_frame = min(30,len(rendered_seq[k]))
                save_vid('%s/%s'%(self.save_dir,k), 
                        rendered_seq[k].cpu().numpy(), 
                        suffix='.gif', upsample_frame=upsample_frame, 
                        is_flow=is_flow)

            # extract mesh sequences
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'sim3_j2c':[],
                       'impath':[],
                       'bone':[],}
            for idx,frameid in enumerate(idx_render):
                print('extracting frame %d'%(frameid))
                # run marching cubes
                if dynamic_mesh:
                    if not opts.queryfw:
                       mesh_dict_rest=None 
                    mesh_dict = self.extract_mesh(self.model,opts.chunk,
                                        opts.sample_grid3d, 
                                    frameid=frameid, mesh_dict_in=mesh_dict_rest)
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
                aux_seq['rtk'].append(self.model.rtk[idx].cpu().numpy())
                sim3_j2c = self.model.sim3_j2c[self.model.dataid[idx].long()]
                aux_seq['sim3_j2c'].append(sim3_j2c.cpu().numpy())
                
                # save image list
                impath = self.model.impath[self.model.frameid[idx].long()]
                aux_seq['impath'].append(impath)

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

        # disable bones before warmup epochs are finished
        if opts.lbs: 
            self.model.num_bone_used = 0
            del self.model.nerf_models['bones']

        # start training
        for epoch in range(0, opts.num_epochs):
            epoch_iter = 0
            self.model.epoch = epoch
            self.model.ep_iters = len(self.dataloader)

            # moidfy cropping factor on the fly TODO
            #self.reset_dataset_crop_factor(float(epoch)/opts.num_epochs)
            
            # evaluation
            rendered_seq, aux_seq = self.eval()                
            mesh_file = os.path.join(self.save_dir, '%s.obj'%opts.logname)
            mesh_rest = aux_seq['mesh'][0]
            mesh_rest.export(mesh_file)

            # reset object bound, only for visualization
            if epoch>int(opts.num_epochs/2):
                self.model.latest_vars['obj_bound'] = 1.2*np.abs(mesh_rest.vertices).max()

            for k,v in rendered_seq.items():
                grid_img = image_grid(rendered_seq[k],3,3)
                if k=='depth_coarse':scale=True
                else: scale=False
                self.add_image(log, k, grid_img, epoch, scale=scale)
               
            # reinit bones based on extracted surface
            if opts.lbs and (epoch==opts.lbs_all_epochs or\
                             epoch==opts.lbs_reinit_epochs):
                reinit_bones(self.model, mesh_rest, opts.num_bones)
                self.init_training() # add new params to optimizer
                self.model.num_bone_used = self.model.num_bones

            # change near-far plane after half epochs
            if epoch==int(opts.num_epochs/2):
                self.model.near_far.data = get_near_far(mesh_rest.vertices,
                                             self.model.near_far.data,
                                             self.model.latest_vars)
 
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
                grad_nerf_coarse=[]
                grad_nerf_fine=[]
                grad_nerf_flowbw=[]
                grad_nerf_skin=[]
                grad_nerf_vis=[]
                grad_nerf_root_rts=[]
                grad_nerf_bone_rts=[]
                grad_embed=[]
                grad_bones=[]
                grad_ks=[]
                grad_nerf_dp=[]
                grad_sim3_j2c=[]
                grad_dp_verts=[]
                for name,p in self.model.named_parameters():
                    try: 
                        pgrad_nan = p.grad.isnan()
                        if pgrad_nan.sum()>0: pdb.set_trace()
                    except: pass
                    if 'nerf_coarse' in name:
                        grad_nerf_coarse.append(p)
                    elif 'nerf_fine' in name:
                        grad_nerf_fine.append(p)
                    elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                        grad_nerf_flowbw.append(p)
                    elif 'nerf_skin' in name:
                        grad_nerf_skin.append(p)
                    elif 'nerf_vis' in name:
                        grad_nerf_vis.append(p)
                    elif 'nerf_root_rts' in name:
                        grad_nerf_root_rts.append(p)
                    elif 'nerf_bone_rts' in name:
                        grad_nerf_bone_rts.append(p)
                    elif 'embedding_time' in name or 'rest_pose_code' in name:
                        grad_embed.append(p)
                    elif 'bones' == name:
                        grad_bones.append(p)
                    elif 'ks' == name:
                        grad_ks.append(p)
                    elif 'nerf_dp' in name:
                        grad_nerf_dp.append(p)
                    elif 'sim3_j2c' == name:
                        grad_sim3_j2c.append(p)
                    elif 'dp_verts' == name:
                        grad_dp_verts.append(p)
                    else: continue

                aux_out['nerf_coarse_g']   = clip_grad_norm_(grad_nerf_coarse,  .1)
                aux_out['nerf_fine_g']     = clip_grad_norm_(grad_nerf_fine,    .1)
                aux_out['nerf_flowbw_g']   = clip_grad_norm_(grad_nerf_flowbw,  .1)
                aux_out['nerf_skin_g']   = clip_grad_norm_(grad_nerf_skin,  .1)
                aux_out['nerf_vis_g']   = clip_grad_norm_(grad_nerf_vis,  .1)
                aux_out['nerf_root_rts_g'] = clip_grad_norm_(grad_nerf_root_rts,.1)
                aux_out['nerf_bone_rts_g'] = clip_grad_norm_(grad_nerf_bone_rts,.1)
                aux_out['embedding_time_g']= clip_grad_norm_(grad_embed,        .1)
                aux_out['bones_g']         = clip_grad_norm_(grad_bones,        .1)
                aux_out['ks_g']            = clip_grad_norm_(grad_ks,           .1)
                aux_out['nerf_dp_g']       = clip_grad_norm_(grad_nerf_dp,      .1)
                aux_out['sim3_j2c_g']      = clip_grad_norm_(grad_sim3_j2c,     .1)
                aux_out['dp_verts_g']      = clip_grad_norm_(grad_dp_verts,     .1)

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
                self.save_network(str(epoch+1))
   
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
        rendered_first = {}
        for k,v in rendered.items():
            bs=v.shape[0]
            if v.dim()>0: rendered_first[k] = v[:bs//2] # remove loss term
        return rendered_first 

    @staticmethod
    def extract_mesh(model,chunk,grid_size,
                      #threshold = 0.01,
                      threshold = 0.,
                      frameid=None,
                      mesh_dict_in=None):
        opts = model.opts
        mesh_dict = {}
        if model.near_far is not None: 
            bound = model.latest_vars['obj_bound']
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
            #vol_o = F.softplus(vol_o)

            #TODO set density of non-observable points to small value
            if model.latest_vars['idk'].sum()>0:
                vis_chunks = []
                for i in range(0, bs_pts, chunk):
                    query_xyz_chunk = query_xyz[i:i+chunk]
                    if opts.nerf_vis:
                        # this leave no room for halucination and is not what we want
                        xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                        vis_chunk_nerf = model.nerf_vis(xyz_embedded)
                        vis_chunk = vis_chunk_nerf[...,0].sigmoid()
                    else:
                        vis_chunk = compute_point_visibility(query_xyz_chunk.cpu(),
                                         model.latest_vars, model.device)[None]
                    vis_chunks += [vis_chunk]
                vol_visi = torch.cat(vis_chunks, 0)
                vol_visi = vol_visi.view(grid_size, grid_size, grid_size)
                vol_o[vol_visi<0.5] = -1

            ## save color of sampled points 
            #cmap = cm.get_cmap('cool')
            #pts_col = cmap(vol_visi.float().view(-1).cpu())
            ##pts_col = cmap(vol_o.sigmoid().view(-1).cpu())
            #mesh = trimesh.Trimesh(query_xyz.view(-1,3).cpu(), vertex_colors=pts_col)
            #mesh.export('0.obj')
            #pdb.set_trace()

            print('fraction occupied:', (vol_o > threshold).float().mean())
            vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
            vertices = (vertices - grid_size/2)/grid_size*2*bound
            mesh = trimesh.Trimesh(vertices, triangles)

            # mesh post-processing 
            if len(mesh.vertices)>0:
                # assign color based on canonical location
                vis = mesh.vertices
                model.vis_min = vis.min(0)[None]
                vis = vis - model.vis_min
                model.vis_max = vis.max(0)[None]
                vis = vis / model.vis_max
                mesh.visual.vertex_colors[:,:3] = vis*255

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
        if 'flo' in tag or 'fdp' in tag: 
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

    @staticmethod
    def del_key(states, key):
        if key in states.keys():
            del states[key]

    def reset_dataset_crop_factor(self, percent):
        """
        percent: percentage of training epochs
        """
        #TODO schedule: 0(maxc) to 0.5 (minc)
        maxc = 10
        minc = 1.2
        for i in range(len(self.dataloader.dataset.datasets)):
            crop_factor = min(max(maxc-percent*(maxc-minc)*2, minc), maxc)
            self.dataloader.dataset.datasets[i].crop_factor = crop_factor
            self.evalloader.dataset.datasets[i].crop_factor = crop_factor

