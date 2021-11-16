import torch
import torch.nn as nn
import os
import os.path as osp
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
#import chamfer3D.dist_chamfer_3D
import torchvision
from torch.autograd import Variable


#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, local_rank=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = network.state_dict()

        if 'latest' not in save_path:
            save_dict = {k:v for k,v in save_dict.items() if 'uncertainty_predictor' not in k}
        save_dict['faces'] = self.model.module.faces.cpu()
        save_dict['epoch_nscore'] = self.epoch_nscore
        save_dict = {k:v for k,v in save_dict.items() if 'zoo_feat_net' not in k}

        if self.opts.reg3d and 'reg3d' in self.opts.name and len(self.model.module.best_list)>0:
            import kornia
            save_dict['rotvid'][1,0] = save_dict['rotvid'][1,0].matmul(kornia.rotation_matrix_to_quaternion(kornia.quaternion_to_rotation_matrix(save_dict['rotvid'][1,0]).matmul( self.model.module.best_list[0][0] )))

        torch.save(save_dict, save_path)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None,model_path=None,freeze_shape=False, finetune=False):
        save_path = model_path
        pretrained_dict = torch.load(save_path,map_location='cpu')
        
        states = pretrained_dict
        score_cams = -states['epoch_nscore']
        numvid, n_hypo = states['mean_v'].shape[:2]
        if self.opts.n_hypo<n_hypo:  # select hypothesis
            optim_cam = score_cams.argmax(-1)
            print('selecting hypothesis')
            print(optim_cam)
            mean_v = torch.zeros_like(states['mean_v'])[:,:1]
            faces = torch.zeros_like(states['faces'])[:,:1]
            tex = torch.zeros_like(states['tex'])[:,:1]
            vfeat = torch.zeros_like(states['vfeat'])[:,:1]
            rotg = torch.zeros_like(states['rotg'])[:numvid]
            rotvid = torch.zeros_like(states['rotvid'])[:numvid]
            ctl_rs = torch.zeros_like(states['ctl_rs'])[:,:1]
            ctl_ts = torch.zeros_like(states['ctl_ts'])[:,:1]
            joint_ts = torch.zeros_like(states['joint_ts'])[:,:1]
            log_ctl= torch.zeros_like(states['log_ctl'])[:,:1]
            for i in range(numvid):
                mean_v[i] = states['mean_v'][i,optim_cam[i]]
                faces[i] = states['faces'][i,optim_cam[i]]
                tex[i] = states['tex'][i,optim_cam[i]]
                vfeat[i] = states['vfeat'][i,optim_cam[i]]
                rotg[i] = states['rotg'].view(numvid,n_hypo,-1,4)[i,optim_cam[i]]
                rotvid[i] = states['rotvid'].view(numvid,n_hypo,-1,4)[i,optim_cam[i]]
                ctl_rs [i]= states['ctl_rs'] [i,optim_cam[i]] 
                ctl_ts [i]= states['ctl_ts'] [i,optim_cam[i]] 
                joint_ts [i]= states['joint_ts'] [i,optim_cam[i]] 
                log_ctl[i]= states['log_ctl'][i,optim_cam[i]] 
                # nerf-tex
                nerfidx = i*n_hypo+optim_cam[i]
                for k,v in states.items():
                    if 'nerf_tex.%d'%nerfidx in k:
                        states[k.replace('nerf_tex.%d'%nerfidx, 'nerf_tex.%d'%i)] = states[k]
                    if 'nerf_feat.%d'%nerfidx in k:
                        states[k.replace('nerf_feat.%d'%nerfidx, 'nerf_feat.%d'%i)] = states[k]
                    if 'nerf_shape.%d'%nerfidx in k:
                        states[k.replace('nerf_shape.%d'%nerfidx, 'nerf_shape.%d'%i)] = states[k]
                    if 'nerf_mshape.%d'%nerfidx in k:
                        states[k.replace('nerf_mshape.%d'%nerfidx, 'nerf_mshape.%d'%i)] = states[k]
                    if 'encoder.%d'%(optim_cam[i]) in k:
                        states[k.replace('encoder.%d.'%(optim_cam[i]), 'encoder.0.')] = states[k]
                    if 'code_predictor.%d'%(optim_cam[i]) in k:
                        states[k.replace('code_predictor.%d.'%(optim_cam[i]), 'code_predictor.0.')] = states[k]




            states['mean_v'] = mean_v
            states['faces'] = faces
            states['tex'] = tex
            states['vfeat'] = vfeat
            states['rotg'] = rotg
            states['rotvid'] = rotvid
            states['ctl_rs'] = ctl_rs 
            states['ctl_ts'] = ctl_ts 
            states['joint_ts'] = joint_ts 
            states['log_ctl'] =log_ctl

        if finetune:
            pretrained_dict = states
       
        # remesh     
        if (not self.opts.symmetric) and (int(self.opts.n_faces)!=states['faces'].shape[2]):
            # do multiple meshes
            numvid = states['mean_v'].shape[0]
            mulv = []
            mulf = []
            vmax = 0
            for i in range(numvid):
                sr.Mesh(states['mean_v'][i:i+1,0], states['faces'][i:i+1,0]).save_obj('tmp/input-%d.obj'%(self.opts.local_rank))
                import subprocess
                print(subprocess.check_output(['./Manifold/build/manifold', 'tmp/input-%d.obj'%(self.opts.local_rank), 'tmp/output-%d.obj'%(self.opts.local_rank), '10000']))
                print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/output-%d.obj'%(self.opts.local_rank), '-o', 'tmp/simple-%d.obj'%(self.opts.local_rank), '-m', '-f', self.opts.n_faces]))
                # load remeshed 
                loadmesh = sr.Mesh.from_obj('tmp/simple-%d.obj'%(self.opts.local_rank))
                mulv.append(loadmesh.vertices)
                mulf.append(loadmesh.faces)
                vmax = max(vmax, loadmesh.vertices.shape[1])
            
            for i in range(len(mulv)):
                padtensor = torch.zeros_like(mulv[0])[:,:vmax-mulv[i].shape[1]]
                mulv[i] = torch.cat([mulv[i], padtensor],1)
            
            # use the first one 
            mulv = [mulv[0]] * self.model.numvid
            mulf = [mulf[0]] * self.model.numvid
 
            self.model.num_verts = mulv[0].shape[-2]
            self.model.num_faces = mulf[0].shape[-2]
            self.model.mean_v.data = torch.cat(mulv,0).view(self.model.numvid, 1, self.model.num_verts,3)
            self.model.faces.data  = torch.cat(mulf,0).view(self.model.numvid, 1,self.model.num_faces,3)
        else:
            self.model.num_verts = states['mean_v'].shape[-2]
            self.model.num_faces = states['faces'].shape[-2]
            self.model.mean_v.data = states['mean_v'] 
            self.model.faces.data  = states['faces']
        self.model.tex.data = torch.zeros_like(self.model.mean_v.data)
        self.model.vfeat.data = torch.zeros(self.model.numvid, self.opts.n_hypo, self.model.num_verts, self.model.nfeat)
        del states['mean_v']
        del states['tex']
        try: del states['vfeat']
        except: pass
        del states['faces']
            
        # change number of bones
        self.model.reinit_bones = False
        if states['code_predictor.0.depth_predictor.pred_layer.bias'].shape[0] != self.opts.n_mesh:  # from rigid body to deformable
            self.model.reinit_bones = True
            rotg = torch.zeros(states['rotg'].shape[0],self.opts.n_mesh,4)
            rotg[:,:1] = states['rotg'][:,:1]
            states['rotg'] = rotg

            nfeat = states['code_predictor.0.quat_predictor.pred_layer.weight'].shape[-1]
            quat_weights = torch.cat( [states['code_predictor.0.quat_predictor.pred_layer.weight'].view(-1,4,nfeat)[:1], self.model.code_predictor[0].quat_predictor.pred_layer.weight.view(self.opts.n_mesh,4,-1)[1:]],0).view(self.opts.n_mesh*4,-1)
            quat_bias =    torch.cat( [states['code_predictor.0.quat_predictor.pred_layer.bias'].view(-1,4)[:1],         self.model.code_predictor[0].quat_predictor.pred_layer.bias.view(self.opts.n_mesh,-1)[1:]],0).view(-1)
            states['code_predictor.0.quat_predictor.pred_layer.weight'] = quat_weights
            states['code_predictor.0.quat_predictor.pred_layer.bias'] = quat_bias
            
            tmp_weights = torch.cat( [states['code_predictor.0.trans_predictor.pred_layer.weight'].view(-1,2,nfeat)[:1], self.model.code_predictor[0].trans_predictor.pred_layer.weight.view(self.opts.n_mesh,2,-1)[1:]],0).view(self.opts.n_mesh*2,-1)
            tmp_bias =    torch.cat( [states['code_predictor.0.trans_predictor.pred_layer.bias'].view(-1,2)[:1],         self.model.code_predictor[0].trans_predictor.pred_layer.bias.view(self.opts.n_mesh,-1)[1:]],0).view(-1)
            states['code_predictor.0.trans_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.0.trans_predictor.pred_layer.bias'] =   tmp_bias
            
            tmp_weights = torch.cat( [states['code_predictor.0.depth_predictor.pred_layer.weight'].view(-1,1,nfeat)[:1], self.model.code_predictor[0].depth_predictor.pred_layer.weight.view(self.opts.n_mesh,1,-1)[1:]],0).view(self.opts.n_mesh*1,-1)
            tmp_bias =    torch.cat( [states['code_predictor.0.depth_predictor.pred_layer.bias'].view(-1,1)[:1],         self.model.code_predictor[0].depth_predictor.pred_layer.bias.view(self.opts.n_mesh,-1)[1:]],0).view(-1)
            states['code_predictor.0.depth_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.0.depth_predictor.pred_layer.bias'] =   tmp_bias

            ## initialize skin based on mean shape 
            #np.random.seed(18)
            del states['ctl_rs']
            del states['log_ctl']
            del states['ctl_ts']
            del states['joint_ts']
            del states['rotvid']

        # reset rotgf
        self.model.rotgf = states['rotg'] / torch.norm(states['rotg'], 2,-1)[:,:,None]
        self.model.rotgf = self.model.rotgf.cuda()

        if 'texhr' in states.keys():
            del states['texhr']
            
        # delete unused vars
        try:
            del states['shapePred.pred_layer.weight']
            del states['shapePred.pred_layer.bias']
            del states['shape_basis']
            del states['code_predictor_class.quat_predictor.pred_layer.weight']
            del states['code_predictor_class.quat_predictor.pred_layer.bias']
            del states['code_predictor_class.scale_predictor.pred_layer.weight']
            del states['code_predictor_class.scale_predictor.pred_layer.bias']
            del states['code_predictor_class.trans_predictor.pred_layer.weight']
            del states['code_predictor_class.trans_predictor.pred_layer.bias']
            del states['code_predictor_class.depth_predictor.pred_layer.weight']
            del states['code_predictor_class.depth_predictor.pred_layer.bias']
        except:pass
    
        if numvid < self.model.numvid:
            self.model.tex.data = self.model.tex.data[:1].repeat(self.model.numvid, 1,1,1)
            self.model.mean_v.data = self.model.mean_v.data[:1].repeat(self.model.numvid, 1,1,1)
            self.model.faces.data = self.model.faces.data[:1].repeat(self.model.numvid, 1,1,1)
            states['rotg'] = states['rotg'][:1].repeat(self.model.numvid, 1,1)
            states['joint_ts'] = states['joint_ts'][:1].repeat(self.model.numvid, 1,1,1)
            states['ctl_ts'] = states['ctl_ts'][:1].repeat(self.model.numvid, 1,1,1)
            states['ctl_rs'] = states['ctl_rs'][:1].repeat(self.model.numvid, 1,1,1)
            states['log_ctl'] = states['log_ctl'][:1].repeat(self.model.numvid,1, 1,1)
            del states['rotvid']
            del states['shape_code_fix']

        network.load_state_dict(pretrained_dict,strict=False)
        if self.opts.self_augment:
            self.model.encoder_copy.load_state_dict(self.model.encoder.state_dict())
            self.model.code_predictor_copy.load_state_dict(self.model.code_predictor.state_dict())

        return
    

    def define_model(self):
        opts = self.opts
        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
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

    def define_criterion_ddp(self):
        self.model.module.projection_loss = loss_utils.kp_l2_loss
        self.model.module.mask_loss_fn = torch.nn.MSELoss()
        self.model.module.entropy_loss = loss_utils.entropy_loss
        self.model.module.deform_reg_fn = loss_utils.deform_l2reg
        self.model.module.camera_loss = loss_utils.camera_loss
        self.model.module.triangle_loss_fn_sr = [loss_utils.LaplacianLoss(self.model.module.mean_v[i,0].cpu(), self.model.module.faces[i,0].cpu()).cuda() for i in range(self.model.module.numvid)]
        self.model.module.arap_loss_fn = [loss_utils.ARAPLoss(self.model.module.mean_v[i,0].cpu(), self.model.module.faces[i,0].cpu()).cuda()             for i in range(self.model.module.numvid)]
        self.model.module.flatten_loss = [loss_utils.FlattenLoss(self.model.module.faces[i,0].cpu()).cuda()                           for i in range(self.model.module.numvid)]
        self.model.module.uniform_loss = loss_utils.local_nonuniform_penalty
        from PerceptualSimilarity.models import dist_model
        self.model.module.ptex_loss = dist_model.DistModel()
        self.model.module.ptex_loss.initialize(model='net', net='alex', use_gpu=False)
        self.model.module.ptex_loss.cuda(self.opts.local_rank)
        self.model.module.ssim1 = loss_utils.SSIM()
        self.model.module.ssim2 = loss_utils.SSIM()
        self.model.module.chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    


    def save(self, epoch_prefix):
        '''Saves the model.'''
        if self.opts.local_rank==0:
            self.save_network(self.model.module, 'pred', epoch_prefix, local_rank=self.opts.local_rank)
        return

    def init_training(self):
        opts = self.opts
        new_params=[]
        nerf_params=[]
        nerf_shape_params=[]
        nerf_time_params=[]
        for name,p in self.model.module.named_parameters():
            if name == 'mean_v': print('found mean v'); continue
            if name == 'rotg': print('found rotg'); continue
            if name == 'rotvid': print('found rotvid'); continue
            if name == 'rots': print('found rots'); continue
            if name == 'focal': print('found fl');continue
            if name == 'pps': print('found pps');continue
            if name == 'tex': print('found tex');continue
            if name == 'texhr': print('found texhr');continue
            if name == 'vfeat': print('found vfeat');continue
            if name == 'body_score': print('found body_score');continue
            if name == 'skin': print('found skin'); continue
            if name == 'score_cams': print('found score cams'); continue
            if name == 'score_frames': print('found score frames'); continue
            if name == 'rest_rs': print('found rest rotation'); continue
            if name == 'ctl_rs': print('found ctl rotation'); continue
            if name == 'ctl_ts': print('found ctl points'); continue
            if name == 'joint_ts': print('found joint points'); continue
            if name == 'light_params': print('found light_params points'); continue
            if name == 'transg': print('found global translation'); continue
            if name == 'log_ctl': print('found log ctl'); continue
            if name == 'shape_basis': print('found shape basis'); continue
            if name == 'shape_code_fix': print('found shape basis'); continue
            if 'nerf_tex' in name or 'nerf_feat' in name or 'nerf_coarse' in name or 'nerf_fine' in name: 
                print('found %s'%name); nerf_params.append(p); continue
            if 'nerf_shape' in name or 'nerf_mshape' in name: 
                print('found %s'%name); nerf_shape_params.append(p); continue
            if 'nerf_time' in name:
                print('found %s'%name); nerf_time_params.append(p); continue
            new_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': new_params},
             {'params': nerf_params, 'lr': 10*opts.learning_rate},
             {'params': nerf_shape_params, 'lr': 10*opts.learning_rate},
             {'params': nerf_time_params, 'lr': 10*opts.learning_rate},
             {'params': self.model.module.mean_v, 'lr':50*opts.learning_rate},
             {'params': self.model.module.rotg, 'lr':50*opts.learning_rate},
             {'params': self.model.module.rotvid, 'lr':50*opts.learning_rate},
             {'params': self.model.module.pps, 'lr':50*opts.learning_rate},
             {'params': self.model.module.tex, 'lr':50*opts.learning_rate},
             {'params': self.model.module.texhr, 'lr':50*opts.learning_rate},
             {'params': self.model.module.vfeat, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_rs, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.joint_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.log_ctl, 'lr':50*opts.learning_rate},
             {'params': self.model.module.light_params, 'lr':50*opts.learning_rate},
             {'params': self.model.module.shape_basis, 'lr':50*opts.learning_rate},
             {'params': self.model.module.shape_code_fix, 'lr':50*opts.learning_rate},
            ],
            lr=opts.learning_rate,betas=(opts.beta1, 0.999),weight_decay=1e-4)

        lr_meanv = 50*opts.learning_rate
        rotglr = 50*opts.learning_rate
        cnnlr = opts.learning_rate
        nerflr= 10*opts.learning_rate
        nerf_shape_lr= 10*opts.learning_rate
        nerf_time_lr= 10*opts.learning_rate
        conslr = 50*opts.learning_rate
        lr_meanv = 50*opts.learning_rate

        pct_start = 0.01
        div_factor = 25
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        [cnnlr,
        nerflr, # nerf-params
        nerf_shape_lr, # nerf shape params
        nerf_time_lr, # nerf shape params
        lr_meanv,
        rotglr, # rotg
        0., # rotvid
        50*opts.learning_rate, # pps
        conslr, # tex
        conslr, # texhr
        conslr, # vfeat
        conslr, # ctl rs 
        conslr, # ctl ts 
        conslr, # joint ts 
        conslr, # log ctl
        conslr, # light_params
        conslr, # shape basis
        conslr, # shape code fix
        ],
        200*len(self.dataloader), pct_start=pct_start, cycle_momentum=False, anneal_strategy='linear',final_div_factor=1./25, div_factor = div_factor)

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

            #self.model.module.epoch = epoch
            ## reinit bones
            #if (self.model.module.reinit_bones or not opts.finetune) and (self.opts.local_rank==0 and epoch==0 and self.opts.n_mesh>1):
            #    for idx in range(self.model.module.mean_v.shape[0]):
            #        for jdx in range(self.model.module.mean_v.shape[1]):
            #            mean_shape = self.model.module.mean_v[idx,jdx].clone()
            #            if self.opts.catemodel:
            #                pred_v_symm = mean_shape[None].clone()
            #                shape_delta = run_network(
            #                self.model.module.nerf_mshape[idx*self.opts.n_hypo+jdx],
            #                pred_v_symm,
            #                None,
            #                131072,
            #                self.model.module.encode_position_fn_shape,
            #                None,
            #                #code=None,
            #                code=torch.zeros(1,self.model.module.codedim).cuda(),
            #                )[:,:,:3]
            #                mean_shape = mean_shape + shape_delta[0]

            #            cluster_ids_x, cluster_centers = kmeans(
            #            X=mean_shape, num_clusters=self.opts.n_mesh-1, distance='euclidean', device=torch.device('cuda:%d'%(opts.local_rank)))
            #            self.model.module.ctl_ts.data[idx,jdx] = cluster_centers.cuda()
            #            self.model.module.joint_ts.data[idx,jdx] = cluster_centers.cuda()
            #    self.model.module.ctl_rs.data[:,:,:] = torch.Tensor([0,0,0,1]).cuda()
            #    self.model.module.log_ctl.data[:]= 0
            #dist.barrier()
            #dist.broadcast(self.model.module.joint_ts, 0)
            #dist.broadcast(self.model.module.ctl_ts, 0)
            #dist.broadcast(self.model.module.ctl_rs, 0)
            #dist.broadcast(self.model.module.log_ctl, 0)
            #print('new bone locations')
            
            # modify dataset
            if False:
                capdata = (epoch+1)*2+1
                if capdata<7: capdata=7
                print('capped to %d frames'%capdata)
                self.dataloader,length = self.data_module.data_loader(opts,capdata=capdata)
                if epoch>20 and (capdata > length+5):exit()
            
            #self.model.module.ep_iters = len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                print(i)
                #self.model.module.iters=i
                #self.model.module.total_steps = total_steps
                input_batch = self.set_input(batch)

                if self.opts.debug:
                    torch.cuda.synchronize()
                    start_time = time.time()

                #self.optimizer.zero_grad()
                total_loss,aux_output = self.model(input_batch)
                #total_loss.mean().backward()
                
                if self.opts.debug:
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

                ## gradient clipping
                #cam_grad = []
                #nerf_mshape_grad = []
                #nerf_tex_grad = []
                #nerf_time_grad = []
                #for name,p in self.model.module.named_parameters():
                #    #print(name)
                #    if 'mean_v' == name and p.grad is not None:
                #        torch.nn.utils.clip_grad_norm_(p, 1.)
                #        self.grad_meanv_norm = p.grad.view(-1).norm(2,-1)
                #    elif p.grad is not None and ('nerf_mshape' in name):
                #        nerf_mshape_grad.append(p)
                #    elif p.grad is not None and ('nerf_tex' in name):
                #        nerf_tex_grad.append(p)
                #    elif p.grad is not None and ('nerf_time' in name):
                #        nerf_time_grad.append(p)
                #    elif p.grad is not None and ('code_predictor' in name or 'encoder' in name):
                #        cam_grad.append(p)
                #    if (not p.grad is None) and (torch.isnan(p.grad).sum()>0):
                #        self.optimizer.zero_grad()
                #self.grad_cam_norm = torch.nn.utils.clip_grad_norm_(cam_grad, 10.)
                #self.grad_nerf_tex_norm = torch.nn.utils.clip_grad_norm_(nerf_tex_grad, 1.)
                #self.grad_nerf_mshape_norm = torch.nn.utils.clip_grad_norm_(nerf_mshape_grad, 1)
                #self.grad_nerf_time_norm = torch.nn.utils.clip_grad_norm_(nerf_time_grad, 0.1)

                #if opts.local_rank==0 and torch.isnan(self.model.module.total_loss):
                #    pdb.set_trace()
                #self.optimizer.step()
                #self.scheduler.step()

                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0:
                    if i==0:
                        #gu = np.asarray(self.model.module.flow[0,0,:,:].detach().cpu()) 
                        #gv = np.asarray(self.model.module.flow[0,1,:,:].detach().cpu())
                        #
                        #flow = np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis]),-1)
                        #warped = warp_flow(np.asarray(255*self.model.module.imgs[opts.batch_size].permute(1,2,0).detach().cpu()).astype(np.uint8), flow/2*opts.img_size)
                        #self.add_image(log,'train/warped_flow', warped[None],epoch, scale=False)
                        #try:
                        #    mask = aux_output['vis_mask'][:,optim_cam]
                        #    mask = np.asarray(mask[0].float().cpu())
                        #except: mask = np.zeros((opts.img_size, opts.img_size))
                        #
                        #gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                        #self.add_image(log,'train/flowobs', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                        #
                        #try:
                        #    gu = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,opts.n_hypo,opts.img_size,opts.img_size,2)[0,optim_cam,:,:,0].detach().cpu())
                        #    gv = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,opts.n_hypo,opts.img_size,opts.img_size,2)[0,optim_cam,:,:,1].detach().cpu())
                        #    gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                        #    self.add_image(log,'train/flowrd', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                        #    error = np.asarray(aux_output['flow_rd_map'][:1,optim_cam].detach().cpu()); error=error*mask
                        #    self.add_image(log,'train/flow_error', opts.img_size*error,epoch)
                        #    self.add_image(log,'train/mask', 255*np.asarray(aux_output['mask_pred'][optim_cam:optim_cam+1].detach().cpu()),epoch)
                        #except: pass
                        #self.add_image(log,'train/exp_rd',  aux_output['exp_rd'].view(2*opts.batch_size,opts.n_hypo,opts.img_size, opts.img_size)[:1,optim_cam].log(),epoch)
                        #self.add_image(log,'train/exp_obs', aux_output['exp_obs'][0].log(),epoch)
                        self.add_image(log,'train/maskgt', 255*np.asarray(self.model.module.masks[:1].detach().cpu()),epoch)

                        img1_j = np.asarray(255*self.model.module.imgs[:1].permute(0,2,3,1).detach().cpu()).astype(np.uint8)
                        self.add_image(log,'train/img1', img1_j,epoch      ,scale=False)
                        self.add_image(log,'train/img2', np.asarray(255*self.model.module.imgs[opts.batch_size:opts.batch_size+1].permute(0,2,3,1).detach().cpu()).astype(np.uint8),epoch, scale=False)
                        if opts.n_mesh>1 and 'part_render' in aux_output.keys():
                            self.add_image(log,'train/part', np.asarray(255*aux_output['part_render'][:1].detach().cpu().permute(0,2,3,1), dtype=np.uint8),epoch)
                        if opts.n_mesh>1 and 'cost_rd' in aux_output.keys():
                            cost = aux_output['cost_rd'][:1].clone()
                            cost = ( cost-cost.min() ) / (cost.max() - cost.min()) * 255
                            self.add_image(log,'train/cost', np.asarray(cost.detach().cpu().permute(0,2,3,1), dtype=np.uint8),epoch)
                        if  hasattr(self.model.module, 'depth_render'):
                            depth_render = self.model.module.depth_render[:1].detach().cpu()
                            depth_render = (depth_render-depth_render.min()) / (depth_render.max()-depth_render.min())
                            self.add_image(log,'train/depth', np.asarray(depth_render),epoch)
                        if  hasattr(self.model.module, 'depth'):
                            depth = self.model.module.depth[:1]
                            depth = (depth-depth.min()) / (depth.max()-depth.min())
                            self.add_image(log,'train/depth_obs', np.asarray(depth.detach().cpu()),epoch)

                        if 'texture_render' in aux_output.keys():
                            texture_j = np.asarray(aux_output['texture_render'][optim_cam:optim_cam+1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_mesh>1:
                                for k in range(aux_output['ctl_proj'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['ctl_proj'][optim_cam].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            self.add_image(log,'train/texture', texture_j,epoch,scale=False)
                            # joints
                            texture_j = np.asarray(aux_output['texture_render'][optim_cam:optim_cam+1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_mesh>1:
                                for k in range(aux_output['joint_proj'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['joint_proj'][optim_cam].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            self.add_image(log,'train/texture_j', texture_j,epoch,scale=False)
                        if 'kp' in aux_output.keys() and 'texture_render' in aux_output.keys():
                            texture_j = np.asarray(self.model.module.imgs[1:2].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            texture_i = np.asarray(aux_output['texture_render'][1:2].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            for k in range(aux_output['kp'].shape[1]):
                               texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(np.asarray((128+128*aux_output['kp'][1].detach()).int().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            self.add_image(log,'train/kp', texture_j[:1],epoch,scale=False)
                            
                        if 'texture_render_pred' in aux_output.keys():
                            texture_j = np.asarray(aux_output['texture_render_pred'][optim_cam:optim_cam+1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_mesh>1:
                                for k in range(aux_output['ctl_proj_pred'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['ctl_proj_pred'][optim_cam].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            self.add_image(log,'train/texture_pred', texture_j,epoch,scale=False)
                        if 'texture_render_hr' in aux_output.keys():
                            self.add_image(log,'train/texture_hr', np.asarray(aux_output['texture_render_hr'][:1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8),epoch,scale=False)
                    if 'total_loss' in aux_output.keys():
                        log.add_scalar('train/total_loss',  aux_output['total_loss'].mean()  , total_steps)
                    if 'mask_loss' in aux_output.keys():
                        log.add_scalar('train/mask_loss' ,  aux_output['mask_loss'].mean()   , total_steps)
                    if 'flow_rd_loss' in aux_output.keys():
                        log.add_scalar('train/flow_rd_loss',aux_output['flow_rd_loss'].mean(), total_steps)
                    if 'skin_ent_loss' in aux_output.keys():
                        log.add_scalar('train/skin_ent_loss',aux_output['skin_ent_loss'].mean(), total_steps)
                    if 'arap_loss' in aux_output.keys():
                        log.add_scalar('train/arap_loss',aux_output['arap_loss'].mean(), total_steps)
                    if 'rotg_sm_sub_loss' in aux_output.keys():
                        log.add_scalar('train/rotg_sm_sub_loss',aux_output['rotg_sm_sub_loss'].mean(), total_steps)
                    if 'velo_loss' in aux_output.keys():
                        log.add_scalar('train/velo_loss',aux_output['velo_loss'].mean(), total_steps)
                    if 'match_loss' in aux_output.keys():
                        log.add_scalar('train/match_loss',aux_output['match_loss'].mean(), total_steps)
                    if 'csm_loss' in aux_output.keys():
                        log.add_scalar('train/csm_loss',aux_output['csm_loss'].mean(), total_steps)
                    if 'flow_s_loss' in aux_output.keys():
                        log.add_scalar('train/flow_s_loss',aux_output['flow_s_loss'].mean(), total_steps)
                    if 'feat_loss' in aux_output.keys():
                        log.add_scalar('train/feat_loss',aux_output['feat_loss'].mean(), total_steps)
                    if 'rank_loss' in aux_output.keys():
                        log.add_scalar('train/rank_loss',aux_output['rank_loss'].mean(), total_steps)
                    if 'kp_loss' in aux_output.keys():
                        log.add_scalar('train/kp_loss',aux_output['kp_loss'].mean(), total_steps)
                    if 'texture_loss' in aux_output.keys():
                        log.add_scalar('train/texture_loss',aux_output['texture_loss'].mean(), total_steps)
                    if 'depth_loss' in aux_output.keys():
                        log.add_scalar('train/depth_loss',aux_output['depth_loss'].mean(), total_steps)
                    if 'rot_pc_loss' in aux_output.keys():
                        log.add_scalar('train/rot_pc_loss',aux_output['rot_pc_loss'].mean(), total_steps)
                    if opts.n_hypo > 1:
                        for ihp in range(opts.n_hypo):
                            log.add_scalar('train/mask_hypo_%d'%ihp,aux_output['mask_hypo_%d'%ihp].mean()   , total_steps)
                            log.add_scalar('train/flow_hypo_%d'%ihp,aux_output['flow_hypo_%d'%ihp].mean(), total_steps)
                            log.add_scalar('train/tex_hypo_%d'%ihp,aux_output['tex_hypo_%d'%ihp].mean(), total_steps)
                    
                    if 'triangle_loss' in aux_output.keys():
                        log.add_scalar('train/triangle_loss',aux_output['triangle_loss'], total_steps)
                    if 'lmotion_loss' in aux_output.keys():
                        log.add_scalar('train/lmotion_loss', aux_output['lmotion_loss'], total_steps)
                    if 'orth_loss' in aux_output.keys():
                        log.add_scalar('train/orth_loss', aux_output['orth_loss'], total_steps)
                    if 'nerf_tex_loss' in aux_output.keys():
                        log.add_scalar('train/nerf_tex_loss', aux_output['nerf_tex_loss'], total_steps)
                    if 'nerf_shape_loss' in aux_output.keys():
                        log.add_scalar('train/nerf_shape_loss', aux_output['nerf_shape_loss'], total_steps)
                    if 'l1_deform_loss' in aux_output.keys():
                        log.add_scalar('train/l1_deform_loss', aux_output['l1_deform_loss'], total_steps)
                    if 'geo_aug_loss' in aux_output.keys():
                        log.add_scalar('train/geo_aug_loss', aux_output['geo_aug_loss'], total_steps)
                    if 'pen_loss' in aux_output.keys():
                        log.add_scalar('train/pen_loss', aux_output['pen_loss'], total_steps)
                    if hasattr(self, 'grad_meanv_norm'): log.add_scalar('train/grad_meanv_norm',self.grad_meanv_norm, total_steps)
                    if hasattr(self, 'grad_cam_norm'):log.add_scalar('train/grad_cam_norm',self.grad_cam_norm, total_steps)
                    if hasattr(self, 'grad_nerf_mshape_norm'):log.add_scalar('train/grad_nerf_mshape_norm',self.grad_nerf_mshape_norm, total_steps)
                    if hasattr(self, 'grad_nerf_tex_norm'):log.add_scalar('train/grad_nerf_tex_norm',self.grad_nerf_tex_norm, total_steps)
                    if hasattr(self, 'grad_nerf_time_norm'):log.add_scalar('train/grad_nerf_time_norm',self.grad_nerf_time_norm, total_steps)
                        
                    if hasattr(self.model.module, 'sampled_img_obs_vis'):
                        if i%10==0:
                            self.add_image(log,'train/sampled_img_obs_vis', np.asarray(255*self.model.module.sampled_img_obs_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            self.add_image(log,'train/sampled_img_rdc_vis', np.asarray(255*self.model.module.sampled_img_rdc_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            self.add_image(log,'train/sampled_img_rdf_vis', np.asarray(255*self.model.module.sampled_img_rdf_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                        log.add_scalar('train/coarse_loss',self.model.module.coarse_loss, total_steps)
                        log.add_scalar('train/sil_coarse_loss',self.model.module.sil_coarse_loss, total_steps)
                        log.add_scalar('train/fine_loss',self.model.module.fine_loss, total_steps)

            #if (epoch+1) % opts.save_epoch_freq == 0:
            #    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
            #    self.save('latest')
            #    self.save(epoch+1)
