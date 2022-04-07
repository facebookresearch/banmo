# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
bash scripts/render_nvs.sh
"""
from absl import flags, app
import sys
sys.path.insert(0,'')
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio
from collections import defaultdict
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')

from utils.io import save_vid, str_to_frame, save_bones, load_root, load_sils
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, \
                                raycast, sample_xy, K2inv, get_near_far, \
                                chunk_rays
from nnutils.rendering import render_rays
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS

# script specific ones
flags.DEFINE_integer('maxframe', 1, 'maximum number frame to render')
flags.DEFINE_integer('vidid', 0, 'video id that determines the env code')
flags.DEFINE_integer('bullet_time', -1, 'frame id in a video to show bullet time')
flags.DEFINE_float('scale', 0.1,
        'scale applied to the rendered image (wrt focal length)')
flags.DEFINE_string('rootdir', 'tmp/traj/','root body directory')
flags.DEFINE_string('nvs_outpath', 'tmp/nvs-','output prefix')

def construct_rays_nvs(img_size, rtks, near_far, rndmask, device):
    """
    rndmask: controls which pixel to render
    """
    bs = rtks.shape[0]
    rtks = torch.Tensor(rtks).to(device)
    rndmask = torch.Tensor(rndmask).to(device).view(-1)>0

    _, xys = sample_xy(img_size, bs, 0, device, return_all=True)
    xys=xys[:,rndmask]
    Rmat = rtks[:,:3,:3]
    Tmat = rtks[:,:3,3]
    Kinv = K2inv(rtks[:,3])
    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    return rays
                
def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    model = trainer.model
    model.eval()

    nerf_models = model.nerf_models
    embeddings = model.embeddings

    # bs, 4,4 (R|T)
    #         (f|p)
    nframe=120
    img_size = int(512 * opts.scale)
    fl = img_size
    pp = img_size/2
    rtks = np.zeros((nframe,4,4))
    rot1 = cv2.Rodrigues(np.asarray([0,np.pi/2,0]))[0]
    rot2 = cv2.Rodrigues(np.asarray([np.pi,0,0]))[0]
    rtks[:,:3,:3] = np.dot(rot1, rot2)[None]
    rtks[:,2,3] = 0.2
    rtks[:,3] = np.asarray([fl,fl,pp,pp])[None]
    sample_idx = np.asarray(range(nframe)).astype(int)

    # determine render image scale
    bs = len(rtks)
    print("render size: %d"%img_size)
    model.img_size = img_size
    opts.render_size = img_size

    vars_np = {}
    vars_np['rtk'] = rtks
    vars_np['idk'] = np.ones(bs)
    near_far = torch.zeros(bs,2).to(model.device)
    near_far = get_near_far(near_far,
                            vars_np,
                            pts=model.latest_vars['mesh_rest'].vertices)
    depth_near = near_far[0,0].cpu().numpy()
    depth_far = near_far[0,1].cpu().numpy()

    vidid = torch.Tensor([opts.vidid]).to(model.device).long()
    source_l = model.data_offset[opts.vidid+1] - model.data_offset[opts.vidid] -1
    embedid = torch.Tensor(sample_idx).to(model.device).long() + \
              model.data_offset[opts.vidid]
    print(embedid)
    rgbs = []
    sils = []
    dphs = []
    viss = []
    for i in range(bs):
        model_path = '%s/%s'% (opts.model_path.rsplit('/',1)[0], 'params_%d.pth'%(i))
        trainer.load_network(model_path, is_eval=True)# load latest
        rndmask = np.ones((img_size, img_size))>0
        rays = construct_rays_nvs(model.img_size, rtks[i:i+1], 
                                       near_far[i:i+1], rndmask, model.device)
        # add env code
        rays['env_code'] = model.env_code(embedid[i:i+1])[:,None]
        rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)
        
        ## add bones
        #time_embedded = model.pose_code(embedid[i:i+1])[:,None]
        #rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        #if opts.lbs and model.num_bone_used>0:
        #    bone_rts = model.nerf_body_rts(embedid[i:i+1])
        #    rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
        #    model.update_delta_rts(rays)

        with torch.no_grad():
            # render images only
            results=defaultdict(list)
            bs_rays = rays['bs'] * rays['nsample'] #
            for j in range(0, bs_rays, opts.chunk):
                rays_chunk = chunk_rays(rays,j,opts.chunk)
                rendered_chunks = render_rays(nerf_models,
                            embeddings,
                            rays_chunk,
                            N_samples = opts.ndepth,
                            perturb=0,
                            noise_std=0,
                            chunk=opts.chunk, # chunk size is effective in val mode
                            use_fine=True,
                            img_size=model.img_size,
                            obj_bound = model.latest_vars['obj_bound'],
                            render_vis=True,
                            opts=opts,
                            )
                for k, v in rendered_chunks.items():
                    results[k] += [v]
           
        for k, v in results.items():
            v = torch.cat(v, 0)
            v = v.view(rays['nsample'], -1)
            results[k] = v
        rgb = results['img_coarse'].cpu().numpy()
        dph = results['depth_rnd'] [...,0].cpu().numpy()
        sil = results['sil_coarse'][...,0].cpu().numpy()
        vis = results['vis_pred']  [...,0].cpu().numpy()
        #sil[sil<0.5] = 0
        #rgb[sil<0.5] = 1

        rgbtmp = np.ones((img_size, img_size, 3))
        dphtmp = np.ones((img_size, img_size))
        siltmp = np.ones((img_size, img_size))
        vistmp = np.ones((img_size, img_size))
        rgbtmp[rndmask>0] = rgb
        dphtmp[rndmask>0] = dph
        siltmp[rndmask>0] = sil
        vistmp[rndmask>0] = vis

        rgb = rgbtmp
        sil = siltmp
        vis = vistmp
        dph = dphtmp
        dph = (dph - depth_near) / (depth_far - depth_near)*2
        dph = np.clip(dph,0,1)
        dph = cmap(dph)
        rgb = rgb * sil[...,None]
        dph = dph * sil[...,None]
    
        rgbs.append(rgb)
        sils.append(sil*255)
        viss.append(vis*255)
        dphs.append(dph*255)
        cv2.imwrite('%s-rgb_%05d.png'%(opts.nvs_outpath,i), rgb[...,::-1]*255)
        cv2.imwrite('%s-sil_%05d.png'%(opts.nvs_outpath,i), sil*255)
        cv2.imwrite('%s-vis_%05d.png'%(opts.nvs_outpath,i), vis*255)
        cv2.imwrite('%s-dph_%05d.png'%(opts.nvs_outpath,i), dph[...,::-1]*255)
    save_vid('%s-rgb'%(opts.nvs_outpath), rgbs, suffix='.mp4')
    save_vid('%s-sil'%(opts.nvs_outpath), sils, suffix='.mp4')
    save_vid('%s-vis'%(opts.nvs_outpath), viss, suffix='.mp4')
    save_vid('%s-dph'%(opts.nvs_outpath), dphs, suffix='.mp4')


if __name__ == '__main__':
    app.run(main)
