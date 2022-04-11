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
flags.DEFINE_integer('maxframe', 0, 'maximum number frame to render')
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
    rtks = load_root(opts.rootdir, 0)  # cap frame=0=>load all
    rndsils = load_sils(opts.rootdir.replace('ctrajs', 'refsil'),0)
    if opts.maxframe>0:
        sample_idx = np.linspace(0,len(rtks)-1,opts.maxframe).astype(int)
        rtks = rtks[sample_idx]
        rndsils = rndsils[sample_idx]
    else:
        sample_idx = np.linspace(0,len(rtks)-1, len(rtks)).astype(int)
    img_size = rndsils[0].shape
    if img_size[0] > img_size[1]:
        img_type='vert'
    else:
        img_type='hori'

    # determine render image scale
    rtks[:,3] = rtks[:,3]*opts.scale
    bs = len(rtks)
    img_size = int(max(img_size)*opts.scale)
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

    vidid = torch.Tensor([opts.vidid]).to(model.device).long()
    source_l = model.data_offset[opts.vidid+1] - model.data_offset[opts.vidid] -1
    embedid = torch.Tensor(sample_idx).to(model.device).long() + \
              model.data_offset[opts.vidid]
    if opts.bullet_time>-1: embedid[:] = opts.bullet_time+model.data_offset[opts.vidid]
    print(embedid)
    rgbs = []
    sils = []
    viss = []
    for i in range(bs):
        rndsil = rndsils[i]
        rndmask = np.zeros((img_size, img_size))
        if img_type=='vert':
            size_short_edge = int(rndsil.shape[1] * img_size/rndsil.shape[0])
            rndsil = cv2.resize(rndsil, (size_short_edge, img_size))
            rndmask[:,:size_short_edge] = rndsil
        else:
            size_short_edge = int(rndsil.shape[0] * img_size/rndsil.shape[1])
            rndsil = cv2.resize(rndsil, (img_size, size_short_edge))
            rndmask[:size_short_edge] = rndsil
        rays = construct_rays_nvs(model.img_size, rtks[i:i+1], 
                                       near_far[i:i+1], rndmask, model.device)
        # add env code
        rays['env_code'] = model.env_code(embedid[i:i+1])[:,None]
        rays['env_code'] = rays['env_code'].repeat(1,rays['nsample'],1)
        
        # add bones
        time_embedded = model.pose_code(embedid[i:i+1])[:,None]
        rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        if opts.lbs and model.num_bone_used>0:
            bone_rts = model.nerf_body_rts(embedid[i:i+1])
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)
            model.update_delta_rts(rays)

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
        sil[sil<0.5] = 0
        rgb[sil<0.5] = 1

        rgbtmp = np.ones((img_size, img_size, 3))
        dphtmp = np.ones((img_size, img_size))
        siltmp = np.ones((img_size, img_size))
        vistmp = np.ones((img_size, img_size))
        rgbtmp[rndmask>0] = rgb
        dphtmp[rndmask>0] = dph
        siltmp[rndmask>0] = sil
        vistmp[rndmask>0] = vis

        if img_type=='vert':
            rgb = rgbtmp[:,:size_short_edge]
            sil = siltmp[:,:size_short_edge]
            vis = vistmp[:,:size_short_edge]
            dph = dphtmp[:,:size_short_edge]
        else:
            rgb = rgbtmp[:size_short_edge]
            sil = siltmp[:size_short_edge]
            vis = vistmp[:size_short_edge]
            dph = dphtmp[:size_short_edge]
    
        rgbs.append(rgb)
        sils.append(sil*255)
        viss.append(vis*255)
        cv2.imwrite('%s-rgb_%05d.png'%(opts.nvs_outpath,i), rgb[...,::-1]*255)
        cv2.imwrite('%s-sil_%05d.png'%(opts.nvs_outpath,i), sil*255)
        cv2.imwrite('%s-vis_%05d.png'%(opts.nvs_outpath,i), vis*255)
    save_vid('%s-rgb'%(opts.nvs_outpath), rgbs, suffix='.mp4',upsample_frame=0)
    save_vid('%s-sil'%(opts.nvs_outpath), sils, suffix='.mp4',upsample_frame=0)
    save_vid('%s-vis'%(opts.nvs_outpath), viss, suffix='.mp4',upsample_frame=0)


if __name__ == '__main__':
    app.run(main)
