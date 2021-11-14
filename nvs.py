from absl import flags, app
import sys
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

from utils.io import save_vid, str_to_frame, save_bones, load_root
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
flags.DEFINE_string('rootdir', 
        'tmp/traj/',
        'root body directory')

def construct_rays_nvs(img_size, rtks, near_far, device):
    bs = rtks.shape[0]
    rtks = torch.Tensor(rtks).to(device)

    _, xys = sample_xy(img_size, bs, 0, device, return_all=True)
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
    del nerf_models['nerf_unc']
    embeddings = model.embeddings

    # bs, 4,4 (R|T)
    #         (f|p)
    rtks = load_root(opts.rootdir, opts.maxframe)  # cap frame=1000

    # determine render image scale
    rtks[:,3] = rtks[:,3]*opts.scale
    fl_mean = rtks[:,3,:2].mean()
    img_size = int(fl_mean)
    model.img_size = img_size
    opts.render_size = img_size

    #TODO need to resize according to object size
    # fl*obj_bound/depth ~= img_size/2
    rtks[:,:3,3] = rtks[:,:3,3] / model.obj_scale
    bs = len(rtks)
    
    vars_np = {}
    vars_np['rtk'] = rtks
    vars_np['idk'] = np.ones(bs)
    near_far = torch.zeros(bs,2).to(model.device)
    near_far = get_near_far(near_far,
                            vars_np,
                            pts=model.latest_vars['mesh_rest'].vertices)

    vidid = torch.Tensor([opts.vidid]).to(model.device).long()
    source_l = model.data_offset[opts.vidid+1] - model.data_offset[opts.vidid] -1
    embedid = torch.Tensor(np.linspace(0,source_l,bs)).to(model.device).long() + \
              model.data_offset[opts.vidid]
    if opts.bullet_time>-1: embedid[:] = opts.bullet_time
    rgbs = []
    sils = []
    viss = []
    for i in range(bs):
        rays = construct_rays_nvs(model.img_size, rtks[i:i+1], 
                                              near_far[i:i+1], model.device)
        # add env code
        rays['env_code'] = model.env_code(vidid)
        rays['env_code'] = rays['env_code'][:,None].repeat(1,rays['nsample'],1)
        
        # add bones
        time_embedded = model.pose_code(embedid[i:i+1])[:,None]
        rays['time_embedded'] = time_embedded.repeat(1,rays['nsample'],1)
        if opts.lbs and model.num_bone_used>0:
            bone_rts = model.nerf_bone_rts(embedid[i:i+1])
            rays['bone_rts'] = bone_rts.repeat(1,rays['nsample'],1)

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
                            )
                for k, v in rendered_chunks.items():
                    results[k] += [v]
           
        for k, v in results.items():
            v = torch.cat(v, 0)
            v = v.view(1,model.img_size, model.img_size, -1)
            results[k] = v
        rgb = results['img_coarse'][0].cpu().numpy()
        dph = results['depth_rnd'][0,...,0].cpu().numpy()
        sil = results['sil_coarse'][0,...,0].cpu().numpy()
        vis = results['vis_pred'][0,...,0].cpu().numpy()
        sil[sil<0.5] = 0
        rgb[sil<0.5] = 0
    
        rgbs.append(rgb)
        sils.append(sil*255)
        viss.append(vis*255)
        cv2.imwrite('tmp/rgb_%05d.png'%i, rgb[...,::-1]*255)
        cv2.imwrite('tmp/sil_%05d.png'%i, sil*255)
        cv2.imwrite('tmp/vis_%05d.png'%i, vis*255)
    save_vid('tmp/rgb', rgbs, suffix='.mp4')
    save_vid('tmp/sil', sils, suffix='.mp4')
    save_vid('tmp/vis', viss, suffix='.mp4')


if __name__ == '__main__':
    app.run(main)