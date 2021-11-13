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
    #TODO modify as args
    dataid = 0
    render_scale = 0.1
    root_dir = 'logdir/ama-female-lbs-rkopt-300-b16-init/init-cam/T_swing1-'
    
    rtks = load_root(root_dir,1000)  # cap frame=1000

    # determine render image scale
    rtks[:,3] = rtks[:,3]*render_scale
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

    rays = construct_rays_nvs(model.img_size, rtks, near_far, model.device)
    # TODO need to add more env_code
    dataid = torch.Tensor(bs*[dataid]).to(model.device).long()
    rays['env_code'] = model.env_code(dataid)
    rays['env_code'] = rays['env_code'][:,None].repeat(1,rays['nsample'],1)

    with torch.no_grad():
        # render images only
        results=defaultdict(list)
        bs_rays = rays['bs'] * rays['nsample'] #
        for i in range(0, bs_rays, opts.chunk):
            rays_chunk = chunk_rays(rays,i,opts.chunk)
            rendered_chunks = render_rays(nerf_models,
                        embeddings,
                        rays_chunk,
                        N_samples = opts.ndepth,
                        perturb=0,
                        noise_std=0,
                        chunk=opts.chunk, # chunk size is effective in val mode
                        use_fine=True,
                        img_size=model.img_size,
                        )
            for k, v in rendered_chunks.items():
                results[k] += [v]
       
    for k, v in results.items():
        v = torch.cat(v, 0)
        v = v.view(bs,model.img_size, model.img_size, -1)
        results[k] = v
    pdb.set_trace()
    rgb = results['img_coarse'].cpu().numpy()
    dph = results['depth_rnd'][...,0].cpu().numpy()
    sil = results['sil_coarse'][...,0].cpu().numpy()
    for i in range(bs):
        cv2.imwrite('tmp/rgb_%05d.png'%i, rgb[i,:,:,::-1]*255)
        cv2.imwrite('tmp/sil_%05d.png'%i, sil[i,:,:]*255)


if __name__ == '__main__':
    app.run(main)
