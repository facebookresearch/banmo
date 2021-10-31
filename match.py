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

from utils.io import save_vid, str_to_frame, save_bones, draw_lines
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam,\
                                Kmatinv, K2mat, K2inv, sample_xy, resample_dp,\
                                raycast
from nnutils.loss_utils import kp_reproj, feat_match
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS
                


def match_frames(trainer, idxs, nsample=50):
    idxs = [int(i) for i in idxs.split(' ')]
    opts = trainer.opts
    bs = len(idxs)
    device = trainer.device
    model = trainer.model

    # load frames
    for dataset in trainer.evalloader.dataset.datasets:
        dataset.load_pair = False
    batch = []
    for i in idxs:
        batch.append( trainer.evalloader.dataset[i] )
    batch = trainer.evalloader.collate_fn(batch)
    model.convert_batch_input(batch)
    model.convert_root_pose()
    rtk =   model.rtk
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])
    
    kaug =  model.kaug
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    Kinv = Kmatinv(Kaug.matmul(Kmat))

    embedid=model.embedid
    embedid = embedid.long().to(device)[:,None]

    near_far = model.near_far[model.frameid.long()]

    # sample points 
    rand_inds, xys = sample_xy(opts.img_size, bs, nsample, device,return_all=False) 
    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    rtk_vec = rays['rtk_vec']
    del rays
    dp_feats_rsmp = resample_dp(model.dp_feats,
            model.dp_bbox, kaug, opts.img_size)
    feats_at_samp = [dp_feats_rsmp[i].view(model.num_feat,-1).T\
                     [rand_inds[i].long()] for i in range(bs)]
    feats_at_samp = torch.stack(feats_at_samp,0) # bs,ns,num_feat

    # TODO implement for se3
    if opts.lbs and model.num_bone_used>0:
        bone_rts = model.nerf_bone_rts(embedid)
        bone_rts = bone_rts.repeat(1,nsample,1)

    # TODO rearrange inputs
    feats_at_samp = feats_at_samp.view(-1, model.num_feat)
    xys = xys.view(-1,1,2)
    rays = {'rtk_vec':  rtk_vec,
            'bone_rts': bone_rts}

    # project
    model.eval()
    with torch.no_grad():
        pts_pred = feat_match(model.nerf_feat, model.embedding_xyz, feats_at_samp,
            model.latest_vars['obj_bound'],grid_size=20,is_training=False)
        pts_pred = pts_pred.view(bs,nsample,3)
        xy_reproj = kp_reproj(pts_pred, xys, model.nerf_models, model.embedding_xyz, rays)

    # draw
    pdb.set_trace() 
    xy_reproj = xy_reproj.view(bs,nsample,2)
    xys = xys.view(bs,nsample, 2)
    sil_at_samp = torch.stack([model.masks[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
    for i in range(bs):
        img = model.imgs[i]
        valid_idx = sil_at_samp[i].bool()[...,0]
        p1s = xys[i][valid_idx]
        p2s = xy_reproj[i][valid_idx]
        img = draw_lines(img, p1s,p2s)
        cv2.imwrite('tmp/%04d.png'%i, img)

def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    #TODO write matching function
    img_match = match_frames(trainer, opts.match_frames)

if __name__ == '__main__':
    app.run(main)
