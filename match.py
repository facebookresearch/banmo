from absl import flags, app
import sys
sys.path.insert(0,'third_party')
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torch
import os
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio

from utils.io import save_vid, str_to_frame, save_bones, draw_lines, vis_viser
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam,\
                                Kmatinv, K2mat, K2inv, sample_xy, resample_dp,\
                                raycast
from nnutils.loss_utils import kp_reproj, feat_match, kp_reproj_loss
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS

def construct_rays(dp_feats_rsmp, model, xys, rand_inds,
        Rmat, Tmat, Kinv, near_far, flip=True):
    device = dp_feats_rsmp.device
    bs,nsample,_ =xys.shape
    opts = model.opts
    embedid=model.embedid
    embedid = embedid.long().to(device)[:,None]

    rays = raycast(xys, Rmat, Tmat, Kinv, near_far)
    rtk_vec = rays['rtk_vec']
    del rays
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
    if flip:
        rtk_vec = rtk_vec.view(bs//2,2,-1).flip(1).view(rtk_vec.shape)
        bone_rts = bone_rts.view(bs//2,2,-1).flip(1).view(bone_rts.shape)

    rays = {'rtk_vec':  rtk_vec,
            'bone_rts': bone_rts}

    return rays, feats_at_samp, xys
                

def match_frames(trainer, idxs, nsample=200):
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

    near_far = model.near_far[model.frameid.long()]
    
    # resample feature
    dp_feats_rsmp = resample_dp(model.dp_feats,
            model.dp_bbox, kaug, opts.img_size)

    # sample points 
    rand_inds, xys = sample_xy(opts.img_size, bs, nsample, device,return_all=False) 
    rays, feats_at_samp, xys = construct_rays(dp_feats_rsmp, model, xys, rand_inds,
                        Rmat, Tmat, Kinv, near_far)

    # project
    model.eval()
    with torch.no_grad():
        pts_pred = feat_match(model.nerf_feat, model.embedding_xyz, feats_at_samp,
            model.latest_vars['obj_bound'],grid_size=20,is_training=False)
        pts_pred = pts_pred.view(bs,nsample,3)
        xy_reproj = kp_reproj(pts_pred, xys, model.nerf_models, model.embedding_xyz, rays)

    # draw
    imgs_trg = model.imgs.view(bs//2,2,-1).flip(1).view(model.imgs.shape)
    xy_reproj = xy_reproj.view(bs,nsample,2)
    xys = xys.view(bs,nsample, 2)
    sil_at_samp = torch.stack([model.masks[i].view(-1,1)[rand_inds[i]] for i in range(bs)],0) # bs,ns,1
    for i in range(bs):
        img1 = model.imgs[i]
        img2 = imgs_trg[i]
        img = torch.cat([img1, img2],2)
        valid_idx = sil_at_samp[i].bool()[...,0]
        p1s = xys[i][valid_idx]
        p2s = xy_reproj[i][valid_idx]
        p2s[...,0] = p2s[...,0] + img1.shape[2]
        img = draw_lines(img, p1s,p2s)
        cv2.imwrite('tmp/%04d.png'%i, img)
            
    # visualization gcc error
    if opts.render_size<=128:
        rand_inds, xys = sample_xy(opts.img_size, bs, nsample, device,return_all=True) 
        rays, feats_at_samp, xys = construct_rays(dp_feats_rsmp, model, xys, rand_inds,
                        Rmat, Tmat, Kinv, near_far, flip=False)
        with torch.no_grad():
            pts_pred = feat_match(model.nerf_feat, model.embedding_xyz, feats_at_samp,
                model.latest_vars['obj_bound'],grid_size=20,is_training=False)
            pts_pred = pts_pred.view(bs,opts.render_size**2,3)

            proj_err = kp_reproj_loss(pts_pred, xys, model.nerf_models, 
                    model.embedding_xyz, rays)
            proj_err = proj_err.view(pts_pred.shape[:-1]+(1,))
            proj_err = proj_err/opts.img_size * 2
            results = {}
            results['feat_err']  =  proj_err
            vis_viser(results, model.masks, model.imgs, 
                        bs,opts.img_size, opts.ndepth)

    # visualize current error stats
    feat_err=model.latest_vars['fp_err'][:,0] 
    proj_err=model.latest_vars['fp_err'][:,1] 
    feat_err = feat_err[feat_err>0]
    proj_err = proj_err[proj_err>0]
    print('feat-med: %f'%(np.median(feat_err)))
    print('proj-med: %f'%(np.median(proj_err)))
    plt.hist(feat_err,bins=100)
    plt.savefig('tmp/viser_feat_err.jpg')
    plt.clf()
    plt.hist(proj_err,bins=100)
    plt.savefig('tmp/viser_proj_err.jpg')

    # visualize codes
    with torch.no_grad():
        pdb.set_trace()
        D=model.pose_code.weight
        D = D-D.mean(0)[None]
        A = D.T.matmul(D)/D.shape[0] # fxf
        U,S,V=torch.svd(A) # 
        out=D.matmul(V[:,:3])
        cmap = matplotlib.cm.get_cmap('cool')
        time = np.asarray(range(len(model.impath)))
        time = time/time.max()
        out=out.detach().cpu().numpy()
        trimesh.Trimesh(out, vertex_colors=cmap(time)).export('tmp/0.obj')
        #plt.scatter(out[:,0], out[:,1], color=cmap(time))
        #plt.xlim(out[:,0].min(), out[:,0].max())
        #plt.ylim(out[:,1].min(), out[:,1].max())
        #plt.savefig('tmp/0.jpg')
    

def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    #TODO write matching function
    img_match = match_frames(trainer, opts.match_frames)

if __name__ == '__main__':
    app.run(main)
