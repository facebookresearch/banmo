# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# TODO: pass ft_cse to use fine-tuned feature
# TODO: pass fine_steps -1 to use fine samples
from absl import flags, app
import sys
sys.path.insert(0,'')
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

from utils.io import save_vid, str_to_frame, save_bones, draw_lines, vis_match
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
        bone_rts = model.nerf_body_rts(embedid)
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
    bs = len(idxs)
    opts = trainer.opts
    device = trainer.device
    model = trainer.model
    model.eval()

    # load frames and aux data
    for dataset in trainer.evalloader.dataset.datasets:
        dataset.load_pair = False
    batch = []
    for i in idxs:
        batch.append( trainer.evalloader.dataset[i] )
    batch = trainer.evalloader.collate_fn(batch)

    model.set_input(batch)
    rtk =   model.rtk
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])

    kaug =  model.kaug # according to cropping, p = Kaug Kmat P
    Kaug = K2inv(kaug)
    Kinv = Kmatinv(Kaug.matmul(Kmat))

    near_far = model.near_far[model.frameid.long()]
    dp_feats_rsmp = model.dp_feats

    # construct rays for sampled pixels
    rand_inds, xys = sample_xy(opts.img_size, bs, nsample, device,return_all=False)
    rays, feats_at_samp, xys = construct_rays(dp_feats_rsmp, model, xys, rand_inds,
                        Rmat, Tmat, Kinv, near_far)
    model.update_delta_rts(rays)

    # re-project
    with torch.no_grad():
        pts_pred = feat_match(model.nerf_feat, model.embedding_xyz, feats_at_samp,
            model.latest_vars['obj_bound'],grid_size=20,is_training=False)
        pts_pred = pts_pred.view(bs,nsample,3)
        xy_reproj = kp_reproj(pts_pred, model.nerf_models, model.embedding_xyz, rays)

    # draw
    imgs_trg = model.imgs.view(bs//2,2,-1).flip(1).view(model.imgs.shape)
    xy_reproj = xy_reproj.view(bs,nsample,2)
    xys = xys.view(bs,nsample, 2)
    sil_at_samp = torch.stack([model.masks[i].view(-1,1)[rand_inds[i]] \
                                                for i in range(bs)],0) # bs,ns,1
    for i in range(bs):
        img1 = model.imgs[i]
        img2 = imgs_trg[i]
        img = torch.cat([img1, img2],2)
        valid_idx = sil_at_samp[i].bool()[...,0]
        p1s = xys[i][valid_idx]
        p2s = xy_reproj[i][valid_idx]
        p2s[...,0] = p2s[...,0] + img1.shape[2]
        img = draw_lines(img, p1s,p2s)
        cv2.imwrite('tmp/match_%04d.png'%i, img)

    # visualize matching error
    if opts.render_size<=128:
        with torch.no_grad():
            rendered, rand_inds = model.nerf_render(rtk, kaug, model.embedid,
                nsample=opts.nsample, ndepth=opts.ndepth)
            xyz_camera   = rendered['xyz_camera_vis'][0].reshape(opts.render_size**2,-1)
            xyz_canonical = rendered['xyz_canonical_vis'][0].reshape(opts.render_size**2,-1)
            skip_idx = len(xyz_camera)//50 # vis 50 rays
            trimesh.Trimesh(xyz_camera[0::skip_idx].reshape(-1,3).cpu()).\
                    export('tmp/match_camera_pts.obj')
            trimesh.Trimesh(xyz_canonical[0::skip_idx].reshape(-1,3).cpu()).\
                    export('tmp/match_canonical_pts.obj')
            vis_match(rendered, model.masks, model.imgs,
                    bs,opts.img_size, opts.ndepth)
        ## construct rays for all pixels
        #rand_inds, xys = sample_xy(opts.img_size, bs, nsample, device,return_all=True)
        #rays, feats_at_samp, xys = construct_rays(dp_feats_rsmp, model, xys, rand_inds,
        #                Rmat, Tmat, Kinv, near_far, flip=False)
        #with torch.no_grad():
        #    pts_pred = feat_match(model.nerf_feat, model.embedding_xyz, feats_at_samp,
        #        model.latest_vars['obj_bound'],grid_size=20,is_training=False)
        #    pts_pred = pts_pred.view(bs,opts.render_size**2,3)

        #    proj_err = kp_reproj_loss(pts_pred, xys, model.nerf_models,
        #            model.embedding_xyz, rays)
        #    proj_err = proj_err.view(pts_pred.shape[:-1]+(1,))
        #    proj_err = proj_err/opts.img_size * 2
        #    results = {}
        #    results['proj_err']  =  proj_err


    ## visualize current error stats
    #feat_err=model.latest_vars['fp_err'][:,0]
    #proj_err=model.latest_vars['fp_err'][:,1]
    #feat_err = feat_err[feat_err>0]
    #proj_err = proj_err[proj_err>0]
    #print('feat-med: %f'%(np.median(feat_err)))
    #print('proj-med: %f'%(np.median(proj_err)))
    #plt.hist(feat_err,bins=100)
    #plt.savefig('tmp/viser_feat_err.jpg')
    #plt.clf()
    #plt.hist(proj_err,bins=100)
    #plt.savefig('tmp/viser_proj_err.jpg')

    # visualize codes
    with torch.no_grad():
        fid = torch.Tensor(range(0,len(model.impath))).cuda().long()
        D=model.pose_code(fid)
        D = D.view(len(fid),-1)
        ##TODO
        #px = torch.Tensor(range(len(D))).cuda()
        #py = px*2
        #pz = px*5+1
        #D = torch.stack([px,py,pz],-1)

        D = D-D.mean(0)[None]
        A = D.T.matmul(D)/D.shape[0] # fxf
        U,S,V=torch.svd(A) #
        code_proj_3d=D.matmul(V[:,:3])
        cmap = matplotlib.cm.get_cmap('cool')
        time = np.asarray(range(len(model.impath)))
        time = time/time.max()
        code_proj_3d=code_proj_3d.detach().cpu().numpy()
        trimesh.Trimesh(code_proj_3d, vertex_colors=cmap(time)).export('tmp/0.obj')

        #plt.figure(figsize=(16,16))
        plot_stack = []
        weight_dir = opts.model_path.rsplit('/',1)[0]
        bne_path = sorted(glob.glob('%s/%s-*bne-mrender*.jpg'%\
                            (weight_dir, opts.seqname)))
        img_path = model.impath.copy()
        ## remove the last img for each video to make shape consistent with bone renders
        #for i in model.data_offset[1:][::-1]:
        #    img_path.remove(img_path[i-1])
        #    code_proj_3d = np.delete(code_proj_3d, i-1,0)
        # plot the first video
        img_path = img_path        [:model.data_offset[1]-2]
        code_proj_3d = code_proj_3d[:model.data_offset[1]-2]
        try:
            bne_path = bne_path    [:model.data_offset[1]-2]
        except:
            pass

        for i in range(len(code_proj_3d)):
            plt.plot(code_proj_3d[i,0], code_proj_3d[i,1], color=cmap(time[i]), marker='o')
            plt.annotate(str(i), (code_proj_3d[i,0], code_proj_3d[i,1]))
            plt.xlim(code_proj_3d[:,0].min(), code_proj_3d[:,0].max())
            plt.ylim(code_proj_3d[:,1].min(), code_proj_3d[:,1].max())

            fig = plt.gcf()
            fig.canvas.draw()
            plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            print('plot pose code of frame id:%03d'%i)
            if len(bne_path) == len(code_proj_3d):
                bneimg = cv2.imread(bne_path[i])
                bneimg = cv2.resize(bneimg,\
                (bneimg.shape[1]*plot.shape[0]//bneimg.shape[0], plot.shape[0]))
                img=cv2.imread(img_path[i])[:,:,::-1]
                img = cv2.resize(img,\
                (img.shape[1]*plot.shape[0]//img.shape[0], plot.shape[0]))
                plot = np.hstack([img, bneimg, plot])
            plot_stack.append(plot)

        save_vid('tmp/code', plot_stack, suffix='.mp4',
                upsample_frame=150.,fps=30)
        save_vid('tmp/code', plot_stack, suffix='.gif',
                upsample_frame=150.,fps=30)

    # vis dps
    cv2.imwrite('tmp/match_dpc.png', model.dp_vis[model.dps[0].long()].cpu().numpy()*255)


def main(_):
    opts.img_size=opts.render_size
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()
    trainer.define_model(data_info)

    #write matching function
    img_match = match_frames(trainer, opts.match_frames)

if __name__ == '__main__':
    app.run(main)
