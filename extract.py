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

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS
                
def save_output(rendered_seq, aux_seq, seqname, save_flo):
    save_dir = '%s/'%(opts.model_path.rsplit('/',1)[0])
    length = len(aux_seq['mesh'])
    mesh_rest = aux_seq['mesh_rest']
    len_max = (mesh_rest.vertices.max(0) - mesh_rest.vertices.min(0)).max()
    mesh_rest.export('%s/mesh-rest.obj'%save_dir)
    aux_seq['mesh_rest_skin'].export('%s/mesh-rest-skin.obj'%save_dir)
    if 'bone_rest' in aux_seq.keys():
        bone_rest = aux_seq['bone_rest']
        save_bones(bone_rest, len_max, '%s/bone-rest.obj'%save_dir)

    flo_gt_vid = []
    flo_p_vid = []
    for i in range(length):
        impath = aux_seq['impath'][i]
        seqname = impath.split('/')[-2]
        save_prefix = '%s/%s'%(save_dir,seqname)
        idx = int(impath.split('/')[-1].split('.')[-2])
        mesh = aux_seq['mesh'][i]
        rtk = aux_seq['rtk'][i]
        
        if len(aux_seq['sim3_j2c'])>0:
            # save the video-specific mesh and bones
            sim3_j2c = aux_seq['sim3_j2c'][i]
            sim3_j2c = torch.Tensor(sim3_j2c)
            Tmat_j2c, Rmat_j2c, Smat_j2c = vec_to_sim3(sim3_j2c)
            Smat_j2c = Smat_j2c.mean(-1)[...,None]

        # convert bones to meshes TODO: warp with a function
        if 'bone' in aux_seq.keys() and len(aux_seq['bone'])>0:
            bones = aux_seq['bone'][i]
            bone_path = '%s-bone-%05d.obj'%(save_prefix, idx)
            save_bones(bones, len_max, bone_path)
       
        if len(aux_seq['sim3_j2c'])>0:
            # save the cameras relative to the joint canonical model
            rtk[:3,3]  = rtk[:3,:3].dot(Tmat_j2c[...,None]*Smat_j2c)[...,0] + rtk[:3,3]
            rtk[:3,:3] = rtk[:3,:3].dot(Rmat_j2c)
            np.savetxt('%s-scale-%05d.txt'%(save_prefix, idx), Smat_j2c)
        
        mesh.export('%s-mesh-%05d.obj'%(save_prefix, idx))
        np.savetxt('%s-cam-%05d.txt'  %(save_prefix, idx), rtk)
            
        img_gt = rendered_seq['img'][i]
        flo_gt = rendered_seq['flo'][i]
        mask_gt = rendered_seq['sil'][i][...,0]
        flo_gt[mask_gt<=0] = 0
        img_gt[mask_gt<=0] = 1
        if save_flo: img_gt = cat_imgflo(img_gt, flo_gt)
        else: img_gt*=255
        cv2.imwrite('%s-img-gt-%05d.jpg'%(save_prefix, idx), img_gt[...,::-1])
        flo_gt_vid.append(img_gt)
        
        img_p = rendered_seq['img_coarse'][i]
        flo_p = rendered_seq['flo_coarse'][i]
        mask_gt = cv2.resize(mask_gt, flo_p.shape[:2][::-1]).astype(bool)
        flo_p[mask_gt<=0] = 0
        img_p[mask_gt<=0] = 1
        if save_flo: img_p = cat_imgflo(img_p, flo_p)
        else: img_p*=255
        cv2.imwrite('%s-img-p-%05d.jpg'%(save_prefix, idx), img_p[...,::-1])
        flo_p_vid.append(img_p)

        flo_gt = cv2.resize(flo_gt, flo_p.shape[:2])
        flo_err = np.linalg.norm( flo_p - flo_gt ,2,-1)
        flo_err_med = np.median(flo_err[mask_gt])
        flo_err[~mask_gt] = 0.
        cv2.imwrite('%s-flo-err-%05d.jpg'%(save_prefix, idx), 
                128*flo_err/flo_err_med)

        img_gt = rendered_seq['img'][i]
        img_p = rendered_seq['img_coarse'][i]
        img_gt = cv2.resize(img_gt, img_p.shape[:2][::-1])
        img_err = np.power(img_gt - img_p,2).sum(-1)
        img_err_med = np.median(img_err[mask_gt])
        img_err[~mask_gt] = 0.
        cv2.imwrite('%s-img-err-%05d.jpg'%(save_prefix, idx), 
                128*img_err/img_err_med)


#    fps = 1./(5./len(flo_p_vid))
    upsample_frame = min(30, len(flo_p_vid))
    save_vid('%s-img-p' %(save_prefix), flo_p_vid, upsample_frame=upsample_frame)
    save_vid('%s-img-gt' %(save_prefix),flo_gt_vid,upsample_frame=upsample_frame)

def transform_shape(mesh,rtk):
    """
    (deprecated): absorb rt into mesh vertices, 
    """
    vertices = torch.Tensor(mesh.vertices)
    Rmat = torch.Tensor(rtk[:3,:3])
    Tmat = torch.Tensor(rtk[:3,3])
    vertices = obj_to_cam(vertices, Rmat, Tmat)

    rtk[:3,:3] = np.eye(3)
    rtk[:3,3] = 0.
    mesh = trimesh.Trimesh(vertices.numpy(), mesh.faces)
    return mesh, rtk

def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    seqname=opts.seqname

    dynamic_mesh = opts.flowbw or opts.lbs
    idx_render = str_to_frame(opts.test_frames, data_info)
#    idx_render[0] += 50
#    idx_render[0] += 374
#    idx_render[0] += 292
#    idx_render[0] += 10
#    idx_render[0] += 340
#    idx_render[0] += 440
#    idx_render[0] += 540
#    idx_render[0] += 640
#    idx_render[0] += trainer.model.data_offset[4]-4 + 37
#    idx_render[0] += 36

    trainer.model.img_size = opts.render_size
    chunk = opts.frame_chunk
    for i in range(0, len(idx_render), chunk):
        rendered_seq, aux_seq = trainer.eval(idx_render=idx_render[i:i+chunk],
                                             dynamic_mesh=dynamic_mesh) 
        rendered_seq = tensor2array(rendered_seq)
        save_output(rendered_seq, aux_seq, seqname, save_flo=opts.use_corresp)
    #TODO merge the outputs

if __name__ == '__main__':
    app.run(main)
