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

    flo_gt_vid = []
    flo_p_vid = []
    for i in range(length):
        impath = aux_seq['impath'][i]
        seqname = impath.split('/')[-2]
        save_prefix = '%s/%s'%(save_dir,seqname)
        idx = int(impath.split('/')[-1].split('.')[-2])
        mesh = aux_seq['mesh'][i]
        rtk = aux_seq['rtk'][i]
        sim3_j2c = aux_seq['sim3_j2c'][i]
                
        # save the video-specific mesh and bones
        sim3_j2c = torch.Tensor(sim3_j2c)
        Tmat_j2c, Rmat_j2c, Smat_j2c = vec_to_sim3(sim3_j2c)
        Smat_j2c = Smat_j2c.mean(-1)[...,None]

        # convert bones to meshes TODO: warp with a function
        if 'bone' in aux_seq.keys() and len(aux_seq['bone'])>0:
            bones = aux_seq['bone'][i]
            B = len(bones)
            elips_list = []
            elips = trimesh.creation.uv_sphere(radius=len_max/20,count=[16, 16])
            # remove identical vertices
            elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
            N_elips = len(elips.vertices)
            for bone in bones:
                center = bone[None,:3]
                orient = bone[3:7] # real first
                orient = orient / np.linalg.norm(orient, 2,-1)
                orient = R.from_quat(orient).as_matrix() # real first
                orient = orient.T # transpose R
                scale =  np.exp(bone[None, 7:10])

                elips_verts = elips.vertices
                elips_verts = elips_verts / scale
                elips_verts = elips_verts.dot(orient)
                elips_verts = elips_verts+center
                elips_list.append( trimesh.Trimesh(vertices = elips_verts, 
                                                           faces=elips.faces) )
            elips = trimesh.util.concatenate(elips_list)
            #elips_verts = torch.Tensor(elips.vertices)
            #elips_verts = obj_to_cam(elips_verts, Rmat_j2c, Tmat_j2c)
            #elips_verts = elips_verts * Smat_j2c
            #elips.vertices = elips_verts.numpy()
            
            colormap = label_colormap()[:B]
            colormap= np.tile(colormap[:,None], (1,N_elips,1)).reshape((-1,3))
            elips.visual.vertex_colors[:len(colormap),:3] = colormap
            elips.export('%s-bone-%05d.obj'%(save_prefix, idx))
        
        #mesh_verts = torch.Tensor(mesh.vertices)
        #mesh_verts = obj_to_cam(mesh_verts, Rmat_j2c, Tmat_j2c)
        #mesh_verts = mesh_verts * Smat_j2c
        #mesh.vertices = mesh_verts.numpy()
        # save the cameras relative to the joint canonical model
        rtk[:3,:3] = rtk[:3,:3].dot(Rmat_j2c)
        rtk[:3,3]  = rtk[:3,:3].dot(Tmat_j2c[...,None]*Smat_j2c)[...,0] + rtk[:3,3]
        
        mesh.export('%s-mesh-%05d.obj'%(save_prefix, idx))
        np.savetxt('%s-cam-%05d.txt'  %(save_prefix, idx), rtk)
        np.savetxt('%s-scale-%05d.txt'%(save_prefix, idx), Smat_j2c)
            
        img_gt = rendered_seq['img'][i]
        flo_gt = rendered_seq['flo'][i]
        if save_flo: img_gt = cat_imgflo(img_gt, flo_gt)
        cv2.imwrite('%s-img-gt-%05d.jpg'%(save_prefix, idx), img_gt)
        flo_gt_vid.append(img_gt)
        
        img_p = rendered_seq['img_coarse'][i]
        flo_p = rendered_seq['flo_coarse'][i]
        if save_flo: img_p = cat_imgflo(img_p, flo_p)
        cv2.imwrite('%s-img-p-%05d.jpg'%(save_prefix, idx), img_p)
        flo_p_vid.append(img_p)

    fps = 1./(5./len(flo_p_vid))
    imageio.mimsave('%s-img-p.gif' %(save_prefix), flo_p_vid, fps=fps)
    imageio.mimsave('%s-img-gt.gif'%(save_prefix), flo_gt_vid,fps=fps)

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
    trainer = v2s_trainer(opts)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info, no_ddp=True, half_bones=True)
    seqname=opts.seqname

    dynamic_mesh = opts.flowbw or opts.lbs
    rendered_seq, aux_seq = trainer.eval(num_view=opts.num_test_views,
                                                    dynamic_mesh=dynamic_mesh) 
    rendered_seq = tensor2array(rendered_seq)
    save_output(rendered_seq, aux_seq, seqname, save_flo=opts.use_corresp)

if __name__ == '__main__':
    app.run(main)
