from absl import flags, app
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R

from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam
opts = flags.FLAGS
                
def save_output(aux_seq, seqname):
    save_dir = '%s/%s'%(opts.model_path.rsplit('/',1)[0],seqname)
    length = len(aux_seq['mesh'])
    for i in range(length):
        idx = aux_seq['idx'][i]
        
        mesh = aux_seq['mesh'][i]
        mesh.export('%s-mesh-%05d.obj'%(save_dir, idx))
        
        rtk = aux_seq['rtk'][i]
        np.savetxt('%s-cam-%05d.txt'%(save_dir, idx), rtk)

        # convert bones to meshes
        if 'bone' in aux_seq.keys():
            bones = aux_seq['bone'][i]
            B = len(bones)
            elips_list = []
            elips = trimesh.creation.uv_sphere(radius=0.05,count=[16, 16])
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
            colormap = label_colormap()[:B]
            colormap= np.tile(colormap[:,None], (1,N_elips,1)).reshape((-1,3))
            elips.visual.vertex_colors[:len(colormap),:3] = colormap
            elips.export('%s-bone-%05d.obj'%(save_dir, idx))
            
    
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
    trainer.init_dataset()    
    trainer.define_model(no_ddp=True)
    seqname=opts.seqname
    num_view=0

    dynamic_mesh = opts.flowbw or opts.lbs
    rendered_seq, aux_seq = trainer.eval(num_view=num_view,
                                                    dynamic_mesh=dynamic_mesh) 
    save_output(aux_seq, seqname)

if __name__ == '__main__':
    app.run(main)
