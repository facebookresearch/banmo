from absl import flags, app
import numpy as np
import torch
import os
import glob
import pdb
import cv2
import trimesh

from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam
opts = flags.FLAGS
                
def save_output(mesh_seq, rtk_seq, rendered_seq, id_seq, seqname):
    save_dir = '%s/%s'%(opts.model_path.rsplit('/',1)[0],seqname)
    for mesh,rtk,idx in zip(mesh_seq,rtk_seq,id_seq):
        mesh, rtk = transform_shape(mesh, rtk)
        mesh.export('%s-%05d.obj'%(save_dir, idx))
        np.savetxt('%s-cam%d.txt'%(save_dir, idx), rtk)
    
def transform_shape(mesh,rtk):
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
    num_eval=0

    dynamic_mesh = opts.flowbw or opts.lbs
    rendered_seq, mesh_seq, rtk_seq, id_seq = trainer.eval(num_eval=num_eval,
                                                    dynamic_mesh=dynamic_mesh) 
    save_output(mesh_seq, rtk_seq, rendered_seq, id_seq, seqname)

if __name__ == '__main__':
    app.run(main)
