import sys, os
import pdb
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)

import subprocess
import imageio
import glob
from utils.io import save_vid
from ext_utils.badja_data import BADJAData
from ext_utils.joint_catalog import SMALJointInfo
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import soft_renderer as sr
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, obj2cam_np
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer
import configparser
import matplotlib
cmap = matplotlib.cm.get_cmap('cool')

from utils.io import config_to_dataloader, draw_cams


parser = argparse.ArgumentParser(description='script to render cameras over epochs')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--cap_frame', default=-1,type=int,
                    help='number of frames to cap')
parser.add_argument('--first_idx', default=0,type=int,
                    help='first frame index to vis')
parser.add_argument('--last_idx', default=-1,type=int,
                    help='last frame index to vis')
args = parser.parse_args()
        
img_size = 1024

def main():
    # read all the data
    logname = args.testdir.split('/')[-2]
    varlist = [i for i in glob.glob('%s/vars_*.npy'%args.testdir) \
                        if 'latest.npy' not in i]
    varlist = sorted(varlist, 
            key=lambda x:int(x.split('/')[-1].split('vars_')[-1].split('.npy')[0]))
    
    # get first index that is used for optimization
    var = np.load(varlist[-1],allow_pickle=True)[()]
    var['rtk'] = var['rtk'][args.first_idx:args.last_idx] 
    first_valid_idx = np.linalg.norm(var['rtk'][:,:3,3], 2,-1)>0
    first_valid_idx = np.argmax(first_valid_idx)
    #varlist = varlist[1:]
    if args.cap_frame>-1:
        varlist = varlist[:args.cap_frame]
    size = len(varlist)

    mesh_cams = []
    mesh_objs = []
    for var_path in varlist:
        # construct camera mesh
        var = np.load(var_path,allow_pickle=True)[()]
        var['rtk'] = var['rtk'][args.first_idx:args.last_idx] 
        mesh_cams.append(draw_cams(var['rtk'][first_valid_idx:]))
        mesh_objs.append(var['mesh_rest'])
       
    frames = []
    # process cameras
    for i in range(size):
        print(i)
        refcam = var['rtk'][first_valid_idx].copy()
        # median camera trans
        mtrans = np.median(np.linalg.norm(var['rtk'][first_valid_idx:,:3,3],2,-1)) 
        refcam[:2,3] = 0  # trans xy
        refcam[2,3] = 4*mtrans # depth
        refcam[3,:2] = 3*img_size/2 # fl
        refcam[3,2] = img_size/2
        refcam[3,3] = img_size/2
        vp_rmat = refcam[:3,:3]
        vp_rmat = cv2.Rodrigues(np.asarray([np.pi/2,0,0]))[0].dot(vp_rmat) # bev
        refcam[:3,:3] = vp_rmat

        # load vertices
        refmesh = mesh_cams[i]
        refface = torch.Tensor(refmesh.faces[None]).cuda()
        verts = torch.Tensor(refmesh.vertices[None]).cuda()

        # render
        Rmat =  torch.Tensor(refcam[None,:3,:3]).cuda()
        Tmat =  torch.Tensor(refcam[None,:3,3]).cuda()
        ppoint =refcam[3,2:]
        focal = refcam[3,:2]

        verts = obj_to_cam(verts, Rmat, Tmat)


        r = OffscreenRenderer(img_size, img_size)
        colors = refmesh.visual.vertex_colors
        
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
        colors= np.concatenate([0.6*colors[:,:3].astype(np.uint8), colors[:,3:]],-1)  # avoid overexposure
            
        smooth=True
        mesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:,:3].cpu()), faces=np.asarray(refface[0].cpu()),vertex_colors=colors)
        meshr = Mesh.from_trimesh(mesh,smooth=smooth)
        meshr._primitives[0].material.RoughnessFactor=.5
        scene.add_node( Node(mesh=meshr ))

        mesh_obj = mesh_objs[i]
        if len(mesh_obj.vertices)>0:
            mesh_obj.vertices = obj2cam_np(mesh_obj.vertices, Rmat, Tmat)
            mesh_obj=Mesh.from_trimesh(mesh_obj,smooth=smooth)
            mesh_obj._primitives[0].material.RoughnessFactor=1.
            scene.add_node( Node(mesh=mesh_obj))

        cam = IntrinsicsCamera(
                focal[0],
                focal[0],
                ppoint[0],
                ppoint[1],
                znear=1e-3,zfar=1000)
        cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
        cam_node = scene.add(cam, pose=cam_pose)
        light_pose =np.asarray([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        direc_l_node = scene.add(direc_l, pose=light_pose)
        color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        
        # save image
        color = color.astype(np.uint8)
        color = cv2.putText(color, 'epoch: %02d'%(i), (30,50), 
                cv2.FONT_HERSHEY_SIMPLEX,2, (256,0,0), 2)
        imoutpath = '%s/mesh-cam-%02d.png'%(args.testdir,i)
        cv2.imwrite(imoutpath,color[:,:,::-1] )
        frames.append(color)

    save_vid('%s/mesh-cam'%args.testdir, frames, suffix='.gif', 
            upsample_frame=-1, fps=2)
    save_vid('%s/mesh-cam'%args.testdir, frames, suffix='.mp4', 
            upsample_frame=-1, fps=2)
if __name__ == '__main__':
    main()
