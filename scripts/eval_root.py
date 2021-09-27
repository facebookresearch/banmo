import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)
import pdb
import glob
import numpy as np
import torch
import cv2
import soft_renderer as sr
import argparse
import trimesh
import configparser
from utils.io import config_to_dataloader, draw_cams, load_root
from nnutils.geom_utils import rot_angle
       
root_a_dir=sys.argv[1]
root_b_dir=sys.argv[2]
cap_frame=int(sys.argv[3])


def align_se3(rootlist_a, rootlist_b):
    dso3 = rootlist_b[0,:3,:3].T.dot(rootlist_a[0,:3,:3])
    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3[None])

    dtrn = np.linalg.norm(rootlist_a[0,:3,3],2,-1) /\
           np.linalg.norm(rootlist_b[0,:3,3],2,-1)
    rootlist_b[:,:3,3] = rootlist_b[:,:3,3] * dtrn

    so3_err = np.matmul(rootlist_a, np.transpose(rootlist_b,[0,2,1]))
    so3_err = rot_angle(torch.Tensor(so3_err))
    so3_err = so3_err / np.pi*180
    so3_err_max = so3_err.max()
    so3_err_mean = so3_err.mean()
    print('max  so3 error (deg): %.1f'%(so3_err_max))
    print('mean so3 error (deg): %.1f'%(so3_err_mean))

    return rootlist_b

def main():
    rootlist_a = load_root(root_a_dir, cap_frame)
    rootlist_b = load_root(root_b_dir, cap_frame)

    # align
    rootlist_b = align_se3(rootlist_a, rootlist_b)

    # construct camera mesh
    mesh_a = draw_cams(rootlist_a)
    mesh_a.visual.vertex_colors[:,:3] = [128,0,0]
    mesh_a.vertex_colors = 1
    mesh_b = draw_cams(rootlist_b)
    mesh = trimesh.util.concatenate([mesh_a, mesh_b])
    mesh.export('0.obj')
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
