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

       
cam_dir=sys.argv[1]
cap_frame=int(sys.argv[2])
save_dir,seqname=cam_dir.rsplit('/',1)
def main():
    # read all the data
    camlist = load_root(cam_dir, cap_frame)
    # construct camera mesh
    mesh = draw_cams(camlist)
    mesh.export('%s/mesh-%s.obj'%(save_dir, seqname))
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
