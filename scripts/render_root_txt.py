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
from utils.io import config_to_dataloader, draw_cams

       
cam_dir=sys.argv[1]
def main():
    # read all the data
    camlist = []
    for idx,path in enumerate(sorted(glob.glob('%s0*.txt'%(cam_dir)))):
        rtk = np.loadtxt(path)
        camlist.append(rtk)
    
    # construct camera mesh
    mesh = draw_cams(camlist)
    mesh.export('0.obj')
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
