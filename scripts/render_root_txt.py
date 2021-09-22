import sys, os
import pdb
import glob
import numpy as np
import torch
import cv2
import soft_renderer as sr
import argparse
import trimesh
import configparser

       
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
