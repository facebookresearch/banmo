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
from nnutils.geom_utils import process_so3_seq

       
file_path=sys.argv[1]
save_dir,seqname=file_path.rsplit('/',1)
def main():
    # read all the data
    rtk_seq = np.load(file_path, allow_pickle=True)
    process_so3_seq(rtk_seq)
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
