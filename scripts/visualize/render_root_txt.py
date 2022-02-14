# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)
import pdb
import glob
import numpy as np
import configparser
from utils.io import config_to_dataloader, draw_cams, render_root_txt

       
cam_dir=sys.argv[1]
cap_frame=int(sys.argv[2])
def main():
    render_root_txt(cam_dir, cap_frame)
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
