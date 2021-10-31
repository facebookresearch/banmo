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

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
from nnutils.geom_utils import obj_to_cam, tensor2array, vec_to_sim3, obj_to_cam
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
opts = flags.FLAGS
                
def main(_):
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)

    #TODO write matching function
    img_match = match_frames(trainer.model, idx1, idx2)

if __name__ == '__main__':
    app.run(main)
