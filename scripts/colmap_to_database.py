# @title Define imports and utility functions.
import pdb
from absl import logging
from io import BytesIO
import numpy as np
import PIL
import tempfile
import sys
nerfies_dir=sys.argv[1]
import os

# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.

sys.path.insert(0, nerfies_dir)
from pathlib import Path
from pprint import pprint
from nerfies import configs

dataname = sys.argv[2]
# @markdown The working directory where the trained model is.
train_dir = './logs/cat_501/'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = 'third_party/nerfies-0.1/dataset/%s/'%dataname  # @param {type: "string"}

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

exp_config = configs.ExperimentConfig()
model_config = configs.ModelConfig()
eval_config = configs.EvalConfig()

# @title Create datasource and show an example.

from nerfies import datasets
from nerfies import image_utils

datasource_spec = exp_config.datasource_spec
if datasource_spec is None:
    datasource_spec = {
        'type': exp_config.datasource_type,
        'data_dir': data_dir,
    }
datasource = datasets.from_config(
  datasource_spec,
  #image_scale=exp_config.image_scale,
  image_scale=1,
  use_appearance_id=model_config.use_appearance_metadata,
  use_camera_id=model_config.use_camera_metadata,
  use_warp_id=model_config.use_warp)

# @title Initialize model
# @markdown Defines the model and initializes its parameters.

from nerfies import models
from nerfies import model_utils
from nerfies import schedules
from nerfies import training
from nerfies import utils

# @title Define pmapped render function.

import functools
from nerfies import evaluation
from importlib import reload  


import trimesh
import mcubes
import cv2
import shutil
save_data=True
seqname='nerfies_%s'%dataname
logname='nerfies-%s'%seqname
bound=(datasource.far-datasource.near)/2

test_camera_paths = datasource.glob_cameras(Path(data_dir, 'camera'))
#test_camera_paths = datasource.glob_cameras('dataset/syn-eagled-/camera/')
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)
frame_len=len(test_cameras)
#render_len=15
render_len=frame_len
#grid_size = 128*2
grid_size = 12
threshold = 20

if save_data:
    cam_save_path = '../vid2shape/database/DAVIS/Cameras/Full-Resolution/%s/'%seqname
    rgb_save_path = '../vid2shape/database/DAVIS/JPEGImages/Full-Resolution/%s/'%seqname
    sil_save_path = '../vid2shape/database/DAVIS/Annotations/Full-Resolution/%s/'%seqname
    try:
        shutil.rmtree(rgb_save_path)
        shutil.rmtree(sil_save_path)
        shutil.rmtree(cam_save_path)
    except: pass
    os.mkdir(cam_save_path)
    os.mkdir(rgb_save_path)
    os.mkdir(sil_save_path)

for tidx in range(0,render_len):
    tidxf = int(tidx/render_len*frame_len)
    print(tidxf)

    camtxt = np.zeros((4,4))
    camtxt[:3,:3] = test_cameras[tidxf].orientation
    camtxt[:3,3] = camtxt[:3,:3].dot(-test_cameras[tidxf].position[:,None])[:,0]
    camtxt[3,:2] = [test_cameras[tidxf].focal_length, 
                    test_cameras[tidxf].focal_length]
    camtxt[3,2:] = test_cameras[tidxf].principal_point
    
    if save_data:
        np.savetxt( '%s/cam-%05d.txt'%(cam_save_path,tidx), camtxt)
        input_rgb = datasource.load_rgb(datasource.train_ids[tidxf])
        input_sil = 1-datasource.load_sil(datasource.train_ids[tidxf])
        cv2.imwrite('%s/%05d.jpg'%(rgb_save_path,tidx), 255*input_rgb[:,:,::-1])
        cv2.imwrite('%s/%05d.png'%(sil_save_path,tidx), 128*input_sil)

print(' '.join( [str(i) for i in camtxt[-1]] ))
print('%f, %f'%(datasource.near, datasource.far))

import configparser
config = configparser.ConfigParser()
config['data'] = {
'datapath': 'database/DAVIS/JPEGImages/Full-Resolution/',
'dframe': '1',
'init_frame': '0',
'end_frame': '-1',
'can_frame': '-1'}

config['data_0'] = {
'ks': ' '.join( [str(i) for i in camtxt[-1]] ),
'rtk_path':  '%s/cam'%cam_save_path}

with open('configs/%s.config'%(seqname), 'w') as configfile:
    config.write(configfile)


