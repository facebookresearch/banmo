# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
python img2lines.py --seqname xx
"""
from absl import flags, app
import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'./')
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
from utils.io import mkdir_p
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
from utils.io import config_to_dataloader
from torch.utils.data import DataLoader
from nnutils.geom_utils import tensor2array
opts = flags.FLAGS
                    
def dict2pix(dict_array, idy):
    dict_px = {}
    dict_px['img'] =                   dict_array['img'][...,idy,:]
    dict_px['mask'] =                 dict_array['mask'][...,idy,:]
    dict_px['vis2d'] =               dict_array['vis2d'][...,idy,:]
    dict_px['flow'] =                 dict_array['flow'][...,idy,:]
    dict_px['occ'] =                   dict_array['occ'][...,idy,:]
    dict_px['dp'] =                     dict_array['dp'][...,idy,:]
    dict_px['dp_feat_rsmp'] = dict_array['dp_feat_rsmp'][...,idy,:]
    return dict_px

def dict2rtk(dict_array):
    dict_out = {}
    dict_out['rtk'] =                   dict_array['rtk']
    dict_out['kaug'] =                   dict_array['kaug']
    return dict_out
                
def main(_):
    seqname=opts.seqname
    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()
    impaths = data_info['impath']
    data_offset = data_info['offset']

    opts_dict = {}
    opts_dict['seqname'] = opts.seqname
    opts_dict['img_size'] = opts.img_size
    opts_dict['rtk_path'] = opts.rtk_path
    opts_dict['batch_size'] = 1
    opts_dict['ngpu'] = 1
    opts_dict['preload'] = False
    opts_dict['dframe'] = [1,2,4,8,16,32]

    dataset = config_to_dataloader(opts_dict,is_eval=True)
    #dataset = config_to_dataloader(opts_dict,is_eval=False)
    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=0, drop_last=False, 
         pin_memory=True, shuffle=False)    
    for dat in dataset.dataset.datasets:
        dat.spec_dt = 1
    
    #TODO
    #overwrite=False
    overwrite=True

    # hardcoded path 
    base_path = 'database/DAVIS/Pixels/Full-Resolution/'
    for i, batch in enumerate(dataset):
        frameid = batch['frameid']
        dataid = batch['dataid']
        dt = frameid[0,1] - frameid[0,0]
        frameid = frameid + data_offset[dataid[0,0].long()]
        if dt<0: continue # only save forward pair (bachward pair is equivalent)
        impath = impaths[frameid.long()[0,0]]
        seqname_sub = impath.split('/')[-2]
        frameid_sub = impath.split('/')[-1].split('.')[0]

        save_dir = '%s/%s'%(base_path, seqname_sub)
        save_dir_t = '%s/%d_%s'%(save_dir, dt, frameid_sub)
        print(save_dir_t)
        if (not overwrite) and os.path.exists(save_dir_t):
            continue
        mkdir_p(save_dir_t)

        dict_array = tensor2array(batch)
        # save each pixel: 00_00000/0000.npy # t,h
        dict_rtk = dict2rtk(dict_array)
        save_path_rtk = '%s/rtk.npy'%(save_dir_t)
        np.save(save_path_rtk, dict_rtk)

        for idy in range(opts.img_size):
            save_path = '%s/%04d.npy'%(save_dir_t, idy)
            dict_px = dict2pix(dict_array, idy)
            np.save(save_path, dict_px)



if __name__ == '__main__':
    app.run(main)
