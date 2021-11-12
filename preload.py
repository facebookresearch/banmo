"""
python preload.py --seqname xx
"""
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
from ext_utils.io import mkdir_p
from ext_utils.util_flow import write_pfm
from ext_utils.flowlib import cat_imgflo 
from utils.io import config_to_dataloader
from torch.utils.data import DataLoader
from nnutils.geom_utils import tensor2array
opts = flags.FLAGS
                
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
    overwrite=True

    # hardcoded path 
    base_path = 'database/DAVIS/Preload/Full-Resolution/'
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
        mkdir_p(save_dir)
        save_path_frame = '%s/%d_%s.npy'%(save_dir, dt, frameid_sub)
        print(save_path_frame)

        dict_array = tensor2array(batch)

        if (not overwrite) and os.path.exists(save_path_frame):
            pass
        else:
            np.save(save_path_frame, dict_array)



if __name__ == '__main__':
    app.run(main)
