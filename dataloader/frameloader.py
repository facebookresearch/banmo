"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app
import random

import torch
from torch.utils.data import Dataset

import pdb
import glob
from torch.utils.data import DataLoader
import configparser
from utils.io import config_to_dataloader

opts = flags.FLAGS
    
def _init_fn(worker_id):
    np.random.seed(1003)
    random.seed(1003)

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts_dict, shuffle=True):
    num_workers = opts_dict['n_data_workers'] * opts_dict['batch_size']
    num_workers = min(num_workers, 8)
    #num_workers = 0
    print('# workers: %d'%num_workers)
    print('# pairs: %d'%opts_dict['batch_size'])

    data_inuse = config_to_dataloader(opts_dict)

    sampler = torch.utils.data.distributed.DistributedSampler(
    data_inuse,
    num_replicas=opts_dict['ngpu'],
    rank=opts_dict['local_rank'],
    shuffle=True
    )

    data_inuse = DataLoader(data_inuse,
         batch_size= opts_dict['batch_size'], num_workers=num_workers, 
         drop_last=True, worker_init_fn=_init_fn, pin_memory=True,
         sampler=sampler)
    return data_inuse

#----------- Eval Data Loader ----------#
#----------------------------------#
def eval_loader(opts_dict):
    num_workers = 0
   
    dataset = config_to_dataloader(opts_dict,is_eval=True)
    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=num_workers, drop_last=False, pin_memory=True, shuffle=False)
    return dataset
