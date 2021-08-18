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
    np.random.seed()
    random.seed()
    

#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    num_workers = opts.n_data_workers * opts.batch_size
    #num_workers = 0
    print('# workers: %d'%num_workers)
    print('# pairs: %d'%opts.batch_size)
   
    opts_dict = {}
    opts_dict['seqname'] = opts.seqname
    opts_dict['img_size'] = opts.img_size
    opts_dict['batch_size'] = opts.batch_size
    opts_dict['ngpu'] = opts.ngpu
    data_inuse = config_to_dataloader(opts_dict)

    sampler = torch.utils.data.distributed.DistributedSampler(
    data_inuse,
    num_replicas=opts.ngpu,
    rank=opts.local_rank,
    shuffle=True
    )

    data_inuse = DataLoader(data_inuse,
         batch_size= opts.batch_size, num_workers=num_workers, drop_last=True, worker_init_fn=_init_fn, pin_memory=True,sampler=sampler)
    return data_inuse

#----------- Eval Data Loader ----------#
#----------------------------------#
def eval_loader(opts):
    num_workers = 0
    print('# pairs: %d'%opts.batch_size)
   
    opts_dict = {}
    opts_dict['seqname'] = opts.seqname
    opts_dict['img_size'] = opts.img_size
    dataset = config_to_dataloader(opts_dict,is_eval=True)
    
    #dataset = get_config_info(opts, config, 'data', 0, is_eval=True)
    #dataset = torch.utils.data.ConcatDataset(dataset)
    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=num_workers, drop_last=False, pin_memory=True, shuffle=False)
    return dataset
