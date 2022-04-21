# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
python scripts/ama-process/ama2davis.py --path ./database/T_swing/
"""
import pdb
import cv2
import numpy as np
import os
import glob
import argparse
import sys
from shutil import copyfile
sys.path.insert(0,'')
from utils.io import mkdir_p

parser = argparse.ArgumentParser(description='script to render cameras over epochs')
parser.add_argument('--path', default='',
                    help='path to ama seq dir')
args = parser.parse_args()

path = '%s/images/*'%args.path
seqname = args.path.strip('/').split('/')[-1]
outdir = './database/DAVIS/'

vid_idx = 0
for rgb_path in sorted(glob.glob(path)):
    vid_idx_tmp = int(rgb_path.split('/')[-1].split('_')[0][5:])
    if vid_idx_tmp != vid_idx:
        idx=0
        vid_idx = vid_idx_tmp

    outsil_dir = '%s/Annotations/Full-Resolution/%s%d'%(outdir, seqname,vid_idx)
    outrgb_dir = '%s/JPEGImages/Full-Resolution/%s%d'%(outdir,  seqname,vid_idx)
    #TODO delete if exists
    mkdir_p(outrgb_dir)
    mkdir_p(outsil_dir)
    
    sil_path = rgb_path.replace('images', 'silhouettes').replace('Image','Silhouette')
    outsil_path = '%s/%05d.png'%(outsil_dir, idx)
    sil = cv2.imread(sil_path,0)
    sil = (sil>0).astype(np.uint8)

    # remove extra sils
    nb_components, output, stats, centroids = \
    cv2.connectedComponentsWithStats(sil, connectivity=8)
    if nb_components>1:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        sil = output == max_label
    sil = (sil>0).astype(np.uint8)*128

    cv2.imwrite(outsil_path, sil)

    outrgb_path = '%s/%05d.jpg'%(outrgb_dir, idx)
    img = cv2.imread(rgb_path)
    cv2.imwrite(outrgb_path, img)
    
    print(outrgb_path)
    print(outsil_path)
    idx = idx+1
