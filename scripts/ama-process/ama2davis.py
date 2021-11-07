"""
python scripts/ama-process/ama2davis.py --path ~/data/AMA/T_swing/
"""
import cv2
import numpy as np
import os
import glob
import argparse
import sys
from shutil import copyfile
sys.path.insert(0,'third_party')
from ext_utils.io import mkdir_p

parser = argparse.ArgumentParser(description='script to render cameras over epochs')
parser.add_argument('--path', default='',
                    help='path to ama seq dir')
args = parser.parse_args()

path = '%s/images/*'%args.path
seqname = args.path.strip('/').split('/')[-1]
outdir = '/private/home/gengshany/data/DAVIS/'
outsil_dir = '%s/Annotations/Full-Resolution/%s'%(outdir, seqname)
outrgb_dir = '%s/JPEGImages/Full-Resolution/%s'%(outdir, seqname)

#TODO delete if exists
mkdir_p(outrgb_dir)
mkdir_p(outsil_dir)


for idx, rgb_path in enumerate(sorted(glob.glob(path))):
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
