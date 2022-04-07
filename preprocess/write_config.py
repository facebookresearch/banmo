# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import configparser
import cv2
import glob
import pdb
import sys

seqname_pre=sys.argv[1]
ishuman=sys.argv[2] # 'y/n'
silroot='database/DAVIS/Annotations/Full-Resolution/'

config = configparser.ConfigParser()
config['data'] = {
'dframe': '1',
'init_frame': '0',
'end_frame': '-1',
'can_frame': '-1'}

seqname_all = sorted(glob.glob('%s/%s[0-9][0-9][0-9]'%(silroot, seqname_pre)))

total_vid = 0
for i,seqname in enumerate(seqname_all):
    seqname = seqname.split('/')[-1]
    img = cv2.imread('%s/%s/00000.png'%(silroot,seqname),0)
    if img is None:continue
    num_fr = len(glob.glob('%s/%s/*.png'%(silroot,seqname)))
    if num_fr < 16:continue

    fl = max(img.shape)
    px = img.shape[1]//2
    py = img.shape[0]//2
    camtxt = [fl,fl,px,py]
    config['data_%d'%total_vid] = {
    'ishuman': ishuman,
    'ks': ' '.join( [str(i) for i in camtxt] ),
    'datapath': 'database/DAVIS/JPEGImages/Full-Resolution/%s/'%seqname,
    }
    total_vid += 1

with open('configs/%s.config'%(seqname_pre), 'w') as configfile:
    config.write(configfile)

