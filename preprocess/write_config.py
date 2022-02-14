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
for i,seqname in enumerate(seqname_all):
    seqname = seqname.split('/')[-1]
    img = cv2.imread('%s/%s/00000.png'%(silroot,seqname),0)
    fl = max(img.shape)
    px = img.shape[1]//2
    py = img.shape[0]//2
    camtxt = [fl,fl,px,py]
    config['data_%d'%i] = {
    'ishuman': ishuman,
    'ks': ' '.join( [str(i) for i in camtxt] ),
    'datapath': 'database/DAVIS/JPEGImages/Full-Resolution/%s/'%seqname,
    }

with open('configs/%s.config'%(seqname_pre), 'w') as configfile:
    config.write(configfile)


