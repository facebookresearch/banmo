# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# python scripts/add_cam_noise.py cam-files/cse-ama/ 30
import cv2
import numpy as np
import pdb
import sys
import glob
import os

cam_dir=sys.argv[1]
std_rot=float(sys.argv[2]) # deg
seqname=cam_dir.split('/')[-2]
std=np.pi/180*std_rot 

odir='%s-gauss-%d'%(cam_dir.rsplit('/',1)[-2],std_rot)
os.makedirs(odir, exist_ok=True)

camlist = glob.glob('%s/*.txt'%(cam_dir))
camlist = sorted(camlist)

for idx,path in enumerate(camlist):
    rtk = np.loadtxt(path)
    rtk_mod = rtk.copy()
    # random rot
    rot_rand = np.random.normal(0,std,3)
    rot_rand = cv2.Rodrigues(rot_rand)[0]
    rtk_mod[:3,:3] = rot_rand.dot(rtk_mod[:3,:3])
    rtk_mod[:2,3] = 0
    rtk_mod[2,3] = 3

    fid = path.rsplit('/',1)[1]
    path_mod = '%s/%s'%(odir,fid)
    np.savetxt(path_mod, rtk_mod)
    print(rtk)
