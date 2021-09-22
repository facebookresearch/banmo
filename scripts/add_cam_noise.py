import cv2
import numpy as np
import pdb
import sys
import glob


cam_dir=sys.argv[1]
seqname=cam_dir.split('/')[-2]
std=0.5

camlist = glob.glob('%s/*.txt'%(cam_dir))
camlist = [i for i in camlist if 'gauss' not in i]
camlist = sorted(camlist)

for idx,path in enumerate(camlist):
    rtk = np.loadtxt(path)
    rtk_mod = rtk.copy()
    # random rot
    rot_rand = np.random.normal(0,std,3)
    rot_rand = cv2.Rodrigues(rot_rand)[0]
    rtk_mod[:3,:3] = rot_rand.dot(rtk_mod[:3,:3])

    path_mod = '%s/gauss-%f-%05d.txt'%(cam_dir.rsplit('/',1)[-2],std,idx)
    np.savetxt(path_mod, rtk_mod)
    print(rtk)
