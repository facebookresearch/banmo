import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'./')

import numpy as np
import trimesh
import torch
import cv2
import pdb
from scipy.spatial.transform import Rotation as R

from ext_utils.io import mkdir_p
import argparse
parser = argparse.ArgumentParser(description='render camera trajectories')
parser.add_argument('--outdir', default='tmp/traj',
                    help='output dir')
parser.add_argument('--nframes', default=90,type=int,
                    help='number of frames to render')
parser.add_argument('--alpha', default=0.5,type=float,
                    help='0-1, percentage of a full cycle')
parser.add_argument('--init_a', default=0.5,type=float,
                    help='0-1, percentage of a full cycle for initial pose')
parser.add_argument('--focal', default=2,type=float,
                    help='focal length')
parser.add_argument('--d_obj', default=3,type=float,
                    help='object depth')
parser.add_argument('--can_rand', dest='can_rand',action='store_true',
                    help='ranomize canonical space')
parser.add_argument('--img_size', default=512,type=int,
                    help='image size')

args = parser.parse_args()
## io
img_size = args.img_size
d_obj = args.d_obj
mkdir_p(args.outdir)

rot_rand = torch.Tensor(R.random().as_matrix()).cuda()

# to be compatible with other seqs
base_rmat = torch.eye(3).cuda()
base_rmat[0,0] = -1
base_rmat[1,1] = -1

for i in range(0,args.nframes):
    # set cameras
    #rotx = np.random.rand()
    rotx=0.
    if i==0: rotx=0.
    roty = args.init_a*6.28+args.alpha*6.28*i/args.nframes
    rotz = 0.
    Rmat = cv2.Rodrigues(np.asarray([rotx, roty, rotz]))[0]
    Rmat = torch.Tensor(Rmat).cuda()
    # random rot
    if args.can_rand:
        Rmat = Rmat.matmul(rot_rand.T)
    Rmat = Rmat.matmul(base_rmat)
    Tmat = torch.Tensor([0,0,d_obj]                                        ).cuda()
    K =    torch.Tensor([args.focal,args.focal,0,0]  ).cuda() 
    Kimg = torch.Tensor([args.focal*img_size/2.,args.focal*img_size/2.,img_size/2.,img_size/2.]  ).cuda() 

    # add RTK: [R_3x3|T_3x1]
    #          [fx,fy,px,py], to the ndc space
    rtk = np.zeros((4,4))
    rtk[:3,:3] = Rmat.cpu().numpy()
    rtk[:3, 3] = Tmat.cpu().numpy()
    rtk[3, :]  = Kimg   .cpu().numpy()
    np.savetxt('%s/%05d.txt' %(args.outdir,i),rtk)
