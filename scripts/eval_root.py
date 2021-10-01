import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)
import pdb
import glob
import numpy as np
import torch
import cv2
import soft_renderer as sr
import argparse
import trimesh
import configparser
from utils.io import config_to_dataloader, draw_cams, load_root
from nnutils.geom_utils import rot_angle
from scipy.spatial.transform import Rotation as R
       
root_a_dir=sys.argv[1]
root_b_dir=sys.argv[2]
cap_frame=int(sys.argv[3])

def umeyama_alignment(x, y, with_scale=False):
    """
    https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

def align_se3(rootlist_a, rootlist_b):
#    ta = np.matmul(-np.transpose(rootlist_a[:,:3,:3],[0,2,1]), 
#                                 rootlist_a[:,:3,3:4])
#    ta = ta[...,0].T
#    tb = np.matmul(-np.transpose(rootlist_b[:,:3,:3],[0,2,1]), 
#                                 rootlist_b[:,:3,3:4])
#    tb = tb[...,0].T
#    dso3,dtrn,dscale=umeyama_alignment(tb, ta,with_scale=False)
#    
#    dscale = np.linalg.norm(rootlist_a[0,:3,3],2,-1) /\
#             np.linalg.norm(rootlist_b[0,:3,3],2,-1)
#    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3.T[None])
#    rootlist_b[:,:3,3:4] = rootlist_b[:,:3,3:4] - \
#            np.matmul(rootlist_b[:,:3,:3], dtrn[None,:,None]) 

    dso3 = np.matmul(np.transpose(rootlist_b[:,:3,:3],(0,2,1)),
                        rootlist_a[:,:3,:3])
    dso3 = R.from_matrix(dso3).mean().as_matrix()
    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3[None])
    dscale = np.linalg.norm(rootlist_a[:,:3,3],2,-1).mean() /\
            np.linalg.norm(rootlist_b[:,:3,3],2,-1).mean()
    rootlist_b[:,:3,3] = rootlist_b[:,:3,3] * dscale

    so3_err = np.matmul(rootlist_a[:,:3,:3], 
            np.transpose(rootlist_b[:,:3,:3],[0,2,1]))
    so3_err = rot_angle(torch.Tensor(so3_err))
    so3_err = so3_err / np.pi*180
    so3_err_max = so3_err.max()
    so3_err_mean = so3_err.mean()
    print(so3_err)
    print('max  so3 error (deg): %.1f'%(so3_err_max))
    print('mean so3 error (deg): %.1f'%(so3_err_mean))

    return rootlist_b

def main():
    rootlist_a = load_root(root_a_dir, cap_frame)
    rootlist_b = load_root(root_b_dir, cap_frame)

    # align
    rootlist_b = align_se3(rootlist_a, rootlist_b)

    # construct camera mesh
    mesh_a = draw_cams(rootlist_a, color='gray')
    mesh_b = draw_cams(rootlist_b)
    mesh = trimesh.util.concatenate([mesh_a, mesh_b])
    mesh.export('0.obj')
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
