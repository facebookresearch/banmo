# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# python scripts/eval_root.py cam-files/adult7-b25/ cam-files/adult-masked-cam/ 1000
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
from nnutils.geom_utils import rot_angle, align_sim3
       
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


def main():
    rootlist_a = load_root(root_a_dir, cap_frame)
    rootlist_b = load_root(root_b_dir, cap_frame)

    # align
    rootlist_b = align_sim3(rootlist_a, rootlist_b)

    # construct camera mesh
    mesh_a = draw_cams(rootlist_a, color='gray')
    mesh_b = draw_cams(rootlist_b)
    mesh = trimesh.util.concatenate([mesh_a, mesh_b])
    mesh.export('0.obj')
    
# python ... path to camera folder
# will draw a trajectory of camera locations
if __name__ == '__main__':
    main()
