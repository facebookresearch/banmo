# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'./')

import numpy as np
import trimesh
import torch
import cv2
import pdb
from scipy.spatial.transform import Rotation as R

from nnutils.geom_utils import obj_to_cam, pinhole_cam, render_color, render_flow
from ext_utils.flowlib import flow_to_image
from ext_utils.util_flow import write_pfm
from utils.io import mkdir_p
import soft_renderer as sr
import argparse
parser = argparse.ArgumentParser(description='render data')
parser.add_argument('--outdir', default='eagle',
                    help='output dir')
parser.add_argument('--model', default='eagle',
                    help='model to render, {eagle, hands}')
parser.add_argument('--rot_axis', default='y',
                    help='axis to rotate around')
parser.add_argument('--nframes', default=3,type=int,
                    help='number of frames to render')
parser.add_argument('--alpha', default=1.,type=float,
                    help='0-1, percentage of a full cycle')
parser.add_argument('--init_a', default=0.25,type=float,
                    help='0-1, percentage of a full cycle for initial pose')
parser.add_argument('--xspeed', default=0,type=float,
                    help='times speed up')
parser.add_argument('--focal', default=2,type=float,
                    help='focal length')
parser.add_argument('--d_obj', default=3,type=float,
                    help='object depth')
parser.add_argument('--can_rand', dest='can_rand',action='store_true',
                    help='ranomize canonical space')
parser.add_argument('--img_size', default=512,type=int,
                    help='image size')
parser.add_argument('--render_flow', dest='render_flow',action='store_true',
                    help='render flow')

args = parser.parse_args()
## io
img_size = args.img_size
bgcolor = None
#bgcolor = np.asarray([0,0,0])
d_obj = args.d_obj
filedir='database'

rot_rand = torch.Tensor(R.random().as_matrix()).cuda()

overts_list = []
for i in range(args.nframes):
    if args.model=='eagle':
        mesh = sr.Mesh.from_obj('database/eagle/Eagle-original_%06d.obj'%int(i*args.xspeed), load_texture=True, texture_res=5, texture_type='surface')
    elif args.model=='hands':
        mesh = sr.Mesh.from_obj('database/hands/hands_%06d.obj'%int(1+i*args.xspeed), load_texture=True, texture_res=100, texture_type='surface')

    overts = mesh.vertices
    if i==0:
        center = overts.mean(1)[:,None]
        scale = max((overts - center)[0].abs().max(0)[0])

    overts -= center
    overts *= 1.0 / float(scale)
    overts[:,:,1]*= -1  # aligh with camera coordinate

    # random rot
    if args.can_rand:
        overts[0] = overts[0].matmul(rot_rand.T)

    overts_list.append(overts)
colors=mesh.textures
faces = mesh.faces

mkdir_p( '%s/DAVIS/JPEGImages/Full-Resolution/%s/'   %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Annotations/Full-Resolution/%s/'  %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Cameras/Full-Resolution/%s/'       %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Meshes/Full-Resolution/%s/'       %(filedir,args.outdir))


# soft renderer
renderer = sr.SoftRenderer(image_size=img_size, sigma_val=1e-12, 
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
               #light_intensity_ambient=0.,light_intensity_directionals=1., light_directions=[-1.,-0.5,1.])

    
verts_ndc_list = []
for i in range(0,args.nframes):
    verts = overts_list[i]

    # set cameras
    #rotx = np.random.rand()
    if args.rot_axis=='x':
        rotx = args.init_a*6.28+args.alpha*6.28*i/args.nframes
    else:
        rotx=0.
#    if i==0: rotx=0.
    if args.rot_axis=='y':
        roty = args.init_a*6.28+args.alpha*6.28*i/args.nframes
    else:
        roty = 0
    rotz = 0.
    Rmat = cv2.Rodrigues(np.asarray([rotx, roty, rotz]))[0]
    Rmat = torch.Tensor(Rmat).cuda()
    # random rot
    if args.can_rand:
        Rmat = Rmat.matmul(rot_rand.T)
    Tmat = torch.Tensor([0,0,d_obj]                                        ).cuda()
    K =    torch.Tensor([args.focal,args.focal,0,0]  ).cuda() 
    Kimg = torch.Tensor([args.focal*img_size/2.,args.focal*img_size/2.,img_size/2.,img_size/2.]  ).cuda() 

    # add RTK: [R_3x3|T_3x1]
    #          [fx,fy,px,py], to the ndc space
    rtk = np.zeros((4,4))
    rtk[:3,:3] = Rmat.cpu().numpy()
    rtk[:3, 3] = Tmat.cpu().numpy()
    rtk[3, :]  = Kimg   .cpu().numpy()

    # obj-cam transform 
    verts = obj_to_cam(verts, Rmat, Tmat)
    mesh_cam = trimesh.Trimesh(vertices=verts[0].cpu().numpy(), 
                               faces=faces[0].cpu().numpy())
    trimesh.repair.fix_inversion(mesh_cam)
    
    # pespective projection
    verts = pinhole_cam(verts, K)
    verts_ndc_list.append(verts.clone())
    
    # render sil+rgb
    rendered = render_color(renderer, verts, faces, colors, texture_type='surface')

    rendered_img = rendered[0,:3].permute(1,2,0).cpu().numpy()*255
    rendered_sil = rendered[0,-1].cpu().numpy()*128
    if bgcolor is None:
        bgcolor = 255-rendered_img[rendered_sil.astype(bool)].mean(0)
    rendered_img[~rendered_sil.astype(bool)]=bgcolor[None]
    cv2.imwrite('%s/DAVIS/JPEGImages/Full-Resolution/%s/%05d.jpg'     %(filedir,args.outdir,i),rendered_img[:,:,::-1])
    cv2.imwrite('%s/DAVIS/Annotations/Full-Resolution/%s/%05d.png'    %(filedir,args.outdir,i),rendered_sil)
    np.savetxt('%s/DAVIS/Cameras/Full-Resolution/%s/%05d.txt'         %(filedir,args.outdir,i),rtk)
    mesh_cam.export('%s/DAVIS/Meshes/Full-Resolution/%s/%05d.obj'      %(filedir,args.outdir,i))
    print(i)

if args.render_flow:
    for dframe in [1,2,4,8,16,32]:
        print('dframe: %d'%(dframe))
        flobw_outdir = '%s/DAVIS/FlowBW_%d/Full-Resolution/%s/'%(filedir,dframe,args.outdir)
        flofw_outdir = '%s/DAVIS/FlowFW_%d/Full-Resolution/%s/'%(filedir,dframe,args.outdir)
        mkdir_p(flofw_outdir)
        mkdir_p(flobw_outdir)
        # render flow
        occ = -np.ones((img_size, img_size)).astype(np.float32)
        for i in range(dframe,args.nframes):
            verts_ndc = verts_ndc_list[i-dframe]
            verts_ndc_n = verts_ndc_list[i]
            flow_fw = render_flow(renderer, verts_ndc,   faces, verts_ndc_n)
            flow_bw = render_flow(renderer, verts_ndc_n, faces, verts_ndc)
            # to pixels
            flow_fw = flow_fw*(img_size-1)/2
            flow_bw = flow_bw*(img_size-1)/2
            flow_fw = flow_fw.cpu().numpy()[0]
            flow_bw = flow_bw.cpu().numpy()[0]
            
            write_pfm(  '%s/flo-%05d.pfm'%(flofw_outdir,i-dframe),flow_fw)
            write_pfm(  '%s/flo-%05d.pfm'%(flobw_outdir,i),  flow_bw)
            write_pfm(  '%s/occ-%05d.pfm'%(flofw_outdir,i-dframe),occ)
            write_pfm(  '%s/occ-%05d.pfm'%(flobw_outdir,i),  occ)
            cv2.imwrite('%s/col-%05d.jpg'%(flofw_outdir,i-dframe),flow_to_image(flow_fw)[:,:,::-1])
            cv2.imwrite('%s/col-%05d.jpg'%(flobw_outdir,i),       flow_to_image(flow_bw)[:,:,::-1])
