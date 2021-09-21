import pdb
import pickle
import sys
sys.path.insert(0,'third_party')
sys.path.insert(0,'')
import glob
import numpy as np
import torch
import cv2
from ext_utils.util_flow import readPFM
from ext_utils.geometry import R_2vect
from ext_utils.io import mkdir_p
import glob
import trimesh
import pytorch3d.ops
import torch.nn.functional as F
import configparser
from nnutils.geom_utils import K2mat, obj_to_cam, pinhole_cam, render_color
import soft_renderer as sr
from utils.io import save_vid

with open('mesh_material/sheep_5004.pkl', 'rb') as f:
    dp = pickle.load(f)
    dp_verts = dp['vertices']
    dp_faces = dp['faces']
    dp_faces = torch.Tensor(dp_faces)
    dp_verts = torch.Tensor(dp_verts)
    # convert to -1,1
    dp_verts -= dp_verts.mean(0)[None]
    dp_verts /= dp_verts.abs().max() 
    
    # visualize
    dp_vis = dp_verts
    dp_vmin =dp_vis.min(0)[0][None]
    dp_vis = dp_vis - dp_vmin
    dp_vmax =dp_vis.max(0)[0][None]
    dp_vis = dp_vis / dp_vmax

    gt_mesh = trimesh.Trimesh(dp_verts, dp_faces, vertex_colors = dp_vis)
    gt_mesh.export('0.obj')


dp_dir=sys.argv[1]
seqname=dp_dir.split('/')[-2]
config_path='configs/%s.config'%seqname
config = configparser.RawConfigParser()
config.read(config_path)
Kvec=torch.Tensor([int(float(i)) for i in \
                 config.get('data_0', 'ks').split(' ')])
Kmat=K2mat(Kvec)[0]

ims=[]
frames=[]
near=[]
far=[]
tvecs=[]
rvecs=[]
norm_th = 15
#TODO a dirty workaround to remove feat-*.pfm
for idx, dp_path in enumerate(glob.glob('%s/0*.pfm'%dp_dir)): 
    # read dp 
    dp = readPFM(dp_path)[0]
    h,w = dp.shape
    norm_path = '%s/norm-%05d.pfm'%(dp_path.rsplit('/',1)[-2],idx)
    dp_norm = readPFM(norm_path)[0]
    img_path = '%s/vis-%05d.jpg'%(dp_path.rsplit('/',1)[-2],idx)
    im = cv2.imread(img_path)
    im = cv2.resize(im, (w,h))

    dp= (dp *50).astype(np.int32)
    dpmask = np.logical_and(dp>0, dp_norm>norm_th) # use norm as uncertainty
    x0,y0  =np.meshgrid(range(w),range(h))
    p2d = np.concatenate([x0[...,None], y0[...,None]],-1).astype(np.float32)
    
    dp_verts_mapped = dp_verts[dp]
    dp_color_mapped = dp_vis[dp]
    
    dp_verts_mapped = dp_verts_mapped[dpmask]
    dp_color_mapped = dp_color_mapped[dpmask]
    p2d = p2d[dpmask]
    
    # from pnp
    retval,rvec,tvec,reproj_err = cv2.solvePnPGeneric(dp_verts_mapped[:,None].numpy(),
                 p2d[:,None], Kmat.numpy(), 0, 
                 flags=cv2.SOLVEPNP_DLS)
    rvec=rvec[0]
    tvec=tvec[0]
    #retval,rvec,tvec,inliers = cv2.solvePnPRansac(dp_verts_mapped[:,None].numpy(),
    #             p2d[:,None], Kmat.numpy(), 0, 
    #             rvec=rvec,
    #             tvec=tvec,
    #             useExtrinsicGuess=True,
    #             flags=cv2.SOLVEPNP_ITERATIVE)
    #print(len(inliers))
    
    rvecs.append(rvec)
    tvecs.append(tvec)
    ims.append(im)

#tvecs=np.asarray(tvecs)
#tmed=np.median(tvecs,0)
#tvecs[:]=tmed

for idx, dp_path in enumerate(glob.glob('%s/0*.pfm'%dp_dir)):
    rvec=rvecs[idx]
    tvec=tvecs[idx]
    im=ims[idx]
    
    rotmat_np = cv2.Rodrigues(rvec)[0]
    rotmat = torch.Tensor(rotmat_np)
    tmat_np = tvec[:,0]
    tmat = torch.Tensor(tmat_np)
    
    #dp_verts_rot = dp_verts_mapped.matmul(rotmat.T)
    #mesh = trimesh.Trimesh(dp_verts_rot, vertex_colors = dp_color_mapped)
    #mesh.export('1.obj')
        
   
    # render 
    verts = obj_to_cam(dp_verts, rotmat, tmat)
    verts = pinhole_cam(verts, torch.Tensor([1,1,0,0]))
    near.append(verts[...,-1].min())
    far.append( verts[...,-1].max())
    renderer = sr.SoftRenderer(image_size=h, sigma_val=1e-12, 
                   camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                   light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
    rendered = render_color(renderer,    verts[None].cuda(), 
                                      dp_faces[None].cuda(), 
                                        dp_vis[None].cuda(), 
                                      texture_type='vertex')
    rendered_img = rendered[0,:3].permute(1,2,0).cpu().numpy()*255
    rendered_img = cv2.resize(rendered_img, (w,h))
    rendered_img = np.concatenate([im, rendered_img[:,:,::-1]],1)
    cv2.imwrite('tmp/%05d.jpg'%idx,rendered_img)
    print('saved %d'%idx)
    frames.append(rendered_img[:,:,::-1])
    
    # save cams
    rtk = np.eye(4)
    rtk[:3,:3] = rotmat_np
    rtk[:3,3] = tmat_np
    rtk[3] = Kvec
    cam_path = dp_path.replace('Densepose', 'Cameras').replace('.pfm', '.txt')
    mkdir_p(cam_path.rsplit('/',1)[-2])
    np.savetxt(cam_path, rtk)

save_vid("tmp/dp-%s"%(seqname), frames, suffix='.gif',upsample_frame=150., is_flow=False)
save_vid("tmp/dp-%s"%(seqname), frames, suffix='.mp4',upsample_frame=150., is_flow=False)

near=max(0,np.min(near))
far= np.max(far)
#tmed = np.linalg.norm(tmed)
#near = tmed-1.5
#far = tmed+1.5
config['data_0']['near_far'] = '%f, %f'%(near, far)
with open('configs/%s.config'%(seqname), 'w') as configfile:
    config.write(configfile)
