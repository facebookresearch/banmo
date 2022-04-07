# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
sys.path.insert(0,'third_party')

import subprocess
import imageio
import glob
from utils.io import save_vid
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pdb
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, obj2cam_np
from dataloader import frameloader
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer
import configparser
import matplotlib
cmap = matplotlib.cm.get_cmap('cool')
from utils.io import config_to_dataloader, draw_cams, str_to_frame, \
        extract_data_info
import pytorch3d
import pytorch3d.ops

parser = argparse.ArgumentParser(description='render mesh')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--seqname', default='camel',
                    help='sequence to test')
parser.add_argument('--outpath', default='/data/gengshay/output.gif',
                    help='output path')
parser.add_argument('--overlay', default='no',
                    help='whether to overlay with the input')
parser.add_argument('--cam_type', default='perspective',
                    help='camera model, orthographic or perspective')
parser.add_argument('--vis_bones', dest='vis_bones',action='store_true',
                    help='whether show transparent surface and vis bones')
parser.add_argument('--vis_cam', dest='vis_cam',action='store_true',
                    help='whether show camera trajectory')
parser.add_argument('--vis_traj', dest='vis_traj', action='store_true',
                    help='whether show trajectory of vertices')
parser.add_argument('--append_img', default='no',
                    help='whether append images before the seq')
parser.add_argument('--append_render', default='yes',
                    help='whether append renderings')
parser.add_argument('--nosmooth', dest='smooth', action='store_false',
                    help='whether to smooth vertex colors and positions')
parser.add_argument('--corresp', dest='corresp', action='store_true',
                    help='whether to render correspondence')
parser.add_argument('--floor', dest='floor', action='store_true',
                    help='whether to add floor')
parser.add_argument('--show_dp', dest='show_dp',action='store_true',
                    help='whether to visualizae densepose if available')
parser.add_argument('--freeze', dest='freeze',action='store_true',
                    help='freeze object at frist frame')
parser.add_argument('--rest', dest='rest',action='store_true',
                    help='render rest object shape')
parser.add_argument('--vp', default=0, type=int,
                    help='which viewpoint to render 0,1,2')
parser.add_argument('--gtdir', default='',
                    help='path to gt dir')
parser.add_argument('--test_frames', default='9',
                    help='a list of video index or num of frames, {0,1,2}, 30')
parser.add_argument('--root_frames', default='',
                    help='a list of video index or num of frames, {0,1,2}, 30')
parser.add_argument('--gt_pmat', 
 default='/private/home/gengshany/data/AMA/T_swing/calibration/Camera1.Pmat.cal',
                    help='path to ama projection matrix, evaluation only')
parser.add_argument('--vis_gtmesh', dest='vis_gtmesh', action='store_true',
                    help='whether to visualize ground-truth mesh in eval')
parser.add_argument('--clean', dest='clean', action='store_true',
                    help='whether to use cc to clean up input mesh')
parser.add_argument('--gray_color', dest='gray_color', action='store_true',
                    help='whether to overwrite color with gray')
args = parser.parse_args()

gt_meshes =   [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(args.gtdir)) )]

def main():
    print(args.testdir)
    if args.rest:
        mesh_rest = trimesh.load('%s/mesh-rest.obj'%(args.testdir),process=False)
    # read all the data
    all_anno = []
    all_mesh = []
    all_bone = []
    all_cam = []
    all_fr = []

    # eval dataloader
    opts_dict = {}
    opts_dict['seqname'] = args.seqname
    opts_dict['img_size'] = 512 # dummy value
    opts_dict['rtk_path'] = ''
    evalloader = frameloader.eval_loader(opts_dict)
    data_info = extract_data_info(evalloader)
    idx_render = str_to_frame(args.test_frames, data_info)

    if args.root_frames=='': idx_render_root = idx_render
    else: idx_render_root = str_to_frame(args.root_frames, data_info)

    # get eval frames
    imglist = []
    for dataset in evalloader.dataset.datasets:
        imglist += dataset.imglist[:-1] # excluding the last frame
    rootlist =[imglist[i] for i in idx_render_root]
    imglist = [imglist[i] for i in idx_render]

    seqname_list = []
    ## subsumple frames ##This may cause bug at nvs##
    #if len(imglist)>150:
    #    imglist = imglist[::(len(imglist)//150)]

    rootlist = [rootlist[i] for i in \
                np.linspace(0,len(rootlist)-1,len(imglist),dtype=int)]

    for idx,name in enumerate(imglist):
        rgb_img = cv2.imread(name)
        if args.show_dp:
            # replace with densepose
            name1, name2 = name.rsplit('/',1)
            dppath = '%s/vis-%s'%(name1.replace('JPEGImages', 'Densepose'), name2)
            if os.path.exists(dppath):
                rgb_img = cv2.resize(cv2.imread(dppath), rgb_img.shape[:2][::-1])


        try: sil_img = cv2.imread(name.replace('JPEGImages', 'Annotations').replace('.jpg', '.png'),0)[:,:,None]
        except: sil_img = np.zeros(rgb_img.shape)[:,:,0]
        all_anno.append([rgb_img,sil_img,0,0,name])
        seqname = name.split('/')[-2]
        seqname_list.append(seqname)
        fr = int(name.split('/')[-1].split('.')[-2])
        all_fr.append(fr)
        print('%s/%d'%(seqname, fr))

        if args.append_render=="yes":
            try:
                mesh = trimesh.load('%s/%s-mesh-%05d.obj'%(args.testdir, seqname, fr),process=False)
                if args.clean:
                    # keep the largest mesh
                    mesh = [i for i in mesh.split(only_watertight=False)]
                    mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                    mesh = mesh[-1]

                if args.gray_color:
                    mesh.visual.vertex_colors[:,:3]=128 # necessary for color override

                all_mesh.append(mesh)
               
                name_root = rootlist[idx]
                seqname_root = name_root.split('/')[-2]
                fr_root = int(name_root.split('/')[-1].split('.')[-2])
                cam = np.loadtxt('%s/%s-cam-%05d.txt'%(args.testdir, seqname_root, fr_root))
                all_cam.append(cam)

                bone = trimesh.load('%s/%s-bone-%05d.obj'%(args.testdir, seqname,fr),process=False)
                all_bone.append(bone)
            except: print('no mesh found')
        else:
            # dummy variable
            mesh = trimesh.creation.uv_sphere(radius=1,count=[2, 2])
            all_mesh.append(mesh)


    # process bones, trajectories and cameras
    num_original_verts = []
    num_original_faces = []
    pts_trajs = []
    col_trajs = []
    traj_len = len(all_mesh) #TODO shuld be dependent on the seqname
    pts_num = len(all_mesh[0].vertices)
    traj_num = min(1000, pts_num)
    traj_idx = np.random.choice(pts_num, traj_num)
    scene_scale = np.abs(all_mesh[0].vertices).max()

    for i in range(len(all_mesh)):
        if args.vis_bones:
            all_mesh[i].visual.vertex_colors[:,-1]=254 # necessary for color override
            num_original_verts.append( all_mesh[i].vertices.shape[0])
            num_original_faces.append( all_mesh[i].faces.shape[0]  )  
            try: bone=all_bone[i]
            except: bone=trimesh.Trimesh()
            all_mesh[i] = trimesh.util.concatenate([all_mesh[i], bone])
    
        # change color according to time 
        if args.vis_traj:
            pts_traj = np.zeros((traj_len, traj_num,2,3))
            col_traj = np.zeros((traj_len, traj_num,2,4))
            for j in range(traj_len):
                if i-j-1<0 or seqname_list[j] != seqname_list[i]: continue
                pts_traj[j,:,0] = all_mesh[i-j-1].vertices[traj_idx]
                pts_traj[j,:,1] = all_mesh[i-j].vertices  [traj_idx]
                col_traj[j,:,0] = cmap(float(i-j-1)/traj_len)
                col_traj[j,:,1] = cmap(float(i-j)/traj_len)
            pts_trajs.append(pts_traj)
            col_trajs.append(col_traj)
    
        # change color according to time 
        if args.vis_cam:
            mesh_cam = draw_cams(all_cam, axis=False)
            mesh_cam.export('%s/mesh_cam-%s.obj'%(args.testdir,seqname))

    # read images
    input_size = all_anno[0][0].shape[:2]
    #output_size = input_size
    output_size = (int(input_size[0] * 480/input_size[1]), 480)# 270x480
    frames=[]
    ctrajs=[]
    rndsils=[]
    cd_ave=[] # average chamfer distance
    f001=[] # f@1%
    f002=[]  
    f005=[]  
    if args.append_img=="yes":
        if args.append_render=='yes':
            if args.freeze: napp_fr = 30
            else:                  napp_fr = int(len(all_anno)//5)
            for i in range(napp_fr):
                frames.append(cv2.resize(all_anno[0][0],output_size[::-1])[:,:,::-1])
        else:
            for i in range(len(all_anno)):
                #silframe=cv2.resize((all_anno[i][1]>0).astype(float),output_size[::-1])*255
                imgframe=cv2.resize(all_anno[i][0],output_size[::-1])[:,:,::-1]
                #redframe=(np.asarray([1,0,0])[None,None] * silframe[:,:,None]).astype(np.uint8)
                #imgframe = cv2.addWeighted(imgframe, 1, redframe, 0.5, 0)
                frames.append(imgframe)
                #frames.append(cv2.resize(all_anno[i][1],output_size[::-1])*255) # silhouette
                #frames.append(cv2.resize(all_anno[i][0],output_size[::-1])[:,:,::-1]) # frame
                
                #strx = sorted(glob.glob('%s/*'%datapath))[i]# kp
                #strx = strx.replace('JPEGImages', 'KP')
                #kpimg = cv2.imread('%s/%s'%(strx.rsplit('/',1)[0],strx.rsplit('/',1)[1].replace('.jpg', '_rendered.png')))
                #frames.append(cv2.resize(kpimg,output_size[::-1])[:,:,::-1]) 
                
                #strx = sorted(glob.glob('%s/*'%datapath))[init_frame:end_frame][::dframe][i]# flow
                #strx = strx.replace('JPEGImages', 'FlowBW')
                #flowimg = cv2.imread('%s/vis-%s'%(strx.rsplit('/',1)[0],strx.rsplit('/',1)[1]))
                #frames.append(cv2.resize(flowimg,output_size[::-1])[:,:,::-1]) 

    # process cameras
    theta = 9*np.pi/9
    #theta = 7*np.pi/9
    init_light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    init_light_pose0 =np.asarray([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
    if args.freeze or args.rest:
        size = len(all_mesh)
        #size = 150
    else:
        size = len(all_mesh)
    for i in range(size):
        if args.append_render=='no':break
        # render flow between mesh 1 and 2
        if args.freeze or args.rest:
            print(i)
            refimg, refsil, refkp, refvis, refname = all_anno[0]
            img_size = max(refimg.shape)
            if args.freeze: refmesh = all_mesh[0]
            elif args.rest: refmesh = mesh_rest
            #refmesh.vertices -= refmesh.vertices.mean(0)[None]
            #refmesh.vertices /= 1.2*np.abs(refmesh.vertices).max()
            refcam = all_cam[0].copy()
            rot_turntb = cv2.Rodrigues(np.asarray([0.,i*2*np.pi/size,0.]))[0]
            refcam[:3,:3] = rot_turntb.dot( refcam[:3,:3] ) 
            refcam[:2,3] = 0  # trans xy
            if args.vis_cam:
                refcam[2,3] = 10 # depth
                refcam[3,:2] = 8*img_size/2 # fl
            refcam[3,2] = refimg.shape[1]/2 # px py
            refcam[3,3] = refimg.shape[0]/2 # px py
        else:
            refimg, refsil, refkp, refvis, refname = all_anno[i]
            print('%s'%(refname))
            img_size = max(refimg.shape)
            refmesh = all_mesh[i]
            refcam = all_cam[i]

        # load vertices
        refface = torch.Tensor(refmesh.faces[None]).cuda()
        verts = torch.Tensor(refmesh.vertices[None]).cuda()

        # change viewpoint
        vp_tmat = refcam[:3,3]
        vp_kmat = refcam[3]
        if args.vp==-1:
            # static camera
            #vp_rmat = (refcam[:3,:3].T).dot(all_cam[0][:3,:3])
            vp_rmat = all_cam[0][:3,:3].dot(refcam[:3,:3].T)
#            vp_rmat = cv2.Rodrigues(np.asarray([np.pi/2,0,0]))[0].dot(vp_rmat) # bev
            vp_tmat = all_cam[0][:3,3]
            vp_kmat = all_cam[0][3].copy()
            vp_kmat[2] = vp_kmat[2]/all_anno[0][0].shape[1]*all_anno[i][0].shape[1]
            vp_kmat[3] = vp_kmat[3]/all_anno[0][0].shape[0]*all_anno[i][0].shape[0]
        elif args.vp==-2:
            # canonical camera
            can_vis_rot = cv2.Rodrigues(np.asarray([0,np.pi/3,0]))[0].dot(\
                          cv2.Rodrigues(np.asarray([np.pi, 0,0 ]))[0])
            vp_rmat = can_vis_rot.dot(refcam[:3,:3].T)
            vp_tmat = np.zeros(3)
            vp_tmat[2] = all_cam[0][2,3]
            vp_kmat = all_cam[0][3].copy()
            vp_kmat[2] = vp_kmat[2]/all_anno[0][0].shape[1]*all_anno[i][0].shape[1]
            vp_kmat[3] = vp_kmat[3]/all_anno[0][0].shape[0]*all_anno[i][0].shape[0]
        elif args.vp==1:
            vp_rmat = cv2.Rodrigues(np.asarray([0,np.pi/2,0]))[0]
        elif args.vp==2:
            vp_rmat = cv2.Rodrigues(np.asarray([np.pi/2,0,0]))[0]
        else:
            vp_rmat = cv2.Rodrigues(np.asarray([0.,0,0]))[0]
        refcam_vp = refcam.copy()
        #refcam_vp[:3,:3] = refcam_vp[:3,:3].dot(vp_rmat)
        refcam_vp[:3,:3] = vp_rmat.dot(refcam_vp[:3,:3])
        if args.vp==1 or args.vp==2:
            vmean = verts[0].mean(0).cpu()
            vp_tmat[:2] = (-refcam_vp[:3,:3].dot(vmean))[:2]
        refcam_vp[:3,3]  = vp_tmat
        refcam_vp[3]     = vp_kmat

        # render
        Rmat =  torch.Tensor(refcam_vp[None,:3,:3]).cuda()
        Tmat =  torch.Tensor(refcam_vp[None,:3,3]).cuda()
        ppoint =refcam_vp[3,2:]
        focal = refcam_vp[3,:2]
        verts = obj_to_cam(verts, Rmat, Tmat)
        r = OffscreenRenderer(img_size, img_size)
        colors = refmesh.visual.vertex_colors
        
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]))
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
        colors= np.concatenate([0.6*colors[:,:3].astype(np.uint8), colors[:,3:]],-1)  # avoid overexposure
            
        # project trajectories to image
        if args.vis_traj:
            pts_trajs[i] = obj2cam_np(pts_trajs[i], Rmat, Tmat)

        if args.vis_cam:
            mesh_cam_transformed = mesh_cam.copy()
            mesh_cam_transformed.vertices = obj2cam_np(mesh_cam_transformed.vertices, Rmat, Tmat)

        # compute error if ground-truth is given
        if len(args.gtdir)>0:
            if len(gt_meshes)>0:
                verts_gt = torch.Tensor(gt_meshes[i].vertices[None]).cuda()
                refface_gt=torch.Tensor(gt_meshes[i].faces[None]).cuda()
            else:
                verts_gt = verts
                refface_gt = refface

            #  ama camera coord -> scale -> our camera coord
            if args.gt_pmat!='canonical':
                pmat = np.loadtxt(args.gt_pmat)
                K,R,T,_,_,_,_=cv2.decomposeProjectionMatrix(pmat)
                Rmat_gt = R
                Tmat_gt = T[:3,0]/T[-1,0]
                Tmat_gt = Rmat_gt.dot(-Tmat_gt[...,None])[...,0]
                K = K/K[-1,-1]
                ppoint[0] = K[0,2]
                ppoint[1] = K[1,2]
                focal[0] = K[0,0]
                focal[1] = K[1,1]
            else:
                Rmat_gt = np.eye(3)
                Tmat_gt = np.asarray([0,0,0]) # assuming synthetic obj has depth 3

            # render ground-truth to different viewpoint according to cam prediction
            #Rmat_gt = refcam[:3,:3].T
            #Tmat_gt = -refcam[:3,:3].T.dot(refcam[:3,3:4])[...,0]
            #Rmat_gt = refcam_vp[:3,:3].dot(Rmat_gt)
            #Tmat_gt = refcam_vp[:3,:3].dot(Tmat_gt[...,None])[...,0] + refcam_vp[:3,3]
            # transform gt to camera
            Rmat_gt = torch.Tensor(Rmat_gt).cuda()[None]
            Tmat_gt = torch.Tensor(Tmat_gt).cuda()[None]
            # max length of axis aligned bbox
            bbox_max = float((verts_gt.max(1)[0]-verts_gt.min(1)[0]).max().cpu())
            verts_gt = obj_to_cam(verts_gt, Rmat_gt, Tmat_gt)

            import chamfer3D.dist_chamfer_3D
            import fscore
            chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

            ## use ICP for ours improve resutls
            fitted_scale = verts_gt[...,-1].median() / verts[...,-1].median()
            verts = verts*fitted_scale

            frts = pytorch3d.ops.iterative_closest_point(verts,verts_gt, \
                    estimate_scale=False,max_iterations=100)
            verts = ((frts.RTs.s*verts).matmul(frts.RTs.R)+frts.RTs.T[:,None])

            ## show registered meshes
            #t=trimesh.Trimesh(verts[0].cpu()).export('tmp/0.obj')
            #t=trimesh.Trimesh(verts_gt[0].cpu()).export('tmp/1.obj')
            #pdb.set_trace()
           
            raw_cd,raw_cd_back,_,_ = chamLoss(verts_gt,verts)  # this returns distance squared
            f1,_,_ = fscore.fscore(raw_cd, raw_cd_back,
                           threshold = (bbox_max*0.01)**2)
            f2,_,_ = fscore.fscore(raw_cd, raw_cd_back,
                           threshold = (bbox_max*0.02)**2)
            f5,_,_ = fscore.fscore(raw_cd, raw_cd_back,
                           threshold = (bbox_max*0.05)**2)
            
            # sum 
            raw_cd = np.asarray(raw_cd.cpu()[0])
            raw_cd_back = np.asarray(raw_cd_back.cpu()[0])
            raw_cd = np.sqrt(raw_cd)
            raw_cd_back = np.sqrt(raw_cd_back)
            cd_mean = raw_cd.mean() + raw_cd_back.mean()
            cd_ave.append(cd_mean)
            f001.append(  f1.cpu().numpy())
            f002.append(  f2.cpu().numpy())
            f005.append(  f5.cpu().numpy())
            print('cd:%.2f cm'%(100*cd_mean))

            cm = plt.get_cmap('plasma')
            if args.vis_gtmesh:
                verts = verts_gt
                refface = refface_gt
                colors = cm(raw_cd*5)
            else:
                colors = cm(raw_cd_back*5)
        
        smooth=args.smooth
        if args.freeze:
            tbone = 0
        else:
            tbone = i
        if args.vis_bones:
            mesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:num_original_verts[tbone],:3].cpu()), faces=np.asarray(refface[0,:num_original_faces[tbone]].cpu()),vertex_colors=colors)
            meshr = Mesh.from_trimesh(mesh,smooth=smooth)
            meshr._primitives[0].material.RoughnessFactor=.5
            scene.add_node( Node(mesh=meshr ))
            
            mesh2 = trimesh.Trimesh(vertices=np.asarray(verts[0,num_original_verts[tbone]:,:3].cpu()), faces=np.asarray(refface[0,num_original_faces[tbone]:].cpu()-num_original_verts[tbone]),vertex_colors=colors[num_original_verts[tbone]:])
            if len(mesh2.vertices)>0:
                mesh2=Mesh.from_trimesh(mesh2,smooth=smooth)
                mesh2._primitives[0].material.RoughnessFactor=.5
                scene.add_node( Node(mesh=mesh2))
        else: 
            mesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:,:3].cpu()), faces=np.asarray(refface[0].cpu()),vertex_colors=colors)
            meshr = Mesh.from_trimesh(mesh,smooth=smooth)
            meshr._primitives[0].material.RoughnessFactor=.5
            scene.add_node( Node(mesh=meshr ))

        if args.vis_traj:
            pts = pts_trajs[i].reshape(-1,3)# np.asarray([[-1,-1,1],[1,1,1]])  # 2TxNx3
            colors = col_trajs[i].reshape(-1,4)#np.random.uniform(size=pts.shape)
            m = Mesh([pyrender.Primitive(pts,mode=1,color_0=colors)])
            scene.add_node( Node(mesh=m)) 
            
        if args.vis_cam:
            mesh_cam_transformed=Mesh.from_trimesh(mesh_cam_transformed)
            mesh_cam_transformed._primitives[0].material.RoughnessFactor=1.
            scene.add_node( Node(mesh=mesh_cam_transformed))

        floor_mesh = trimesh.load('./mesh_material/wood.obj',process=False)
        floor_mesh.vertices = np.concatenate([floor_mesh.vertices[:,:1], floor_mesh.vertices[:,2:3], floor_mesh.vertices[:,1:2]],-1 )
        xfloor = 10*mesh.vertices[:,0].min() + (10*mesh.vertices[:,0].max()-10*mesh.vertices[:,0].min())*(floor_mesh.vertices[:,0:1] - floor_mesh.vertices[:,0].min())/(floor_mesh.vertices[:,0].max()-floor_mesh.vertices[:,0].min()) 
        yfloor = floor_mesh.vertices[:,1:2]; yfloor[:] = (mesh.vertices[:,1].max())
        zfloor = 0.5*mesh.vertices[:,2].min() + (10*mesh.vertices[:,2].max()-0.5*mesh.vertices[:,2].min())*(floor_mesh.vertices[:,2:3] - floor_mesh.vertices[:,2].min())/(floor_mesh.vertices[:,2].max()-floor_mesh.vertices[:,2].min())
        floor_mesh.vertices = np.concatenate([xfloor,yfloor,zfloor],-1)
        floor_mesh = trimesh.Trimesh(floor_mesh.vertices, floor_mesh.faces, vertex_colors=255*np.ones((4,4), dtype=np.uint8))
        if args.floor:
            scene.add_node( Node(mesh=Mesh.from_trimesh(floor_mesh))) # overrides the prev. one
       
        if args.cam_type=='perspective': 
            cam = IntrinsicsCamera(
                    focal[0],
                    focal[0],
                    ppoint[0],
                    ppoint[1],
                    znear=1e-3,zfar=1000)
        else:
            cam = pyrender.OrthographicCamera(xmag=1., ymag=1.)
        cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
        cam_node = scene.add(cam, pose=cam_pose)
        light_pose = init_light_pose
        direc_l_node = scene.add(direc_l, pose=light_pose)
        #if args.vis_bones:
        #    color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        #else:
        #    color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        color = color[:refimg.shape[0],:refimg.shape[1],:3]
        rndsil = (depth[:refimg.shape[0],:refimg.shape[1]]>0).astype(int)*100
        if args.overlay=='yes':
            color = cv2.addWeighted(color, 0.5, refimg[:,:,::-1], 0.5, 0)
        prefix = (args.outpath).split('/')[-1].split('.')[0]
        color = color.copy(); color[0,0,:] = 0
        imoutpath = '%s/%s-mrender%03d.jpg'%(args.testdir, prefix,i)
        cv2.imwrite(imoutpath,color[:,:,::-1] )
        color = cv2.resize(color, output_size[::-1])

        frames.append(color)

        # TODO save cams
        cam_scale = output_size[1] / rndsil.shape[1]
        ctraj = torch.cat([Rmat, Tmat[...,None]],-1).cpu().numpy() # 1,3,4
        kmat = np.asarray([focal[0]*cam_scale, 
                           focal[0]*cam_scale, 
                           ppoint[0]*cam_scale, 
                           ppoint[1]*cam_scale])
        ctraj = np.concatenate([ctraj,kmat[None,None,:]],1) # 1,4,4
        ctrajs.append(ctraj[0])
        rndsil = cv2.resize(rndsil.astype(np.int16), output_size[::-1])
        rndsils.append(rndsil)

    if args.gtdir != '':
        cd_ave = np.asarray(cd_ave)
        print('ave chamfer dis: %.1f cm'%(100*cd_ave.mean()))
        print('max chamfer dis: %.1f cm'%(100*np.max(cd_ave)))
        f001 = np.asarray(f001)
        print('ave f-score at d=1%%: %.1f%%'%(100*np.mean(f001)))
        print('min f-score at d=1%%: %.1f%%'%(100*np.min( f001)))
        f002 = np.asarray(f002)
        print('ave f-score at d=2%%: %.1f%%'%(100*np.mean(f002)))
        print('min f-score at d=2%%: %.1f%%'%(100*np.min( f002)))
        f005 = np.asarray(f005)
        print('ave f-score at d=5%%: %.1f%%'%(100*np.mean(f005)))
        print('min f-score at d=5%%: %.1f%%'%(100*np.min( f005)))
    save_vid(args.outpath, frames, suffix='.gif')
    save_vid(args.outpath, frames, suffix='.mp4',upsample_frame=0)


    # save camera trajectory and reference sil
    for idx in range(len(ctrajs)):
        save_path = '%s-ctrajs-%05d.txt'%(args.outpath, idx)
        np.savetxt(save_path, ctrajs[idx])
        save_path = '%s-refsil-%05d.png'%(args.outpath, idx)
        cv2.imwrite(save_path, rndsils[idx])

if __name__ == '__main__':
    main()
