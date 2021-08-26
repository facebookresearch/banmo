import cv2
import glob
import numpy as np
import pdb
import os
import shutil

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
import torch
import torch.nn.functional as F
import torchvision
import sys
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0,curr_dir)
detbase='/private/home/gengshany/code/detectron2/'
sys.path.insert(0,'%s/projects/PointRend/'%detbase)
sys.path.insert(0,'%s/projects/DensePose/'%detbase)
sys.path.insert(0,'third_party/ext_utils')
from utils.cselib import create_cse, run_cse
from utils.io import save_vid, visObj
from util_flow import write_pfm
import point_rend
  

seqname=sys.argv[1]
odir='/private/home/gengshany/data/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
dpdir='%s/Densepose/Full-Resolution/%s'%(odir,seqname)
if os.path.exists(dpdir): shutil.rmtree(dpdir)
os.mkdir(dpdir)

config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml'%(detbase)
weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl'
predictor_dp, embedder, mesh_vertex_embeddings = create_cse(config_path,
                                                            weight_path)
   
counter=0 
frames = []
for i,path in enumerate(sorted(glob.glob('%s/*'%imgdir))):
    print(path)
    img = cv2.imread(path)
    msk = cv2.imread(path.replace('JPEGImages', 'Annotations').replace('.jpg', '.png'),0)
    h,w = img.shape[:2]
    
    # recompte mask    
    msk = msk/np.sort(np.unique(msk))[1]
    occluder = msk==255
    msk[occluder] = 0
   
    # resize to some empirical size
    if h>w: h_rszd,w_rszd = 1333, 1333*w//h 
    else:   h_rszd,w_rszd = 1333*h//w, 1333 
    img_rszd = cv2.resize(img,(w_rszd,h_rszd))
    msk_rszd = cv2.resize(msk,(w_rszd,h_rszd))

    # densepose
    clst_verts, image_bgr1 = run_cse(predictor_dp, embedder, mesh_vertex_embeddings, 
                                                    img_rszd, msk_rszd, 
                                                    mesh_name='sheep_5004')
    
    clst_verts = cv2.resize(clst_verts, (w,h), interpolation=cv2.INTER_NEAREST)

    # assume max 10k/200 max
    clst_verts = (clst_verts/50.).astype(np.float32)

    write_pfm(  '%s/%05d.pfm'%(dpdir,counter), clst_verts)
    
    # vis
    #v = Visualizer(img_rszd, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    #outvis = visObj()
    #outvis.image_height = h 
    #outvis.image_width = w 
    #outvis._fields = {}
    #outvis._fields["pred_boxes"] = np.asarray([[0,0,h,w,1.]])
    #vis = v.draw_instance_predictions(outvis)
    #vis = vis.get_image()
    vis=img_rszd
    alpha_mask = 0.8*(msk_rszd>0)[...,None]
    mask_result = vis*(1-alpha_mask) + image_bgr1 * alpha_mask
    cv2.imwrite('%s/vis-%05d.jpg'%(dpdir,counter), mask_result)
    
    counter+=1
    frames.append(mask_result[:,:,::-1])    
  
save_vid('%s/vis'%dpdir, frames, suffix='.mp4')
save_vid('%s/vis'%dpdir, frames, suffix='.gif')