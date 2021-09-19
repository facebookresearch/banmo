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
detbase='../detectron2/'
sys.path.insert(0,'%s/projects/PointRend/'%detbase)
sys.path.insert(0,'%s/projects/DensePose/'%detbase)
sys.path.insert(0,'third_party/ext_utils')
from utils.cselib import create_cse, run_cse
from utils.io import save_vid
from util_flow import write_pfm
import point_rend
        

seqname=sys.argv[1]
datadir='tmp/%s/images/'%seqname
odir='database/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
dpdir='%s/Densepose/Full-Resolution/%s'%(odir,seqname)
if os.path.exists(imgdir): shutil.rmtree(imgdir)
if os.path.exists(maskdir): shutil.rmtree(maskdir)
if os.path.exists(dpdir): shutil.rmtree(dpdir)
os.mkdir(imgdir)
os.mkdir(maskdir)
os.mkdir(dpdir)


cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.9
predictor = DefaultPredictor(cfg)

config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml'%(detbase)
weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl'
predictor_dp, embedder, mesh_vertex_embeddings = create_cse(config_path,
                                                            weight_path)
   
counter=0 
frames = []
for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    print(path)
    img = cv2.imread(path)
    h,w = img.shape[:2]
   
    # resize to some empirical size
    if h>w: h_rszd,w_rszd = 1333, 1333*w//h 
    else:   h_rszd,w_rszd = 1333*h//w, 1333 
    img_rszd = cv2.resize(img,(w_rszd,h_rszd))

    # pointrend
    outputs = predictor(img_rszd)
    outputs = outputs['instances'].to('cpu')
    mask_rszd = np.zeros((h_rszd,w_rszd))
    for it,ins_cls in enumerate(outputs.pred_classes):
        print(ins_cls)
        #if ins_cls ==15: # cat
        #if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):
        if ins_cls >= 14 and ins_cls <= 23:
            mask_rszd += np.asarray(outputs.pred_masks[it])

    nb_components, output, stats, centroids = \
    cv2.connectedComponentsWithStats(mask_rszd.astype(np.uint8), connectivity=8)
    if nb_components>1:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        mask_rszd = output == max_label

    mask_rszd = mask_rszd.astype(bool).astype(int)
    if (mask_rszd.sum())<1000: continue

    # densepose
    clst_verts, image_bgr1, embedding, bbox = run_cse(predictor_dp, embedder, 
                                                    mesh_vertex_embeddings, 
                                                    img_rszd, mask_rszd, 
                                                    mesh_name='sheep_5004')
    
    mask_rszd = np.concatenate([mask_rszd[:,:,np.newaxis]* 128,
                                np.zeros((h_rszd, w_rszd, 1)),
                                np.zeros((h_rszd, w_rszd, 1))],-1)
     
    mask = cv2.resize(mask_rszd,(w,h))
    clst_verts = cv2.resize(clst_verts, (w,h), interpolation=cv2.INTER_NEAREST)

    # assume max 10k/200 max
    clst_verts = (clst_verts/50.).astype(np.float32)

    cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
    cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask)
    write_pfm(  '%s/%05d.pfm'%(dpdir,counter), clst_verts)
    
    # vis
    v = Visualizer(img_rszd, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    outputs.remove('pred_masks')
    vis = v.draw_instance_predictions(outputs)
    vis = vis.get_image()
    alpha_mask = 0.8*(mask_rszd.sum(-1)>0)[...,None]
    mask_result = vis*(1-alpha_mask) + image_bgr1 * alpha_mask
    cv2.imwrite('%s/vis-%05d.jpg'%(maskdir,counter), mask_result)
    
    counter+=1
    frames.append(mask_result[:,:,::-1])    
  
save_vid('%s/vis'%maskdir, frames, suffix='.mp4')
save_vid('%s/vis'%maskdir, frames, suffix='.gif')
