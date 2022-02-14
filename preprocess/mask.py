# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


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

try:
    detbase='./third_party/detectron2/'
    sys.path.insert(0,'%s/projects/PointRend/'%detbase)
    import point_rend
except:
    detbase='./third_party/detectron2_old/'
    sys.path.insert(0,'%s/projects/PointRend/'%detbase)
    import point_rend

sys.path.insert(0,'third_party/ext_utils')
from utils.io import save_vid
from util_flow import write_pfm
        

seqname=sys.argv[1]
ishuman=sys.argv[2] # 'y/n'
datadir='tmp/%s/images/'%seqname
odir='database/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
#if os.path.exists(imgdir): shutil.rmtree(imgdir)
#if os.path.exists(maskdir): shutil.rmtree(maskdir)
#os.mkdir(imgdir)
#os.mkdir(maskdir)


cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.9
predictor = DefaultPredictor(cfg)

counter=0 
frames = []
for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    print(path)
    img = cv2.imread(path)
    h,w = img.shape[:2]
   
    # store at most 1080p videos
    scale = np.sqrt(1920*1080/(h*w))
    if scale<1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)) )
    h,w = img.shape[:2]

    # resize to some empirical size
    if h>w: h_rszd,w_rszd = 1333, 1333*w//h 
    else:   h_rszd,w_rszd = 1333*h//w, 1333 
    img_rszd = cv2.resize(img,(w_rszd,h_rszd))

    # pad borders to make sure detection works when obj is out-of-frame
    pad=100
    img_rszd = cv2.copyMakeBorder(img_rszd,pad,pad,pad,pad,cv2.BORDER_REPLICATE)

    # pointrend
    outputs = predictor(img_rszd)
    outputs = outputs['instances'].to('cpu')
    mask_rszd = np.zeros((h_rszd+pad*2,w_rszd+pad*2))
    for it,ins_cls in enumerate(outputs.pred_classes):
        print(ins_cls)
        #if ins_cls ==15: # cat
        #if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):
        if ishuman=='y':
            if ins_cls ==0:
                mask_rszd += np.asarray(outputs.pred_masks[it])
        else:
            if ins_cls >= 14 and ins_cls <= 23:
                mask_rszd += np.asarray(outputs.pred_masks[it])

    nb_components, output, stats, centroids = \
    cv2.connectedComponentsWithStats(mask_rszd.astype(np.uint8), connectivity=8)
    if nb_components>1:
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
        mask_rszd = output == max_label
    
    mask_rszd = mask_rszd.astype(bool).astype(int)
    if (mask_rszd.sum())<1000: continue
    mask_rszd                 = mask_rszd  [pad:-pad,pad:-pad]
    img_rszd                   = img_rszd  [pad:-pad,pad:-pad]
    outputs.pred_masks=outputs.pred_masks[:,pad:-pad,pad:-pad]
    outputs.pred_boxes.tensor[:,:2] -= pad
    mask_rszd = np.concatenate([mask_rszd[:,:,np.newaxis]* 128,
                                np.zeros((h_rszd, w_rszd, 1)),
                                np.zeros((h_rszd, w_rszd, 1))],-1)
    mask = cv2.resize(mask_rszd,(w,h))

    cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
    cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask)
    
    # vis
    v = Visualizer(img_rszd, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    #outputs.remove('pred_masks')
    vis = v.draw_instance_predictions(outputs)
    vis = vis.get_image()
    cv2.imwrite('%s/vis-%05d.jpg'%(maskdir,counter), vis)
    
    counter+=1
    frames.append(vis[:,:,::-1])    
  
save_vid('%s/vis'%maskdir, frames, suffix='.mp4')
save_vid('%s/vis'%maskdir, frames, suffix='.gif')
