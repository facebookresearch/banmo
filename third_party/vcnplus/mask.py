import cv2
import glob
import numpy as np
import pdb
import os

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")


detbase='/private/home/gengshany/code/detectron2/'
import sys
seqname=sys.argv[1]
datadir='/private/home/gengshany/data/tmp/%s/images/'%seqname
odir='/private/home/gengshany/data/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
import shutil
if os.path.exists(imgdir): shutil.rmtree(imgdir)
if os.path.exists(maskdir): shutil.rmtree(maskdir)
os.mkdir(imgdir)
os.mkdir(maskdir)

import sys
sys.path.insert(0,'%s/projects/PointRend/'%detbase)
import point_rend
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.1
cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

predictor = DefaultPredictor(cfg)
   
counter=0 
for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    print(path)
    img = cv2.imread(path)
    #if i==0:
    shape = img.shape[:2]
    mask = np.zeros(shape)
    
    #imgt = cv2.resize(img,(shape[1]//2,shape[0]//2))
    imgt = img
    segs = predictor(imgt)['instances'].to('cpu')
    #import torch.nn.functional as F
    #segs.pred_masks = F.interpolate(segs.pred_masks[None].float(),shape,mode='bilinear').bool()[0]
    
    for it,ins_cls in enumerate(segs.pred_classes):
        print(ins_cls)
        #if ins_cls ==15: # cat
        if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):
            mask += np.asarray(segs.pred_masks[it])

    ## for fashion
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #mask = np.logical_and(mask.astype(bool), gray<250)
    
    if (mask.sum())<1000: continue

    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    #max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    #mask = output == max_label

    mask = mask.astype(bool).astype(int)*128
    mask = np.concatenate([mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]],-1)
    mask[:,:,:2] = 0

    cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
    cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask)
    
    ## vis
    #v = Visualizer(img, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    #vis = v.draw_instance_predictions(segs)
    #mask_result = np.concatenate([vis.get_image(), mask],1)
    #cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), mask_result)
    

    counter+=1

