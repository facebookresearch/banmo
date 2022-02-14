# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import pickle
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import pdb
import trimesh

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes as create_boxes

import sys
try:
    sys.path.insert(0,'./third_party/detectron2//projects/DensePose/')
    from densepose import add_densepose_config
    from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
    from densepose.data.build import get_class_to_mesh_name_mapping
    from densepose.modeling import build_densepose_embedder
    from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding
    from densepose.vis.base import Boxes, Image, MatrixVisualizer
except:
    sys.path.insert(0,'./third_party/detectron2_old//projects/DensePose/')
    from densepose import add_densepose_config
    from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
    from densepose.data.build import get_class_to_mesh_name_mapping
    from densepose.modeling import build_densepose_embedder
    from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding
    from densepose.vis.base import Boxes, Image, MatrixVisualizer

# load model
def create_cse(config_fpath, weights_fpath):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.WEIGHTS = weights_fpath
    model = build_model(cfg)  # returns a torch.nn.Module
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
    
    embedder = build_densepose_embedder(cfg)
    class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
    mesh_vertex_embeddings = {
        mesh_name: embedder(mesh_name).cuda()
        for mesh_name in class_to_mesh_name.values()
        if embedder.has_embeddings(mesh_name)
    }
    return model, embedder, mesh_vertex_embeddings


def run_cse(model, embedder, mesh_vertex_embeddings, image, mask, mesh_name='smpl_27554'):
    h,w,_=image.shape
    
    # resize
    max_size=1333
    if h>w:
        h_rszd, w_rszd = max_size, max_size*w//h
    else:
        h_rszd, w_rszd = max_size*h//w, max_size
    image = cv2.resize(image, (w_rszd, h_rszd))
    mask = cv2.resize(mask.astype(float), (w_rszd, h_rszd)).astype(np.uint8)

    # pad
    h_pad = (1+h_rszd//32)*32
    w_pad = (1+w_rszd//32)*32
    image_tmp = np.zeros((h_pad,w_pad,3)).astype(np.uint8)
    mask_tmp =   np.zeros((h_pad,w_pad)).astype(np.uint8)
    image_tmp[:h_rszd,:w_rszd] = image
    mask_tmp[:h_rszd,:w_rszd] = mask
    image = image_tmp
    mask = mask_tmp
    image_raw = image.copy()
   
    # preprocess image and box 
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( int((xid.max()-xid.min())*1.//2), int((yid.max()-yid.min())*1.//2))
    bbox = [center[0]-length[0], center[1]-length[1],length[0]*2, length[1]*2]
    bboxw = bbox[2]
    bboxh = bbox[3]
    bbox = [max(0,bbox[0]), 
            max(0,bbox[1]),
            min(w_pad, bbox[0]+bbox[2]),
            min(h_pad, bbox[1]+bbox[3])]
    
    image=torch.Tensor(image).cuda().permute(2,0,1)[None]
    image = torch.stack([(x - model.pixel_mean) / model.pixel_std for x in image])
    pred_boxes = torch.Tensor([bbox]).cuda()
    pred_boxes = create_boxes(pred_boxes)
    
    # inference
    model.eval()
    with torch.no_grad():
        features = model.backbone(image)
        features = [features[f] for f in model.roi_heads.in_features]
        features = [model.roi_heads.decoder(features)]
        features_dp = model.roi_heads.densepose_pooler(features, [pred_boxes])
        densepose_head_outputs = model.roi_heads.densepose_head(features_dp)
        densepose_predictor_outputs = model.roi_heads.densepose_predictor(densepose_head_outputs)
        coarse_segm_resized = densepose_predictor_outputs.coarse_segm[0]
        embedding_resized = densepose_predictor_outputs.embedding[0]
    
    # use input mask
    x, y, xx, yy= bbox
    mask_box = mask[y:yy, x:xx]
    mask_box = torch.Tensor(mask_box).cuda()[None,None]
    mask_box = F.interpolate(mask_box, coarse_segm_resized.shape[1:3], mode='bilinear')[0,0]>0

    # find closest match (in the cropped/resized coordinate)
    clst_verts_pad = torch.zeros(h_pad, w_pad).long().cuda()
    clst_verts_box = torch.zeros(mask_box.shape, dtype=torch.long).cuda()
    all_embeddings = embedding_resized[:, mask_box].t()
    assign_mat = squared_euclidean_distance_matrix(all_embeddings, mesh_vertex_embeddings[mesh_name])
    clst_verts_box[mask_box] = assign_mat.argmin(dim=1)

    clst_verts_box = F.interpolate(clst_verts_box[None,None].float(), (yy-y,xx-x),mode='nearest')[0,0].long()
    clst_verts_pad[y:yy,x:xx] = clst_verts_box
    
    # output embedding
    embedding = embedding_resized # size does not matter for a image code
    embedding = embedding * mask_box.float()[None]
    
    # embedding norm
    embedding_norm = embedding.norm(2,0)
    embedding_norm_pad = torch.zeros(h_rszd, w_rszd).cuda()
    embedding_norm_box = F.interpolate(embedding_norm[None,None], (yy-y,xx-x),mode='bilinear')[0,0]
    embedding_norm_pad[y:yy,x:xx] = embedding_norm_box
    embedding_norm = embedding_norm_pad[:h_rszd, :w_rszd]
    embedding_norm = F.interpolate(embedding_norm[None,None], (h,w),mode='bilinear')[0][0]
    
    embedding = embedding.cpu().numpy()
    embedding_norm = embedding_norm.cpu().numpy()

    # visualization
    embed_map = get_xyz_vertex_embedding(mesh_name, 'cuda')
    vis = (embed_map[clst_verts_pad].clip(0, 1) * 255.0).cpu().numpy()
    mask_visualizer = MatrixVisualizer(
                inplace=False, cmap=cv2.COLORMAP_JET, val_scale=1.0, alpha=0.7
            )
    image_bgr = mask_visualizer.visualize(image_raw, mask, vis, [0,0,w_pad,h_pad])

    image_bgr = image_bgr[:h_rszd,:w_rszd]
    image_bgr = cv2.resize(image_bgr, (w,h))
    clst_verts =clst_verts_pad[:h_rszd, :w_rszd]
    clst_verts = F.interpolate(clst_verts[None,None].float(), (h,w),mode='nearest')[0,0].long()
    clst_verts =clst_verts.cpu().numpy()
    return clst_verts, image_bgr, embedding, embedding_norm, bbox
