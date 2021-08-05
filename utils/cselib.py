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
from densepose import add_densepose_config
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
from densepose.data.build import get_class_to_mesh_name_mapping
from densepose.modeling import build_densepose_embedder
from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding

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


def run_cse(model, embedder, mesh_vertex_embeddings, images, mask, mesh_name='smpl_27554'):
    from detectron2.structures import Boxes
    # resize
    ooh,oow,_=images.shape
    max_size=1333
    if ooh>oow:
        oh, ow = max_size, max_size*oow//ooh
    else:
        oh, ow = max_size*ooh//oow, max_size
        
    images = cv2.resize(images, (ow, oh))
    mask = cv2.resize(mask.astype(float), (ow, oh)).astype(np.uint8)

    oh,ow,_=images.shape
    h = (1+oh//32)*32
    w = (1+ow//32)*32
    imagesn = np.zeros((h,w,3)).astype(np.uint8)
    maskn = np.zeros((h,w)).astype(np.uint8)
    imagesn[:oh,:ow] = images
    maskn[:oh,:ow] = mask
    images = imagesn
    mask = maskn


    image_raw = images.copy()
    
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( int((xid.max()-xid.min())*1.//2), int((yid.max()-yid.min())*1.//2))
    bbox = [center[0]-length[0], center[1]-length[1],length[0]*2, length[1]*2]
    bboxw = bbox[2]
    bboxh = bbox[3]
    bbox = [max(0,bbox[0]), 
            max(0,bbox[1]),
            min(w, bbox[0]+bbox[2]),
            min(h, bbox[1]+bbox[3])]
    #cv2.imwrite('./2.png', images[bbox[1]:bbox[3],bbox[0]:bbox[2]])
    
    images=torch.Tensor(images).cuda().permute(2,0,1)[None]
    images = torch.stack([(x - model.pixel_mean) / model.pixel_std for x in images])
    pred_boxes = torch.Tensor([bbox]).cuda()
    pred_boxes = Boxes(pred_boxes)
    
    # train/test
    model.eval()
    features = model.backbone(images)
    features = [features[f] for f in model.roi_heads.in_features]
    features = [model.roi_heads.decoder(features)]
    features_dp = model.roi_heads.densepose_pooler(features, [pred_boxes])
    densepose_head_outputs = model.roi_heads.densepose_head(features_dp)
    densepose_predictor_outputs = model.roi_heads.densepose_predictor(densepose_head_outputs)
    S = densepose_predictor_outputs.coarse_segm[0]
    E = densepose_predictor_outputs.embedding[0]
    N = S.size(0)
    
    embedding_resized = E
    coarse_segm_resized = S
    #mask = coarse_segm_resized.argmax(0) > 0 
    # use input mask
    mask = mask[bbox[1]:bbox[3], 
                bbox[0]:bbox[2]]
    mask = torch.Tensor(mask).cuda()[None,None]
    mask = F.interpolate(mask, S.shape[1:3], mode='bilinear')[0,0]>0
    #cv2.imwrite('./1.png', np.asarray(mask.float().cpu()))
        

    closest_vertices = torch.zeros(mask.shape, dtype=torch.long).cuda()
    all_embeddings = embedding_resized[:, mask].t()
    #size_chunk = 10_000  # Chunking to avoid possible OOM
    #edm = []
    #for chunk in range((len(all_embeddings) - 1) // size_chunk + 1):
    #    chunk_embeddings = all_embeddings[size_chunk * chunk : size_chunk * (chunk + 1)]
    #    edm.append(
    #        squared_euclidean_distance_matrix(
    #            chunk_embeddings, mesh_vertex_embeddings[mesh_name]
    #        ).argmin(dim=1)
    #    )
    #closest_vertices[mask] = torch.cat(edm)
    assign_mat = squared_euclidean_distance_matrix(all_embeddings, mesh_vertex_embeddings[mesh_name])
    closest_vertices[mask] = assign_mat.argmin(dim=1)
    closest_px = assign_mat.argmin(dim=0)

    embed_map = get_xyz_vertex_embedding(mesh_name, 'cuda')
    vis = (embed_map[closest_vertices].clip(0, 1) * 255.0).cpu().numpy()
    mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)
    from densepose.vis.base import Boxes, Image, MatrixVisualizer
    mask_visualizer = MatrixVisualizer(
                inplace=False, cmap=cv2.COLORMAP_JET, val_scale=1.0, alpha=0.7
            )
   
    x, y, xx, yy= pred_boxes.tensor[0].tolist()
    w=xx-x; h=yy-y
    image_bgr = mask_visualizer.visualize(image_raw, mask_numpy, vis, [x,y,w,h])

#    closest_vertices = F.interpolate(closest_vertices[None,None].float(), (bboxh, bboxw))[0,0].long()
    image_bgr = image_bgr[:oh,:ow]
    image_bgr = cv2.resize(image_bgr, (oow,ooh))
    return closest_vertices, closest_px, image_bgr, bbox, [bboxw,bboxh]
