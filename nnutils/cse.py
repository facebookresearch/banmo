# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import cv2, pdb, os, sys, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, curr_dir)
detbase = './third_party/detectron2/'
sys.path.insert(0, '%s/projects/DensePose/' % detbase)
from detectron2.structures import Boxes
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes
from densepose import add_densepose_config
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
from utils.cselib import create_cse, run_cse

class CSENet(nn.Module):

    def __init__(self, ishuman):
        super(CSENet, self).__init__()
        if ishuman:
            config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml' % detbase
            weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl'
            self.mesh_name = 'smpl_27554'
        else:
            config_path = '%s/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml' % detbase
            weight_path = 'https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl'
            self.mesh_name = 'sheep_5004'
        self.net, self.embedder, self.mesh_vertex_embeddings = create_cse(config_path, weight_path)

    def forward(self, img, msk):
        bs = img.shape[0]
        h = img.shape[2]
        device = img.device
        img = img * 255
        img = torch.flip(img, [1])
    
        pad = h
        img = F.pad(img, (pad, pad, pad, pad))
        msk = F.pad(msk, (pad, pad, pad, pad))
        img = F.interpolate(img, (384, 384),mode='bilinear') 
        msk = F.interpolate(msk[:,None], (384, 384),mode='nearest')[:,0]
  
        bboxes = []
        for i in range(bs):
            indices = torch.where(msk[i]>0); 
            xid = indices[1]; yid = indices[0]
            bbox = [xid.min(), yid.min(),
                    xid.max(), yid.max()]
            bbox = torch.Tensor([bbox]).to(device)
            bbox = Boxes(bbox)            
            bboxes.append(bbox)
        #dps = []
        #feats = []
        #for i in range(bs):
        #    img_sub = img[i].permute(1, 2, 0).cpu().numpy()
        #    msk_sub = msk[i].cpu().numpy()
        #    # put into a bigger image: out size 112/512 
        #    dp, img_bgr, feat, feat_norm, bbox = run_cse((self.net), (self.embedder), (self.mesh_vertex_embeddings),
        #      img_sub,
        #      msk_sub,
        #      mesh_name=(self.mesh_name))
        #    pdb.set_trace()
        #    dp = torch.Tensor(dp).to(device)
        #    feat = torch.Tensor(feat).to(device)
        #    dps.append(dp)
        #    feats.append(feat)
        #dps = torch.stack(dps, 0)
        #feats = torch.stack(feats, 0)
        #pdb.set_trace()

        self.net.eval()
        with torch.no_grad():
            img = torch.stack([(x - self.net.pixel_mean) / self.net.pixel_std\
                                for x in img])
            features = self.net.backbone(img)
            features = [features[f] for f in self.net.roi_heads.in_features]
            features = [self.net.roi_heads.decoder(features)]
            features_dp = self.net.roi_heads.densepose_pooler(features, bboxes).detach()

        densepose_head_outputs = self.net.roi_heads.densepose_head(features_dp)
        densepose_predictor_outputs = self.net.roi_heads.densepose_predictor(densepose_head_outputs)
        feats = densepose_predictor_outputs.embedding # (xxx,112,112)

        with torch.no_grad():
            dps = []
            for i in range(bs):
                assign_mat = squared_euclidean_distance_matrix(feats[i].view(16,-1).T, 
                               self.mesh_vertex_embeddings[self.mesh_name])
                dp = assign_mat.argmin(dim=1).view(112,112)
                dps.append(dp)
            dps = torch.stack(dps,0)
        return feats, dps
