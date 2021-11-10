# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.8 (default, Apr 13 2021, 19:58:26)
# [GCC 7.3.0]
# Embedded file name: /private/home/gengshany/code/vid2shape/nnutils/cse.py
# Compiled at: 2021-11-09 23:43:08
# Size of source mod 2**32: 4044 bytes
import cv2, pdb, os, sys, numpy as np, torch
import torch.nn as nn
import torchvision
curr_dir = os.path.abspath(os.getcwd())
sys.path.insert(0, curr_dir)
detbase = '../detectron2/'
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
        device = img.device
        img = img * 255
        img = torch.flip(img, [1])
        pdb.set_trace()
        #dps = []
        #feats = []
        #for i in range(bs):
        #    img_sub = img[i].permute(1, 2, 0).cpu().numpy()
        #    msk_sub = msk[i].cpu().numpy()
        #    dp, _, feat, _, _ = run_cse((self.net), (self.embedder), (self.mesh_vertex_embeddings),
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

        img = torch.stack([(x - self.net.pixel_mean) / self.net.pixel_std\
                            for x in img])
        pred_boxes = torch.Tensor([[0,0,img.shape[2], img.shape[3]]]).cuda()
        pred_boxes = Boxes(pred_boxes)            
        features = self.net.backbone(img)
        features = [features[f] for f in self.net.roi_heads.in_features]
        features = [self.net.roi_heads.decoder(features)]
        features_dp = self.net.roi_heads.densepose_pooler(features, [pred_boxes]*img.shape[0]).detach()
        densepose_head_outputs = self.net.roi_heads.densepose_head(features_dp)
        densepose_predictor_outputs = self.net.roi_heads.densepose_predictor(densepose_head_outputs)
        feats = densepose_predictor_outputs.embedding

        dps = []
        for i in range(bs):
            pdb.set_trace()
            assign_mat = squared_euclidean_distance_matrix(feats[i].view(16,-1).T, 
                           self.mesh_vertex_embeddings[self.mesh_name])
            dp = assign_mat.argmin(dim=1).view(112,112)
            dps.append(dp)
        dps = torch.stack(dps,0)
        return feats, dps
