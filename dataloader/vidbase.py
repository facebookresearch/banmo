"""
Base data loading class.

Should output:
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import numpy as np

from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import cv2
import time
from scipy.ndimage import binary_erosion

from ext_utils.util_flow import readPFM
from ext_utils import image as image_utils
from ext_utils.flowlib import warp_flow

def read_json(filepath, mask):
    import json
    with open(filepath) as f: 
        maxscore=-1
        for pid in  json.load(f)['people']:
            ppose = np.asarray(pid['pose_keypoints_2d']).reshape((-1,3))
            pocc = cv2.remap(mask.astype(int), ppose[:,0].astype(np.float32),ppose[:,1].astype(np.float32),interpolation=cv2.INTER_NEAREST)
            pscore = pocc.sum()
            if pscore>maxscore: maxscore = pscore; maxpose = ppose
    return maxpose

# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, kp, pose data loader
    '''

    def __init__(self, opts, filter_key=None):
        # Child class should define/load:
        # self.kp_perm
        # self.img_dir
        # self.anno
        # self.anno_sfm
        self.opts = opts
        self.img_size = opts['img_size']
        self.random_geo = 1.
        self.filter_key = filter_key
        self.flip=0
        self.crop_factor = 1.2
        self.load_pair = True
    
    def mirror_image(self, img, mask):
        if np.random.rand(1) > 0.5:
            # Need copy bc torch collate doesnt like neg strides
            img_flip = img[:, ::-1, :].copy()
            mask_flip = mask[:, ::-1].copy()
            
            return img_flip, mask_flip
        else:
            return img, mask

    def rot_augment(self,x0):
        if np.random.rand(1) > self.random_geo:
            # centralize
            A0 = np.asarray([[1,0,x0.shape[0]/2.],
                            [0,1,x0.shape[1]/2.],
                            [0,0,1]])
            # rotate
            A = cv2.Rodrigues(np.asarray([0.,0.,2.]))[0]
            A = A0.dot(A).dot(np.linalg.inv(A0))
            A = A.T
        else:
            A = np.eye(3)
        return A

    def geo_augment(self,x0):
        if self.flip:
            # mirror
            A = np.asarray([[-1,0,x0.shape[0]-1],
                            [0,1,0             ],
                            [0,0,1]]).T
        else:
            A = np.eye(3)

        A = A.dot(self.rot_augment(x0))
        return A

    def __len__(self):
        return self.num_imgs
    
    def read_raw(self, im0idx, flowfw):
        img_path = self.imglist[im0idx]
        img = cv2.imread(img_path)[:,:,::-1] / 255.0
        shape = img.shape
        if len(shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

        mask = cv2.imread(self.masklist[im0idx],0)
        mask = mask/np.sort(np.unique(mask))[1]
        occluder = mask==255
        mask[occluder] = 0
        if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
            mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            mask = binary_erosion(mask,iterations=2)
        mask = np.expand_dims(mask, 2)

        # flow
        if flowfw:
            flowpath = self.flowfwlist[im0idx]
        else:
            flowpath = self.flowbwlist[im0idx]
        try:
            flow = readPFM(flowpath)[0]
            occ = readPFM(flowpath.replace('flo-', 'occ-'))[0]
            h,w,_ = mask.shape
            oh,ow=flow.shape[:2]
            factor_h = h/oh
            factor_w = w/ow
            flow = cv2.resize(flow, (w,h))
            occ  = cv2.resize(occ, (w,h))
            flow[...,0] *= factor_w
            flow[...,1] *= factor_h
        except:
            print('warning: loading empty flow from %s'%(flowpath))
            flow = np.zeros_like(img)
            occ = np.zeros_like(mask)
        occ[occluder] = 0

        try:
            dp = readPFM(self.dplist[im0idx])[0]
        except:
            print('error loading densepose surface')
            dp = np.zeros_like(occ)
        try:
            dp_feat = readPFM(self.featlist[im0idx])[0]
            dp_bbox =  np.loadtxt(self.bboxlist[im0idx])
        except:
            print('error loading densepose feature')
            dp_feat =  np.zeros((16*112,112))
            dp_bbox =  np.zeros((4))
        dp= (dp *50).astype(np.int32)
        dp_feat = dp_feat.reshape((16,112,112)).copy()

        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        try:
            rtk_path = self.rtklist[im0idx]
            rtk = np.loadtxt(rtk_path)
        except:
            print('warning: loading empty camera')
            print(rtk_path)
            rtk = np.zeros((4,4))
            rtk[:3,:3] = np.eye(3)
            rtk[:3, 3] = np.asarray([0,0,10])
            rtk[3, :]  = np.asarray([512,512,256,256]) 

        # create mask for visible vs unkonwn
        vis2d = np.ones_like(mask)
        
        # crop the image according to mask
        kaug, hp0, A, B= self.compute_crop_params(mask)
        x0 = hp0[:,:,0].astype(np.float32)
        y0 = hp0[:,:,1].astype(np.float32)
        img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR)
        mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
        flow = cv2.remap(flow,x0,y0,interpolation=cv2.INTER_LINEAR)
        occ = cv2.remap(occ,x0,y0,interpolation=cv2.INTER_LINEAR)
        dp   =cv2.remap(dp,   x0,y0,interpolation=cv2.INTER_NEAREST)
        vis2d=cv2.remap(vis2d.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        mask = (mask>0).astype(float)

        rt_dict = {}
        rt_dict['img']   = img     
        rt_dict['mask']  = mask  
        rt_dict['flow']  = flow  
        rt_dict['occ']   = occ   
        rt_dict['dp']    = dp    
        rt_dict['vis2d'] = vis2d 
        rt_dict['dp_feat'] = dp_feat
        rt_dict['dp_bbox'] = dp_bbox
        rt_dict['rtk'] = rtk
        return rt_dict, kaug, hp0, A,B
    
    def compute_crop_params(self, mask):
        indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
        center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
        length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
        length = (int(self.crop_factor*length[0]), int(self.crop_factor*length[1]))

        maxw=self.img_size;maxh=self.img_size
        orisize = (2*length[0], 2*length[1])
        alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
        
        # intrinsics induced by augmentation: augmented to to original img
        # correct cx,cy at clip space (not tx, ty)
        if self.flip==0:
            pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
        else:
            pps  = np.asarray([-float( center[0] - length[0] ), float( center[1] - length[1]  )])
        kaug = np.asarray([alp[0], alp[1], pps[0], pps[1]])

        # geometric augmentation for img, mask, flow, occ
        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        A = self.geo_augment(x0)
        B = np.asarray([[alp[0],0,(center[0]-length[0])],
                        [0,alp[1],(center[1]-length[1])],
                        [0,0,1]]).T
        hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
        hp0 = np.dot(hp0,A).dot(B)                   # image coord
        return kaug, hp0, A,B

    def flow_process(self,flow, flown, occ, occn, hp0, hp1, A,B,Ap,Bp):
        maxw=self.img_size;maxh=self.img_size
        # augmenta flow
        hp1c = np.concatenate([flow[:,:,:2] + hp0[:,:,:2], np.ones_like(hp0[:,:,:1])],-1) # image coord
        hp1c = hp1c.dot(np.linalg.inv(Ap.dot(Bp)))   # screen coord
        flow[:,:,:2] = hp1c[:,:,:2] - np.stack(np.meshgrid(range(maxw),range(maxh)),-1)
        
        hp0c = np.concatenate([flown[:,:,:2] +hp1[:,:,:2], np.ones_like(hp0[:,:,:1])],-1) # image coord
        hp0c = hp0c.dot(np.linalg.inv(A.dot(B)))   # screen coord
        flown[:,:,:2] =hp0c[:,:,:2] - np.stack(np.meshgrid(range(maxw),range(maxh)),-1)

        #fb check
        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord

        dis = warp_flow(hp0 + flown, flow[:,:,:2]) - hp0
        dis = np.linalg.norm(dis[:,:,:2],2,-1) * 0.1
        occ[occ!=0] = dis[occ!=0]

        disn = warp_flow(hp0 + flow, flown[:,:,:2]) - hp0
        disn = np.linalg.norm(disn[:,:,:2],2,-1) * 0.1
        occn[occn!=0] = disn[occn!=0]

        # ndc
        flow[:,:,0] = 2 * (flow[:,:,0]/maxw)
        flow[:,:,1] = 2 * (flow[:,:,1]/maxh)
        flow[:,:,2] = np.logical_and(flow[:,:,2]!=0, occ<10)  # as the valid pixels
        flown[:,:,0] = 2 * (flown[:,:,0]/maxw)
        flown[:,:,1] = 2 * (flown[:,:,1]/maxh)
        flown[:,:,2] = np.logical_and(flown[:,:,2]!=0, occn<10)  # as the valid pixels

        flow = np.transpose(flow, (2, 0, 1))
        flown = np.transpose(flown, (2, 0, 1))
        return flow, flown, occ, occn

    def __getitem__(self, index):
        #pdb.set_trace()
        #ss = time.time()
        try:dataid = self.dataid
        except: dataid=0

        im0idx = self.baselist[index]
        if self.directlist[index]==1:
            # forward flow
            im1idx = im0idx + self.dframe 
            flowfw = True
        else:
            im1idx = im0idx - self.dframe
            flowfw = False

        rt_dict, kaug, hp0, A,B = self.read_raw(im0idx, flowfw=flowfw)
        img     = rt_dict['img']  
        mask    = rt_dict['mask']
        flow    = rt_dict['flow']
        occ     = rt_dict['occ']
        dp      = rt_dict['dp']
        vis2d   = rt_dict['vis2d']
        dp_feat = rt_dict['dp_feat']
        dp_bbox = rt_dict['dp_bbox'] 
        rtk     = rt_dict['rtk'] 
        frameid = im0idx
        is_canonical = self.can_frame == im0idx

        if self.load_pair:
            rt_dictn,kaugn,hp1,Ap,Bp = self.read_raw(im1idx, flowfw=(not flowfw))
            imgn  =    rt_dictn['img']
            maskn =    rt_dictn['mask']
            flown =    rt_dictn['flow']
            occn  =    rt_dictn['occ']
            dpn   =    rt_dictn['dp'] 
            vis2dn=    rt_dictn['vis2d']
            dp_featn = rt_dictn['dp_feat']
            dp_bboxn = rt_dictn['dp_bbox'] 
            rtkn     = rt_dictn['rtk'] 
            is_canonicaln = self.can_frame == im1idx
       
            flow, flown, occ, occn = self.flow_process(flow, flown, occ, occn,
                                        hp0, hp1, A,B,Ap,Bp)
            
            # stack data
            img = np.stack([img, imgn])
            mask= np.stack([mask,maskn])
            flow= np.stack([flow, flown])
            occ = np.stack([occ, occn])
            dp  = np.stack([dp, dpn])
            vis2d= np.stack([vis2d, vis2dn])
            dp_feat= np.stack([dp_feat, dp_featn])
            dp_bbox = np.stack([dp_bbox, dp_bboxn])
            rtk= np.stack([rtk, rtkn])         
            kaug= np.stack([kaug,kaugn])
            dataid= np.stack([dataid, dataid])
            frameid= np.stack([im0idx, im1idx])
            is_canonical= np.stack([is_canonical, is_canonicaln])

        elem = {}
        elem['img']           =  img
        elem['mask']          =  mask
        elem['flow']          =  flow
        elem['occ']           =  occ
        elem['dp']            =  dp
        elem['dp_feat']       =  dp_feat
        elem['dp_bbox']       =  dp_bbox
        elem['vis2d']         =  vis2d
        elem['rtk']           =  rtk
        elem['kaug']          =  kaug
        elem['dataid']        =  dataid
        elem['frameid']       =  frameid
        elem['is_canonical']  =  is_canonical
        return elem
