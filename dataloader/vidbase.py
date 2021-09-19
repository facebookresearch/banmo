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
        Ap = A

        A = A.dot(self.rot_augment(x0))
        Ap=Ap.dot(self.rot_augment(x0))

        return A,Ap

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        #pdb.set_trace()
        #ss = time.time()
        im0idx = self.baselist[index]
        im1idx = im0idx + self.dframe if self.directlist[index]==1 else im0idx-self.dframe
        #if im0idx==0:pdb.set_trace()
        img_path = self.imglist[im0idx]
        #img = imread(img_path) / 255.0
        img = cv2.imread(img_path)[:,:,::-1] / 255.0

        img_path = self.imglist[im1idx]
        #imgn = imread(img_path) / 255.0
        imgn = cv2.imread(img_path)[:,:,::-1] / 255.0
        # Some are grayscale:
        shape = img.shape
        if len(shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
            imgn = np.repeat(np.expand_dims(imgn, 2), 3, axis=2)

        mask = cv2.imread(self.masklist[im0idx],0)
        mask = mask/np.sort(np.unique(mask))[1]
        occluder = mask==255
        mask[occluder] = 0
        if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
            mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            mask = binary_erosion(mask,iterations=2)
        mask = np.expand_dims(mask, 2)

        maskn = cv2.imread(self.masklist[im1idx],0)
        maskn = maskn/np.sort(np.unique(maskn))[1]
        occludern = maskn==255
        maskn[occludern] = 0
        if maskn.shape[0]!=imgn.shape[0] or maskn.shape[1]!=imgn.shape[1]:
            maskn = cv2.resize(maskn, imgn.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            maskn = binary_erosion(maskn,iterations=1)
        maskn = np.expand_dims(maskn, 2)

        # compement color
        color = 1-img[mask[:,:,0].astype(bool)].mean(0)[None,None,:]
        colorn = 1-imgn[maskn[:,:,0].astype(bool)].mean(0)[None,None,:]
        color[:]=0
        colorn[:]=0
        #color = np.random.uniform(0,1,shape)
        #colorn = np.random.uniform(0,1,shape)
#        img =   img*(mask>0).astype(float)    + color  *(1-(mask>0).astype(float))
#        imgn =   imgn*(maskn>0).astype(float) + colorn *(1-(maskn>0).astype(float))

        # flow
        if self.directlist[index]==1:
            flowpath = self.flowfwlist[im0idx]
            flowpathn =self.flowbwlist[im0idx+self.dframe]
        else:
            flowpath = self.flowbwlist[im0idx]
            flowpathn =self.flowfwlist[im0idx-self.dframe]
        try:
            flow = readPFM(flowpath)[0]
            flown =readPFM(flowpathn)[0]
            occ = readPFM(flowpath.replace('flo-', 'occ-'))[0]
            occn =readPFM(flowpathn.replace('flo-', 'occ-'))[0]
        except:
            print('warning: loading empty flow')
            flow = np.zeros_like(img)
            flown = np.zeros_like(img)
            occ = np.zeros_like(mask)
            occn = np.zeros_like(mask)
        occ[occluder] = 0
        occn[occludern] = 0
        try:
            depth = readPFM(flowpath.replace('flo-', 'depth-').replace('FlowBW', 'Depth').replace('FlowFW', 'Depth'))[0]
            depthn= readPFM(flowpathn.replace('flo-', 'depth-').replace('FlowBW', 'Depth').replace('FlowFW', 'Depth'))[0]
        except:
            depth = np.zeros_like(occ)
            depthn = np.zeros_like(occ)
        #print('time: %f'%(time.time()-ss))

        # read kp
        try:
            kp = read_json('%s'%self.kplist[im0idx], mask)
            kpn= read_json('%s'%self.kplist[im1idx], maskn)
        except:
            kp = np.zeros((25,3))
            kpn = np.zeros((25,3))
        
        try:
            dp = readPFM(self.dplist[im0idx])[0]
            dpn= readPFM(self.dplist[im1idx])[0]
            dp_feat = readPFM(self.featlist[im0idx])[0]
            dp_featn= readPFM(self.featlist[im1idx])[0]
            dp_bbox =  np.loadtxt(self.bboxlist[im0idx])
            dp_bboxn = np.loadtxt(self.bboxlist[im1idx])
        except:
            print('error loading densepose')
            dp = np.zeros_like(occ)
            dpn = np.zeros_like(occ)
            dp_feat =  np.zeros((16*112,112))
            dp_featn = np.zeros((16*112,112))
            dp_bbox =  np.zeros((4))
            dp_bboxn = np.zeros((4))
        dp= (dp *50).astype(np.int32)
        dpn=(dpn*50).astype(np.int32)
        dp_feat = dp_feat.reshape((16,112,112))
        dp_featn=dp_featn.reshape((16,112,112))

        # create mask for visible vs unkonwn
        vis2d = np.ones_like(mask)
        vis2dn= np.ones_like(mask)

        # crop box
        indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
        indicesn = np.where(maskn>0); xidn = indicesn[1]; yidn = indicesn[0]
        center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
        centern = ( (xidn.max()+xidn.min())//2, (yidn.max()+yidn.min())//2)
        length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
        lengthn = ( (xidn.max()-xidn.min())//2, (yidn.max()-yidn.min())//2)
        
        #maxlength = int(1.2*max(length))
        #maxlengthn = int(1.2*max(lengthn))
        #length = (maxlength,maxlength)
        #lengthn = (maxlengthn,maxlengthn)

        length = (int(self.crop_factor*length[0]), int(self.crop_factor*length[1]))
        lengthn= (int(self.crop_factor*lengthn[0]),int(self.crop_factor*lengthn[1]))

        maxw=self.img_size;maxh=self.img_size
        orisize = (2*length[0], 2*length[1])
        orisizen= (2*lengthn[0], 2*lengthn[1])
        alp =  [orisize[0]/maxw  ,orisize[1]/maxw]
        alpn = [orisizen[0]/maxw ,orisizen[1]/maxw]
        x0,y0  =np.meshgrid(range(maxw),range(maxh))
        # geometric augmentation for img, mask, flow, occ
        A,Ap = self.geo_augment(x0)
        B = np.asarray([[alp[0],0,(center[0]-length[0])],
                        [0,alp[1],(center[1]-length[1])],
                        [0,0,1]]).T
        Bp= np.asarray([[alpn[0],0,(centern[0]-lengthn[0])],
                        [0,alpn[1],(centern[1]-lengthn[1])],
                        [0,0,1]]).T
        
        hp0 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
        hp1 = np.stack([x0,y0,np.ones_like(x0)],-1)  # screen coord
        hp0 = np.dot(hp0,A).dot(B)                   # image coord
        hp1 = np.dot(hp1,Ap).dot(Bp)                  # image coord
        x0 = hp0[:,:,0].astype(np.float32)
        y0 = hp0[:,:,1].astype(np.float32)
        x0n = hp1[:,:,0].astype(np.float32)
        y0n = hp1[:,:,1].astype(np.float32)
        
        kp[:,:2] = np.concatenate([kp[:,:2], np.ones_like(kp[:,:1])],-1).dot(np.linalg.inv(A.dot(B)))[:,:2]
        kpn[:,:2]= np.concatenate([kpn[:,:2],np.ones_like(kpn[:,:1])],-1).dot(np.linalg.inv(Ap.dot(Bp)))[:,:2]
        kp[:,0] = kp[:,0] / maxw * 2-1
        kp[:,1] = kp[:,1] / maxh * 2-1
        kpn[:,0] = kpn[:,0] / maxw * 2-1
        kpn[:,1] = kpn[:,1] / maxh * 2-1
        
        img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=color[0,0])
        mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
        flow = cv2.remap(flow,x0,y0,interpolation=cv2.INTER_LINEAR)
        occ = cv2.remap(occ,x0,y0,interpolation=cv2.INTER_LINEAR)
        depth=cv2.remap(depth,x0,y0,interpolation=cv2.INTER_LINEAR)
        dp   =cv2.remap(dp,   x0,y0,interpolation=cv2.INTER_NEAREST)
        vis2d=cv2.remap(vis2d.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)

        imgn = cv2.remap(imgn,x0n,y0n,interpolation=cv2.INTER_LINEAR,borderValue=colorn[0,0])
        maskn = cv2.remap(maskn.astype(int),x0n,y0n,interpolation=cv2.INTER_NEAREST)
        flown = cv2.remap(flown,x0n,y0n,interpolation=cv2.INTER_LINEAR)
        occn = cv2.remap(occn,x0n,y0n,interpolation=cv2.INTER_LINEAR)
        depthn = cv2.remap(depthn,x0n,y0n,interpolation=cv2.INTER_LINEAR)
        dpn    =cv2.remap(dpn,    x0n,y0n,interpolation=cv2.INTER_NEAREST)
        vis2dn=cv2.remap(vis2dn.astype(int),x0n,y0n,interpolation=cv2.INTER_NEAREST)

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

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        mask = (mask>0).astype(float)
        
        imgn = np.transpose(imgn, (2, 0, 1))
        maskn = (maskn>0).astype(float)
        flow = np.transpose(flow, (2, 0, 1))
        flown = np.transpose(flown, (2, 0, 1))
            

        cam = np.zeros((7,))
        cam = np.asarray([1.,0.,0. ,1.,0.,0.,0.])
        camn = np.asarray([1.,0.,0. ,1.,0.,0.,0.])
        #depth=0.; depthn=0.
        # correct cx,cy at clip space (not tx, ty)
        if self.flip==0:
            pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
            ppsn = np.asarray([float( centern[0]- lengthn[0]), float(centern[1] - lengthn[1] )])
        else:
            pps  = np.asarray([-float( center[0] - length[0] ), float( center[1] - length[1]  )])
            ppsn = np.asarray([-float( centern[0]- lengthn[0]), float(centern[1] - lengthn[1] )])
        if False:#osp.exists(self.camlist[im0idx]):
            cam0=np.loadtxt(self.camlist[im0idx]).astype(np.float32)
            cam1=np.loadtxt(self.camlist[im1idx]).astype(np.float32)
            cam[:]=cam0[:-1]
            camn[:]=cam1[:-1]
            #cam[0]/=alp   # modify focal length according to rescale
            #camn[0]/=alpn
            cam[0]=1./alp[0]   # modify focal length according to rescale
            camn[0]=1./alpn[0]
            depth = cam0[-1:]
            depthn = cam1[-1:]
        else:
            cam[:1]=1./alp[0]   # modify focal length according to rescale
            camn[:1]=1./alpn[0]
            cam[1:2]=1./alp[1]   # modify focal length according to rescale
            camn[1:2]=1./alpn[1]

        mask = np.stack([mask,maskn])
        vis2d= np.stack([vis2d, vis2dn])

        try:dataid = self.dataid
        except: dataid=0

        # add RTK: [R_3x3|T_3x1]
        #          [fx,fy,px,py], to the ndc space
        try:
            rtk_path = self.rtklist[im0idx]
            rtkn_path =self.rtklist[im1idx]
            rtk = np.loadtxt(rtk_path)
            rtkn = np.loadtxt(rtkn_path)
        except:
            print('warning: loading empty camera')
            rtk = np.zeros((4,4))
            rtk[:3,:3] = np.eye(3)
            rtk[:3, 3] = np.asarray([0,0,10])
            rtk[3, :]  = np.asarray([512,512,256,256]) 
            rtkn = rtk.copy()

        # intrinsics induced by augmentation: augmented to to original img
        kaug = np.asarray([alp[0], alp[1], pps[0], pps[1]])
        kaugn= np.asarray([alpn[0],alpn[1],ppsn[0],ppsn[1]])

        # remove background
        elem = {
            'img':          np.stack([img, imgn]),
            'mask':         mask,
            'flow':         np.stack([flow, flown]),
            'occ':          np.stack([occ, occn]),
            'pps':          np.stack([pps, ppsn]),
            'depth':        np.stack([depth, depthn]),
            'dp':           np.stack([dp, dpn]),
            'dp_feat':      np.stack([dp_feat, dp_featn]),
            'dp_bbox':      np.stack([dp_bbox, dp_bboxn]),
            'vis2d':        vis2d,
            'cam':          np.stack([cam, camn]),
            'kp':           np.stack([kp, kpn]),
            'rtk':          np.stack([rtk, rtkn]),            
            'kaug':         np.stack([kaug,kaugn]),            

            'dataid':       np.stack([dataid, dataid]),
            'frameid':      np.stack([im0idx, im1idx]),
            'is_canonical': np.stack([self.can_frame == im0idx, self.can_frame == im1idx]),
            }
        return elem
