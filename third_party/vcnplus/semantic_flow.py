from __future__ import print_function
import sys
import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import time
from utils.io import mkdir_p
from utils.util_flow import write_flow, save_pfm
import glob
cudnn.benchmark = False

resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--refname', default='',
                    help='reference name to make comaprison against')
args = parser.parse_args()

# semantic
from models.feature_extraction import FeatureExtraction
model = FeatureExtraction(
    feature_extraction_cnn='vgg19', normalization='false', last_layer="relu1_2,relu2_2,relu3_4,relu4_4,relu5_4")
model.cuda()

def flow_inference(img1, img2,vis=False):
    img1o = torch.Tensor(img1).cuda()[None]
    img2o = torch.Tensor(img2).cuda()[None]
    img1 = resnet_transform(img1o)
    img2 = resnet_transform(img2o)
    # extract cost volume from image features
    c1 = model(img1)[-1][0] # b, f, h, w
    c2 = model(img2)[-1][0] # b, f, h, w
    c_all = model(img1)
    c1n = c1 / (c1.norm(dim=0, keepdim=True)+1e-9)
    c2n = c2 / (c2.norm(dim=0, keepdim=True)+1e-9)

    _,height,width = c1.shape
    #cost = (-100*(c1[:,:,:,None,None]-c2[:,None,None]).norm(2,0)).exp()
    #cost1 = cost / cost.sum(-1).sum(-1)[:,:,None,None]
    #cost2 = cost / cost.sum(0).sum(0)[None,None]
    cost1 = (c1n[:,:,:,None,None]*c2n[:,None,None]).sum(0)
    
    if vis:
        cost1[cost1<0.5] = 0
        for ii in range(height):
            for jj in range(width):
                costvis = np.zeros((height, width, 3)).astype(np.float32)
                costvis[ii,jj] = 255
                costvis = cv2.resize(costvis, (256,256),interpolation=cv2.INTER_NEAREST)
                imgvis_bg = np.asarray(img1o[0].permute(1,2,0).cpu()*255)[:,:,::-1]
                imgvisa = cv2.addWeighted(costvis, 1, imgvis_bg, 1, 0)

                costvis = 255*np.asarray(cost1[ii,jj][:,:,None].repeat(1,1,3).cpu().detach())
                costvis = cv2.resize(costvis, (256,256),interpolation=cv2.INTER_NEAREST)
                imgvis_bg = np.asarray(img2o[0].permute(1,2,0).cpu()*255)[:,:,::-1]
                imgvisb = cv2.addWeighted(costvis, 1, imgvis_bg, 1, 0)
                
                hcostvis = costvis.copy()
                hcostvis[hcostvis!=hcostvis.max()]=0
                hcostvis = cv2.addWeighted(hcostvis, 1, imgvis_bg, 1, 0)

                imgvis = np.concatenate([imgvisa, imgvisb, costvis, hcostvis], 1)
                cv2.imwrite('/data/gengshay/%02d_%02d.png'%(ii,jj), imgvis)

    return c_all

def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path)[:,:,::-1] / 255.

    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

    mask = cv2.imread(img_path.replace('JPEGImages', 'Annotations').replace('.jpg','.png'),0)
    if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
        mask = cv2.resize(mask, img.shape[:2][::-1])
    mask = np.expand_dims(mask, 2)
    
    color = img[mask[:,:,0].astype(bool)].mean(0)
    #img =   img*(mask>0).astype(float) + np.random.rand(mask.shape[0],mask.shape[1],1)*(1-(mask>0).astype(float))
    img =   img*(mask>0).astype(float) + (1-color )[None,None,:]*(1-(mask>0).astype(float))
    img_black =   img*(mask>0).astype(float) + (1-(mask>0).astype(float))

    # crop box
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
#    maxlength = int(1.2*max(length))
#    length = (maxlength,maxlength)
    length = (int(1.2*length[0]), int(1.2*length[1]))

    x0,y0=np.meshgrid(range(2*length[0]),range(2*length[1]))
    #x0 = x0.shape[0]-1-x0
    x0=(x0+(center[0]-length[0])).astype(np.float32)
    y0=(y0+(center[1]-length[1])).astype(np.float32)
    img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=(1-color))
    img_black = cv2.remap(img_black,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=img_black[0,0])
    mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)

    maxw=256;maxh=256
    img = cv2.resize(img ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
    img_black = cv2.resize(img_black ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask ,  (maxw,maxh), interpolation=cv2.INTER_NEAREST)
    #alp = 2*length[0]/maxw
    alp = [2*length[0]/maxw, 2*length[1]/maxw]

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    img_black = np.transpose(img_black, (2, 0, 1))

    return img, alp, img_black, mask

def main():
    test_left_img = sorted(glob.glob('%s/*'%(args.datapath)))
    seqname = args.datapath.strip().split('/')[-2]
    corrdir = args.datapath.replace('JPEGImages', 'AppFeat')
    mkdir_p(corrdir)
    
    model.eval()
    for inx in range(len(test_left_img)):
        jnx=inx
        print('%s/%s'%(test_left_img[inx],test_left_img[jnx]))
        imgL_o,_,_,_ = preprocess_image(test_left_img[inx])
        imgR_o,_,_,_ = preprocess_image(test_left_img[jnx])
        c_all = flow_inference(imgL_o, imgR_o)
        for layer in range( len(c_all) ):
            if layer<3: continue 
            with open('%s/feat%d-%06d.pfm'% (corrdir,layer,inx),'w') as f:
                c1 = c_all[layer][0]
                c1 = c1.view(c1.shape[0],-1)
                c1 = np.asarray(c1.cpu())
                save_pfm(f,c1[::-1].astype(np.float32))
       
        if args.refname!='no':
            ref_left_img =  sorted(glob.glob('%s/*'%(args.datapath.replace(seqname, args.refname))))
            jnx=4
            print('%s/%s'%(test_left_img[inx],ref_left_img[jnx]))
            imgL_o,_,_,_ = preprocess_image(test_left_img[inx])
            imgR_o,_,_,_ = preprocess_image(ref_left_img[jnx])
            _ = flow_inference(imgL_o, imgR_o, vis=True)
            pdb.set_trace()
            torch.cuda.empty_cache()
            

if __name__ == '__main__':
    main()

