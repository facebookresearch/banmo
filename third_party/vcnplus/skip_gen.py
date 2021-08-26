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
from torch.autograd import Variable
import torch.nn.functional as F
import time
from flowutils.io import mkdir_p
from flowutils.util_flow import write_flow, save_pfm
import glob
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--dframe', type=int ,default=2,
                    help='how many frames to skip')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
args = parser.parse_args()


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCNplus import VCN
from models.VCNplus import WarpModule, flow_reg
model = VCN([1, 256, 256], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    #pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'dcnet' not in k}
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    ## flow 
    #pretrained_dict = torch.load('../../expansion/weights/exp-kitti-train/exp-kitti-train.pth')
    #pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    #model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def flow_inference(imgL_o, imgR_o):
    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

    # resize
    # set test res
    if args.testres == -1:
        testres = np.sqrt(2*1e6/(imgL_o.shape[0]*imgL_o.shape[1]))
        #testres = np.sqrt(1e6/(imgL_o.shape[0]*imgL_o.shape[1]))
    else:
        testres = args.testres
    maxh = imgL_o.shape[0]*testres
    maxw = imgL_o.shape[1]*testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))
    imgL_noaug = torch.Tensor(imgL/255.)[np.newaxis].float().cuda()

    # flip channel, subtract mean
    imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    # modify module according to inputs
    for i in range(len(model.module.reg_modules)):
        model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                        ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                        maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                        fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
    for i in range(len(model.module.warp_modules)):
        model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()

    # get intrinsics
    intr_list = [torch.Tensor(inxx).cuda() for inxx in [[1],[1],[1],[1],[1],[0],[0],[1],[0],[0]]]
    fl_next = 1
    intr_list.append(torch.Tensor([input_size[1] / max_w]).cuda())
    intr_list.append(torch.Tensor([input_size[0] / max_h]).cuda())
    intr_list.append(torch.Tensor([fl_next]).cuda())
    
    disc_aux = [None,None,None,intr_list,imgL_noaug,None]
    
    # forward
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        imgLR = torch.cat([imgL,imgR],0)
        model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        rts = model(imgLR, disc_aux)
        torch.cuda.synchronize()
        ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        flow, occ, logmid, logexp, fgmask,hm,pm,disp = rts

    # upsampling
    occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    disp = cv2.resize(disp.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    logexp = cv2.resize(logexp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    logmid = cv2.resize(logmid.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    fgmask = cv2.resize(fgmask.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    flow = torch.squeeze(flow).data.cpu().numpy()
    flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                            cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
    flow[:,:,0] *= imgL_o.shape[1] / max_w
    flow[:,:,1] *= imgL_o.shape[0] / max_h

    # deal with unequal size
    x0,y0  =np.meshgrid(range(input_size[1]),range(input_size[0]))
    hp0 = np.stack([x0,y0],-1)  # screen coord
    hp1 = flow + hp0
    hp1[:,:,0] = hp1[:,:,0]/float(imgL_o.shape[1])*float(imgR_o.shape[1]) 
    hp1[:,:,1] = hp1[:,:,1]/float(imgL_o.shape[0])*float(imgR_o.shape[0])
    flow = hp1 - hp0
    #
    #from utils.dydepth import warp_flow
    #warped = warp_flow(imgR_o, flow[:,:,:2])
    #cv2.imwrite('../tmp/0.png', imgL_o)
    #cv2.imwrite('../tmp/1.png', warped)
    #cv2.imwrite('../tmp/2.png', imgR_o)

    flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
    return flow, occ, disp



def main():
    test_left_img = sorted(glob.glob('%s/*'%(args.datapath)))
    seqname = args.datapath.strip().split('/')[-2]
    fwdir = args.datapath.replace('JPEGImages', 'FlowFW').replace(seqname, '%s_%02d'%(seqname, args.dframe))
    bwdir = args.datapath.replace('JPEGImages', 'FlowBW').replace(seqname, '%s_%02d'%(seqname, args.dframe))
    mkdir_p(fwdir)
    mkdir_p(bwdir)
    
    model.eval()
    inx=0;jnx=args.dframe
    while True:
        print('%s/%s'%(test_left_img[inx],test_left_img[jnx]))
        if inx%args.dframe==0:
            imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
            imgR_o = cv2.imread(test_left_img[jnx])[:,:,::-1]
            flowfw, occfw,_ = flow_inference(imgL_o, imgR_o)
            flowbw, occbw,_ = flow_inference(imgR_o, imgL_o)
        else:
            flowfw = np.zeros((10,10,3))
            flowbw = np.zeros((10,10,3))
            occfw = np.zeros((10,10))
            occbw = np.zeros((10,10))
        # save predictions
        with open('%s/flo-%05d.pfm'% (fwdir,inx),'w') as f:
            save_pfm(f,flowfw[::-1].astype(np.float32))
        with open('%s/occ-%05d.pfm'% (fwdir,inx),'w') as f:
            save_pfm(f,occfw[::-1].astype(np.float32))
        with open('%s/flo-%05d.pfm'% (bwdir,jnx),'w') as f:
            save_pfm(f,flowbw[::-1].astype(np.float32))
        with open('%s/occ-%05d.pfm'% (bwdir,jnx),'w') as f:
            save_pfm(f,occbw[::-1].astype(np.float32))
        inx+=1
        jnx+=1
        torch.cuda.empty_cache()
                
            

if __name__ == '__main__':
    main()

