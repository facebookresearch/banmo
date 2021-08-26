import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
import time

# construct model, VCN-expansion
from .models.VCNplus import VCN
from .models.VCNplus import WarpModule, flow_reg

def flow_inference(model, imgL_o,imgR_o, mean_L, mean_R, testres=-1):
    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

    # resize
    # set test res
    if testres == -1:
        testres = np.sqrt(2*1e6/(imgL_o.shape[0]*imgL_o.shape[1]))
        #testres = np.sqrt(1e6/(imgL_o.shape[0]*imgL_o.shape[1]))
    else:
        testres = testres
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
