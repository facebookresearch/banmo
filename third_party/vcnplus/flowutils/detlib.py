import pdb
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
import kornia

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
      
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
      np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
  
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
  
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
  
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def get_polarmask(mask):
    # single mask
    mask = np.asarray(mask.cpu()).astype(np.uint8)
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv 4.x
    #_,contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv 3.x
    #contour = [i for i in contour if len(i)>50]
    img = np.zeros(mask.shape+(3,))
    #import pdb; pdb.set_trace()
    img = cv2.drawContours(img, contour, -1, (0, 255, 0), 3) 
    #cv2.imwrite('/data/gengshay/3.png',mask)
    #cv2.imwrite('/data/gengshay/4.png',img)
    contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
    '''debug IndexError: list index out of range'''
    try:
        count = contour[0][:, 0, :]
    except:
        pdb.set_trace()
    try:
        center = get_centerpoint(count)
    except:
        x,y = count.mean(axis=0)
        center=[int(x), int(y)]

    contour = contour[0]
    contour = torch.Tensor(contour).float()
    dists, coords = get_36_coordinates(center[0], center[1], contour)
    return dists, np.asarray(center)


def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0
    a = len(lis)
    for i in range(a):
        lat = lis[i][0]
        lng = lis[i][1]
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]
        fg = (lat * lng1 - lng * lat1) / 2.0
        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0
    x = x / area
    y = y / area

    return [int(x), int(y)]

def get_36_coordinates(c_x, c_y, pos_mask_contour):
    ct = pos_mask_contour[:, 0, :]
    x = ct[:, 0] - c_x
    y = ct[:, 1] - c_y
    # angle = np.arctan2(x, y)*180/np.pi
    angle = torch.atan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.int()
    # dist = np.sqrt(x ** 2 + y ** 2)
    dist = torch.sqrt(x ** 2 + y ** 2)
    angle, idx = torch.sort(angle)
    dist = dist[idx]

    new_coordinate = {}
    for i in range(0, 360, 10):
        if i in angle:
            d = dist[angle==i].max()
            new_coordinate[i] = d
        elif i + 1 in angle:
            d = dist[angle == i+1].max()
            new_coordinate[i] = d
        elif i - 1 in angle:
            d = dist[angle == i-1].max()
            new_coordinate[i] = d
        elif i + 2 in angle:
            d = dist[angle == i+2].max()
            new_coordinate[i] = d
        elif i - 2 in angle:
            d = dist[angle == i-2].max()
            new_coordinate[i] = d
        elif i + 3 in angle:
            d = dist[angle == i+3].max()
            new_coordinate[i] = d
        elif i - 3 in angle:
            d = dist[angle == i-3].max()
            new_coordinate[i] = d


    distances = torch.zeros(36)

    for a in range(0, 360, 10):
        if not a in new_coordinate.keys():
            new_coordinate[a] = torch.tensor(1e-6)
            distances[a//10] = 1e-6
        else:
            distances[a//10] = new_coordinate[a]
    # for idx in range(36):
    #     dist = new_coordinate[idx * 10]
    #     distances[idx] = dist

    return distances, new_coordinate

def polar_reg(output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss,pred

def rigid_transform(p03d,p13d,quat, tran,mask):
    mask = torch.Tensor(mask).cuda()

    for it in range(mask.max().int()):
        obj_mask = mask==(it+1)
        # compute rigid transform
        quatx = torch.nn.functional.normalize(quat[it],2,-1)
        quatx = kornia.quaternion_to_rotation_matrix(quatx)
        p13d[obj_mask] = quatx.matmul(p03d[obj_mask][:,:,None])[:,:,0]+tran[it]
    return p03d,p13d

def pose_reg(quat, tran, pose_px_ind, ind, gt_p03d, gt_p13d, gt_depth, max_obj, p03d_feat,img):
    # solve the scale
    alpha = torch.ones(quat.shape[0]).cuda()
    for i in range(quat.shape[0]):
        d1 = p03d_feat[i,-1]
        d2 = gt_p03d[i,-1].view(-1)
        alpha[i] = (d1*d2).sum()/(d1*d1).sum()
        #pdb.set_trace()
        #from utils.fusion import pcwrite
        #pc1 = np.asarray(p03d_feat[0].T.cpu())
        #pc2 =  np.asarray(gt_p03d[0].view(3,-1).T.cpu())
        #pc1 = pc1*np.asarray(alpha[i].cpu())
        #pcwrite('/data/gengshay/0.ply',np.concatenate([pc1,pc1],-1)) 
        #pcwrite('/data/gengshay/1.ply',np.concatenate([pc2,pc2],-1)) 
    alpha = alpha.detach()    

    vis = torch.zeros_like(gt_depth)
    quat = _transpose_and_gather_feat(quat, ind).view(-1,4)
    tran = _transpose_and_gather_feat(tran, ind).view(-1,3) 
    gt_p03d = gt_p03d.permute(0,2,3,1)
    gt_p13d = gt_p13d.permute(0,2,3,1)
    gt_depth = gt_depth.permute(0,2,3,1)

    loss = []
    for it,obj_mask in enumerate(pose_px_ind):
        imgid = it//max_obj
        if len(obj_mask)>0:
            p03d = gt_p03d[imgid][obj_mask]  
            p13d = gt_p13d[imgid][obj_mask]  
            depth =gt_depth[imgid][obj_mask]  
            
            # compute rigid transform
            quatx = torch.nn.functional.normalize(quat[it],2,-1)
            quatx = kornia.quaternion_to_rotation_matrix(quatx)
            pred_p13d = quatx.matmul(p03d[:,:,None])[:,:,0]+tran[it] * alpha[imgid]
            #pdb.set_trace()
            #from utils.fusion import pcwrite
            #pc1 = np.asarray(p03d.cpu())
            #pc2 =  np.asarray(pred_p13d.detach().cpu())
            #pc3 =  np.asarray(p13d.cpu())
            #rgb = img[imgid][obj_mask].cpu()*255
            #pcwrite('/data/gengshay/0.ply',np.concatenate([pc1,rgb],-1)) 
            #pcwrite('/data/gengshay/1.ply',np.concatenate([pc2,rgb],-1)) 
            #pcwrite('/data/gengshay/2.ply',np.concatenate([pc3,rgb],-1)) 

            sub_loss = ((p13d - pred_p13d)/depth).abs()
            loss.append( sub_loss.mean() )
            # vis
            sub_vis = torch.zeros_like(vis[0,0])
            sub_vis[obj_mask] = sub_loss.mean(-1)
            vis[imgid,0] += sub_vis
    if len(loss)>0:
        loss = torch.stack(loss).mean()
    else:
        loss = 0
    return loss, vis


def distance2mask(points, distances, angles, max_shape=None):
    '''Decode distance prediction to 36 mask points
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 36,from angle 0 to 350.
        angles (Tensor):
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded masks.
    '''
    num_points = points.shape[0]
    points = points[:, :, None].repeat(1, 1, 36)
    c_x, c_y = points[:, 0], points[:, 1]

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = sin[None, :].repeat(num_points, 1)
    cos = cos[None, :].repeat(num_points, 1)

    x = distances * sin + c_x
    y = distances * cos + c_y

    if max_shape is not None:
        x = x.clamp(min=0, max=max_shape[1] - 1)
        y = y.clamp(min=0, max=max_shape[0] - 1)

    res = torch.cat([x[:, None, :], y[:, None, :]], dim=1)
    return res

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100,quat=None,tran =None,p03d=None):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1)
      ys = ys.view(batch, K, 1)
    scores = scores.view(batch, K, 1)

    pdist_ct = torch.cat([xs,ys],-1)
    pdist_ind=(ys*width+xs).long()
    pdist_pred = _transpose_and_gather_feat(wh, pdist_ind[:,:,0])
    if quat is not None:
        quat_pred = _transpose_and_gather_feat(quat, pdist_ind[:,:,0])
        tran_pred = _transpose_and_gather_feat(tran, pdist_ind[:,:,0])
    pdist_mask = (scores>0.1)[:,:,0]

    contour_pred = np.zeros(wh.shape[2:])
    mask_pred = np.zeros(wh.shape[2:])
    angles = torch.range(0, 350, 10).cuda() / 180 * math.pi
    bboxs = np.zeros((0,4))
    p03d = p03d[0].permute(1,2,0)
    p13d = p03d.clone()
    if pdist_mask.sum()>0:
        contour = distance2mask(pdist_ct[0][pdist_mask[0]], pdist_pred[0][pdist_mask[0]], angles, wh.shape[2:])
        contour = np.asarray(contour.permute(0,2,1).cpu()[:,:,None],dtype=int)
        contour_pred = cv2.drawContours(contour_pred, contour, -1,1,3)
        mask_pred,bboxs = draw_masks(mask_pred, np.asarray(pdist_ct[0][pdist_mask[0]].cpu()), contour)
        #pdb.set_trace()
        if quat is not None:
            quat_pred = quat_pred[0][pdist_mask[0]]
            tran_pred = tran_pred[0][pdist_mask[0]]
        #p03d,p13d = rigid_transform(p03d,p13d,quat_pred,tran_pred, mask_pred)
    pred = np.concatenate([contour_pred, mask_pred],0)
    rt = {}
    rt['mask'] = pred
    scores = np.asarray(scores[scores>0.1].cpu())
    rt['bbox'] = np.concatenate([bboxs.reshape((-1,4)), scores[:,None]],-1)
    if quat is not None:
        rt['quat'] = np.asarray(kornia.quaternion_to_rotation_matrix(quat_pred).cpu())
        rt['tran'] = np.asarray(tran_pred.cpu())
    #rt['p03d'] = np.asarray(p03d.cpu())
    #rt['p13d'] = np.asarray(p13d.cpu())
    return rt

def label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [255, 0, 0]
  colormap[2] = [0, 255, 0]
  colormap[3] = [250, 250, 0]
  colormap[4] = [0, 215, 230]
  colormap[5] = [190, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [102, 102, 156]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [0, 0, 230]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [244, 35, 232]
  colormap[18] = [119, 11, 32]
  return colormap

def draw_masks(mask, ct, contour):
    colormap = label_colormap()
    bboxs = []
    for i in np.argsort(ct[:,1]):
        mask = cv2.drawContours(mask, contour[i:i+1], -1,float(i+1),-1)  # x,y
        bboxs.append(np.hstack( (contour[i,:,0].min(0), contour[i,:,0].max(0)) )[None])
    #cv2.imwrite('/data/gengshay/0.png',mask)
    return mask, np.concatenate(bboxs,0)

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
