gpuid = 1
import pdb
import sys
import torch
import numpy as np
import cv2
def write_calib(K,bl,shape,maxd,path):
    str1 = 'camera.A=[%f 0 %f; 0 %f %f; 0 0 1]'%(K[0,0], K[0,2], K[1,1],K[1,2])
    str2 = 'camera.height=%d'%(shape[0])
    str3 = 'camera.width=%d' %(shape[1])
    str4 = 'camera.zmax=%f'%(maxd)
    str5 = 'rho=%f'%(bl*K[0,0])
    with open(path,'w') as f:
        f.write('%s\n%s\n%s\n%s\n%s'%(str1,str2,str3,str4,str5))

def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n".encode() if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)

def triangulation(disp, xcoord, ycoord, bl=1, fl = 450, cx = 479.5, cy = 269.5):
    mask = (disp<=0).flatten()
    depth = bl*fl / (disp) # 450px->15mm focal length
    X = (xcoord - cx) * depth / fl
    Y = (ycoord - cy) * depth / fl
    Z = depth
    P = np.concatenate((X[np.newaxis],Y[np.newaxis],Z[np.newaxis]),0).reshape(3,-1)
    P = np.concatenate((P,np.ones((1,P.shape[-1]))),0)
    P[:,mask]=0
    return P

def midpoint_triangulate(x, cam):
    """
    Args:
        x:   Set of 2D points in homogeneous coords, (3 x n x N) matrix
        cam: Collection of n objects, each containing member variables
                 cam.P - 3x4 camera matrix [0]
                 cam.R - 3x3 rotation matrix [1]
                 cam.T - 3x1 translation matrix [2]
    Returns:
        midpoint: 3D point in homogeneous coords, (4 x 1) matrix
    """
    n = len(cam)                                         # No. of cameras
    N = x.shape[-1]

    I = np.eye(3)                                        # 3x3 identity matrix
    A = np.zeros((3,n))
    B = np.zeros((3,n,N))
    sigma2 = np.zeros((3,N))

    for i in range(n):
        a = -np.linalg.inv(cam[i][:3,:3]).dot(cam[i][:3,-1:])        # ith camera position # 
        A[:,i,None] = a
        if i==0:
            b = np.linalg.pinv(cam[i][:3,:3]).dot(x[:,i])              # Directional vector # 4, N
        else:
            b = np.linalg.pinv(cam[i]).dot(x[:,i])              # Directional vector # 4, N
            b = b / b[3:]
            b = b[:3,:] - a  # 3,N
        b = b / np.linalg.norm(b,2,0)[np.newaxis]
        B[:,i,:] = b

        sigma2 = sigma2 + b * (b.T.dot(a).reshape(-1,N)) # 3,N
        
    Bo = B.transpose([2,0,1])
    Bt = B.transpose([2,1,0])    
    
    Bo = torch.DoubleTensor(Bo)
    Bt = torch.DoubleTensor(Bt)
    A = torch.DoubleTensor(A)
    sigma2 = torch.DoubleTensor(sigma2)
    I = torch.DoubleTensor(I)
    
    BoBt = torch.matmul(Bo, Bt)
    C = (n * I)[np.newaxis] -  BoBt# N,3,3
    Cinv = C.inverse()
    sigma1 = torch.sum(A, axis=1)[:,None]
    m1 = I[np.newaxis] + torch.matmul(BoBt,Cinv)
    m2 = torch.matmul(Cinv,sigma2.T[:,:,np.newaxis])
    midpoint = (1/n) * torch.matmul(m1,sigma1[np.newaxis]) - m2       
    
    midpoint = np.asarray(midpoint)   
    return midpoint[:,:,0].T, np.asarray(Bo)

def register_disp_fast(id_flow, id_mono, mask, inlier_th=0.01,niters=100):
    """ 
    input: disp_flow, disp_mono, mask
    output: inlier_mask, registered
    register up-to-scale rough depth to motion-based depth
    """
    shape = id_mono.shape
    id_mono = id_mono.flatten()
    disp_flow = id_flow[mask] # register to flow with mono
    disp_mono = id_mono[mask]
    
    num_samp = min(3000,len(disp_flow))
    np.random.seed(0)
    submask = np.random.choice(range(len(disp_flow)), num_samp)
    disp_flow = disp_flow[submask]
    disp_mono = disp_mono[submask]
    
    n = len(disp_flow)
    sample_size=niters
    rand_idx = np.random.choice(range(n),sample_size)
    scale_cand = (disp_flow/disp_mono)[rand_idx]
    dis_cand = np.abs(np.log(disp_mono[:,np.newaxis]*scale_cand[np.newaxis])-np.log(disp_flow[:,np.newaxis]))
    
    rank_metric = (dis_cand<inlier_th).sum(0)
    scale_idx = np.argmax(rank_metric)
    scale = scale_cand[scale_idx]

#    # another way to align scale
#    from scipy.optimize import minimize
#    def cost_function(alpha, K):
#        return np.mean(np.abs(alpha*K - 1))
#
#    # MRE minimize
#    output = minimize(cost_function, 1., args=(disp_mono/disp_flow),method='Nelder-Mead')
#    if output.success:
#        scale = output.x
    
    dis = np.abs(np.log(disp_mono*scale)-np.log(disp_flow))
    ninliers = (dis<inlier_th).sum()/n
    registered_flow=(id_flow.reshape(shape))/scale

    return registered_flow, scale, ninliers


def testEss(K0,K1,R,T,p1,p2):
    testP = cv2.triangulatePoints(K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)), 
                          K1.dot(np.concatenate( (R,T), -1)), 
                          p1[:2],p2[:2])
    Z1 = testP[2,:]/testP[-1,:]
    Z2 = (R.dot(Z1*np.linalg.inv(K0).dot(p1))+T)[-1,:]
    if ((Z1>0).sum() > (Z1<=0).sum()) and ((Z2>0).sum() > (Z2<=0).sum()):
        #print(Z1)
        #print(Z2)
        return True
    else:
        return False
    
def pose_estimate(K0,K1,hp0,hp1,strict_mask,rot,th=0.0001):
#    # epipolar geometry
#    from models.submodule import F_ngransac
#    tmphp0 = hp0[:,strict_mask]
#    tmphp1 = hp1[:,strict_mask]
#    #num_samp = min(300000,tmphp0.shape[1])
#    num_samp = min(30000,tmphp0.shape[1])
#    #num_samp = min(3000,tmphp0.shape[1])
#    submask = np.random.choice(range(tmphp0.shape[1]), num_samp)
#    tmphp0 = tmphp0[:,submask]
#    tmphp1 = tmphp1[:,submask]
#    
#    rotx,transx,Ex = F_ngransac(torch.Tensor(tmphp0.T[np.newaxis]).cuda(),
#                                torch.Tensor(tmphp1.T[np.newaxis]).cuda(),
#                                torch.Tensor(K0[np.newaxis]).cuda(),
#                                False,0,
#                                Kn = torch.Tensor(K1[np.newaxis]).cuda())
#    R01 = cv2.Rodrigues(np.asarray(rotx[0]))[0]
#    T01 = np.asarray(transx[0])
#    E =  np.asarray(Ex[0])
#    _,R01,T01,_ = cv2.recoverPose(E.astype(float), tmphp0[:2].T, tmphp1[:2].T, K0)  # RT are 0->1 points transform
#    T01 = T01[:,0]
#    R01=R01.T
#    T01=-R01.dot(T01)  # now are 1->0 points transform

    E, maskk = cv2.findEssentialMat(np.linalg.inv(K0).dot(hp0[:,strict_mask])[:2].T, 
                                    np.linalg.inv(K1).dot(hp1[:,strict_mask])[:2].T, np.eye(3),
                                   cv2.LMEDS,threshold=th)

    valid_points = np.ones((strict_mask.sum())).astype(bool)
    valid_points[~maskk[:,0].astype(bool)]=False
    fmask = strict_mask.copy()
    fmask[strict_mask]=valid_points    

    R1, R2, T = cv2.decomposeEssentialMat(E) 
    for rott in [(R1,T),(R2,T),(R1,-T),(R2,-T)]:
        if testEss(K0,K1,rott[0],rott[1],hp0[:,fmask], hp1[:,fmask]):
            R01=rott[0].T
            T01=-R01.dot(rott[1][:,0])
    if not 'T01' in locals():
        T01 = np.asarray([0,0,1])
        R01 = np.eye(3)
    T01t = T01.copy()

    # compensate R
    H01 = K0.dot(R01).dot(np.linalg.inv(K1)) # plane at infinity
    comp_hp1 = H01.dot(hp1)
    comp_hp1 = comp_hp1/comp_hp1[-1:]
    return R01,T01,H01,comp_hp1,E

def evaluate_tri(t10,R01,K0,K1,hp0,hp1,disp0,ent,bl,inlier_th=0.1,select_th=0.4, valid_mask=None):   
    if valid_mask is not None:
        hp0 = hp0[:,valid_mask]
        hp1 = hp1[:,valid_mask]
        disp0 = disp0.flatten()[valid_mask]
        ent = ent.flatten()[valid_mask]
    # triangluation
    #import time; beg = time.time()
    cams = [K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)),
           K1.dot(np.concatenate( (R01.T,-R01.T.dot(t10[:,np.newaxis])), -1)) ]
    P_pred,_ = midpoint_triangulate( np.concatenate([hp0[:,np.newaxis],hp1[:,np.newaxis]],1),cams)
    #print(1000*(time.time()-beg))
    idepth_p3d = np.clip(K0[0,0]*bl/P_pred[2], 1e-6, np.inf)

    # discard points with small disp
    entmask = np.logical_and(idepth_p3d>1e-12, ~np.isinf(idepth_p3d))
    entmask_tmp = entmask[entmask].copy()
    entmask_tmp[np.argsort(-idepth_p3d[entmask])[entmask.sum()//2:]]=False  # remove sky
    entmask[entmask] = entmask_tmp
    med = np.median(idepth_p3d[entmask])
    entmask = np.logical_and(entmask, np.logical_and(idepth_p3d>med/5., idepth_p3d<med*5))
    if entmask.sum()<10:
        return None,None,None
    registered_p3d,scale,ninliers = register_disp_fast(idepth_p3d, disp0, entmask, 
                                                  inlier_th=inlier_th,niters=100)
    print('size/inlier ratio: %d/%.2f'%(entmask.sum(),ninliers)) 
    
    disp_ratio = np.abs(np.log(registered_p3d.flatten()/disp0.flatten()))
    agree_mask = disp_ratio<np.log(select_th)
    rank = np.argsort(disp_ratio)
    return agree_mask,t10*scale,rank

def rb_fitting(bgmask_pred,mask_pred,idepth,flow,ent,K0,K1,bl,parallax_th=2,mono=True,sintel=False,tranpred=None,quatpred=None):
    if sintel: parallax_th = parallax_th*0.25
    # prepare data
    shape = flow.shape[:2]
    x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
    x0=x0.astype(np.float32)
    y0=y0.astype(np.float32)
    x1=x0+flow[:,:,0]
    y1=y0+flow[:,:,1]
    hp0 = np.concatenate((x0[np.newaxis],y0[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
    hp1 = np.concatenate((x1[np.newaxis],y1[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
    
    # use bg + valid pixels to compute R/t
    valid_mask = np.logical_and(bgmask_pred, ent<0).flatten()
    R01,T01,H01,comp_hp1,E = pose_estimate(K0,K1,hp0,hp1,valid_mask,[0,0,0])    

    parallax = np.transpose((comp_hp1[:2]-hp0[:2]),[1,0]).reshape(x1.shape+(2,))        
    parallax_mag = np.linalg.norm(parallax[:,:,:2],2,2)
    flow_mag = np.linalg.norm(flow[:,:,:2],2,2)
    print('[BG Fitting] mean pp/flow: %.1f/%.1f px'%(parallax_mag[bgmask_pred].mean(), flow_mag[bgmask_pred].mean()))

    reg_flow_P = triangulation(idepth, x0, y0, bl=bl, fl = K0[0,0], cx = K0[0,2], cy = K0[1,2])[:3]
    if parallax_mag[bgmask_pred].mean()<parallax_th:
        # static camera
        print("static")
        scene_type = 'H'
        T01_c = [0,0,0]
    else:
        scene_type = 'F'    
        # determine scale of translation / reconstruction
        aligned_mask,T01_c,ranked_p = evaluate_tri(T01,R01,K0,K1,hp0,hp1,idepth,ent,bl,inlier_th=0.01,select_th=1.2,valid_mask=valid_mask)
        if not mono:
             # PnP refine
             aligned_mask[ranked_p[50000:]]=False
             tmp = valid_mask.copy()
             tmp[tmp] = aligned_mask
             aligned_mask = tmp
             _,rvec, T01=cv2.solvePnP(reg_flow_P.T[aligned_mask.flatten(),np.newaxis],
                                        hp1[:2].T[aligned_mask.flatten(),np.newaxis], K0, 0, 
                                        flags=cv2.SOLVEPNP_DLS)
             _,rvec, T01,=cv2.solvePnP(reg_flow_P.T[aligned_mask,np.newaxis],
                                     hp1[:2].T[aligned_mask,np.newaxis], K0, 0,rvec, T01,useExtrinsicGuess=True, 
                                     flags=cv2.SOLVEPNP_ITERATIVE)
             R01 = cv2.Rodrigues(rvec)[0].T
             T01_c = -R01.dot(T01)[:,0]

    RTs = []
    for i in range(0,mask_pred.max()):    
        obj_mask = (mask_pred==i+1).flatten()
        valid_mask = np.logical_and(obj_mask, ent.reshape(obj_mask.shape)<0) 
        if valid_mask.sum()<10 or (valid_mask.sum() / obj_mask.sum() < 0.3):
            RT01 = None
        else:
            if tranpred is None:
                R01x,T01_cx,_,comp_hp1,_ = pose_estimate(K0,K1,hp0,hp1,valid_mask,[0,0,0])
                parallax = np.transpose((comp_hp1[:2]-hp0[:2]),[1,0])
                parallax_mag = np.linalg.norm(parallax,2,-1)
                center_coord = hp0[:,obj_mask].mean(-1)
                print('[FG-%03d Fitting] center/mean pp/flow: (%d,%d)/%.1f/%.1f px'%(i,
                 center_coord[0], center_coord[1], parallax_mag[obj_mask].mean(), 
                 flow_mag.flatten()[obj_mask].mean()))
                if parallax_mag[obj_mask].mean()<parallax_th: RTs.append(None);continue
            else:
                R01x = quatpred[i].T
                T01_cx = -quatpred[i].T.dot(tranpred[i][:,None])[:,0]
                T01_cx = T01_cx / np.linalg.norm(T01_cx)
            aligned_mask,T01_cx,ranked_p = evaluate_tri(T01_cx,R01x,K0,K1,hp0,hp1,idepth,ent,bl,inlier_th=0.01,select_th=1.2,valid_mask=valid_mask)
            if T01_cx is None: RTs.append(None); continue
            if not mono:
                aligned_mask[ranked_p[50000:]]=False
                tmp = valid_mask.copy()
                tmp[tmp] = aligned_mask
                obj_mask = tmp
                if tranpred is None:
                    _,rvec, T01_cx=cv2.solvePnP(reg_flow_P.T[obj_mask,np.newaxis],
                                           hp1[:2].T[obj_mask,np.newaxis], K0, 0, 
                                           flags=cv2.SOLVEPNP_DLS)
                else:
                    rvec = cv2.Rodrigues(R01x.T)[0]
                    T01_cx = -R01x.T.dot(T01_cx[:,None])
                _,rvec, T01_cx=cv2.solvePnP(reg_flow_P.T[obj_mask,np.newaxis],
                                           hp1[:2].T[obj_mask,np.newaxis], K0, 0,rvec, T01_cx,useExtrinsicGuess=True, 
                                           flags=cv2.SOLVEPNP_ITERATIVE)
                R01x = cv2.Rodrigues(rvec)[0].T
                T01_cx = -R01x.dot(T01_cx)[:,0]
            if T01_cx is None:
                RT01=None
            else:                        
                RT01 = [R01x, T01_cx]
        RTs.append(RT01)

    return scene_type, T01_c, R01,RTs

def mod_flow(bgmask,mask_pred, idepth,disp1,flow,ent,bl,K0,K1,scene_type, T01_c,R01, RTs, segs_unc, oracle=None, mono=True,sintel=False):
    # prepare data
    idepth = idepth.copy()
    flow = flow.copy()
    shape = flow.shape[:2]
    x0,y0=np.meshgrid(range(shape[1]),range(shape[0]))
    x0=x0.astype(np.float32)
    y0=y0.astype(np.float32)
    x1=x0+flow[:,:,0]
    y1=y0+flow[:,:,1]
    hp0 = np.concatenate((x0[np.newaxis],y0[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
    hp1 = np.concatenate((x1[np.newaxis],y1[np.newaxis],np.ones(x1.shape)[np.newaxis]),0).reshape((3,-1))
    reg_flow_P = triangulation(idepth, x0, y0, bl=bl, fl = K0[0,0], cx = K0[0,2], cy = K0[1,2])[:3]
    
    # modify motion fields
    if scene_type == 'H':
        H,maskh = cv2.findHomography(hp0.T[ent.flatten()<0], hp1.T[ent.flatten()<0], cv2.FM_RANSAC,ransacReprojThreshold=5)
        mod_mask = np.logical_and(bgmask,ent>0)
        comp_hp0 = H.dot(hp0); comp_hp0 = comp_hp0/comp_hp0[-1:]
        flow[mod_mask] = np.transpose((comp_hp0-hp0).reshape((3,)+shape), (1,2,0))[mod_mask]
    elif scene_type == 'F':
        mod_mask = bgmask
        # modify disp0 | if monocular
        if not (T01_c is None or np.isinf(np.linalg.norm(T01_c))):
            print('[BG Update] cam trans mag: %.2f'%np.linalg.norm(T01_c))
            if mono:
                cams = [K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)),
                            K1.dot(np.concatenate( (R01.T,-R01.T.dot(T01_c[:,np.newaxis])), -1)) ]
                pts = np.concatenate([hp0[:,np.newaxis,mod_mask.flatten()],
                                  hp1[:,np.newaxis,mod_mask.flatten()]],1)
                P_flow,cray = midpoint_triangulate(pts ,cams)
                cflow = 1-(1/(1 + np.exp(-ent)) )
                cmotion = 1-segs_unc
                angle_th = 0.2
                cangle = np.clip(np.arccos(np.abs(np.sum(cray[:,:,0] * cray[:,:,1],-1))) / np.pi * 180, 0,angle_th) # N,3,2
                cangle = 1-np.power((cangle-angle_th)/angle_th,2)
                cangle_tmp = np.zeros(shape)
                cangle_tmp[mod_mask] = cangle
                conf_depth = (cmotion*cflow*cangle_tmp)
                lflow = (cmotion*cangle_tmp)
                dcmask = np.logical_or(lflow[mod_mask]<0.25, P_flow[-1]<1e-12)
                P_flow[:,dcmask] = reg_flow_P[:,mod_mask.flatten()][:,dcmask] # dont change
                reg_flow_P[:,mod_mask.flatten()] = P_flow        
        
            # disp 1
            reg_flow_PP = R01.T.dot(reg_flow_P)-R01.T.dot(T01_c)[:,np.newaxis]
            hpp1 = K0.dot(reg_flow_PP)
            hpp1 = hpp1/hpp1[-1:]
            if not mono:
                flow[mod_mask] = (hpp1 - hp0).T.reshape(shape+(3,))[mod_mask]
            disp1[mod_mask] = bl*K0[0,0]/reg_flow_PP[-1].reshape(shape)[mod_mask]
       
    # obj
    for i in range(0,mask_pred.max()):
        if sintel:break
        obj_mask = mask_pred==i+1
        if oracle is not None:
            if (obj_mask).sum()>0:
                 # use midas depth
                if np.median(idepth[obj_mask])==0:  continue
                reg_flow_P[2,obj_mask.flatten()] = bl*K0[0,0] / (np.median(oracle[obj_mask]) / np.median(idepth[obj_mask])  * idepth[obj_mask])
        else:
            if RTs[i] is not None:
                mod_mask = obj_mask
                T01_c_sub = RTs[i][1]
                if not np.isinf(np.linalg.norm(T01_c_sub)):
                    R01_sub = RTs[i][0]
                    print('[FG-%03d Update] ins trans norm: %.2f'%(i,np.linalg.norm(T01_c_sub)))
                    if mono:
                        # mono replace
                        cams = [K0.dot(np.concatenate( (np.eye(3),np.zeros((3,1))), -1)),
                        K1.dot(np.concatenate( (R01_sub.T,-R01_sub.T.dot(T01_c_sub[:,np.newaxis])), -1)) ]
                        pts = np.concatenate([hp0[:,np.newaxis,mod_mask.flatten()],
                                              hp1[:,np.newaxis,mod_mask.flatten()]],1)
                        P_flow,det = midpoint_triangulate(pts ,cams)
                        med = np.median(P_flow[2])
                        reg_flow_P[:,mod_mask.flatten()] = P_flow # modify disp0 | if monocular
                        print('[FG-%03d Update] size:%d/center:%.1f,%.1f/med:%.1f'%(i, P_flow.shape[1],pts[:,0].mean(-1)[0],pts[:,0].mean(-1)[1], med))

                    # disp 1
                    reg_flow_PP = R01_sub.T.dot(reg_flow_P)-R01_sub.T.dot(T01_c_sub)[:,np.newaxis]
                    hpp1 = K0.dot(reg_flow_PP)
                    hpp1 = hpp1/hpp1[-1:]
                    if not mono:
                        flow[mod_mask] = (hpp1 - hp0).T.reshape(shape+(3,))[mod_mask]
                    disp1[mod_mask] = bl*K0[0,0]/reg_flow_PP[-1].reshape(shape)[mod_mask]
            
    idepth = bl*K0[0,0] / reg_flow_P[-1].reshape(shape)
    return idepth,flow, disp1

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def extract_trajectory(cams_gt):
    # world matrix of the camera object: point from world to current frame
    cam_traj_gt = []
    for cam in cams_gt:
        cam_pos_gt = cams_gt[0].dot(np.linalg.inv(cam))[:3,-1]
        cam_traj_gt.append(cam_pos_gt)
    cam_traj_gt = np.stack(cam_traj_gt)
    return cam_traj_gt


def extract_delta(cams_gt):
    # world matrix of the camera object: point from world to current frame
    cam_traj_gt = [np.zeros(3)]
    for i,cam in enumerate(cams_gt):
        if i==0:continue
        cam_traj_gt.append(cams_gt[i-1].dot(np.linalg.inv(cam))[:3,-1])
    cam_traj_gt = np.stack(cam_traj_gt)
    return cam_traj_gt

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy().astype(np.float32)
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def lin_interp(shape, xyd):
    import scipy
    import scipy.interpolate.LinearNDInterpolator as LinearNDInterpolator
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def colmap_cam_read(auxdir,framename):
    K = np.eye(3)
    with open(auxdir, 'r') as f:
        lines = f.readlines()
    if len(lines) == 4:
        # shared intrinsics
        _,_,_,_,fl, cx, cy, _ = lines[-1].split(' ')
        K[0,0] = fl
        K[1,1] = fl
        K[0,2] = cx
        K[1,2] = cy
        return K
