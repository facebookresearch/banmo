import pdb
import numpy as np
from pytorch3d import transforms
import torch
import torch.nn.functional as F
import soft_renderer as sr

def bone_transform(bones_in, rts):
    """ 
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B ririd transforms
    rts are transformation applied to bone coordinates (right multiply)
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1,B,10).clone()
    rts = rts.view(-1,B,3,4)
    bs = rts.shape[0] 

    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    scale =  bones[:,:,7:10]
    Rmat = rts[:,:,:3,:3]   
    Tmat = rts[:,:,:3,3:4]   

    center = transforms.quaternion_to_matrix(orient).matmul(Tmat)[...,0]+center
    Rquat = transforms.matrix_to_quaternion(Rmat)
    orient = transforms.quaternion_multiply(orient, Rquat)

    scale = scale.repeat(bs,1,1)
    bones = torch.cat([center,orient,scale],-1)
    return bones 

def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1,3,4).clone()
    Rmat = rts[:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:3,3:]
    Rmat_i=Rmat.permute(0,2,1)
    Tmat_i=-Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i],-1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i

def skinning(bones, pts):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    B = bones.shape[-2]
    N = pts.shape[-2]
    bones = bones.view(-1,B,10)
    pts = pts.view(-1,N,3)
    bs = pts.shape[0]
    
    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    orient = orient.permute(0,1,3,2) # transpose R
    scale =  bones[:,:,7:10].exp()

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3
    mdis = orient.view(bs,1,B,3,3).matmul(mdis[...,None]) # bs,N,B,3,1
    mdis = mdis[...,0]
    mdis = scale.view(bs,1,B,3) * mdis.pow(2)
    mdis = (-10 * mdis.sum(3)) # bs,N,B
    
    # truncated softmax
    topk, indices = mdis.topk(3, 2, largest=True)
    mdis = torch.zeros_like(mdis).fill_(-np.inf)
    mdis = mdis.scatter(2, indices, topk)
    skin = mdis.softmax(2)
    return skin

def blend_skinning(bones, rts, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = rts.shape[-3]
    N = pts.shape[-2]
    bones = bones.view(-1,B,10)
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]
    device = Tmat.device

    # convert from bone to root transforms
    bs = Rmat.shape[0]
    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    gmat = torch.eye(4)[None,None].repeat(bs, B, 1, 1).to(device)
    
    # root to bone
    gmat_r2b = gmat.clone()
    gmat_r2b[:,:,:3,:3] = orient.permute(0,1,3,2)
    gmat_r2b[:,:,:3,3] = -orient.permute(0,1,3,2).matmul(center[...,None])[...,0]
   
    # bone to root
    gmat_b2r = gmat.clone()
    gmat_b2r[:,:,:3,:3] = orient
    gmat_b2r[:,:,:3,3] = center

    # bone to bone  
    gmat_b = gmat.clone()
    gmat_b[:,:,:3,:3] = Rmat
    gmat_b[:,:,:3,3] = Tmat
   
    gmat = gmat_b2r.matmul(gmat_b.matmul(gmat_r2b))
    Rmat = gmat[:,:,:3,:3]
    Tmat = gmat[:,:,:3,3]
    
    # compute skinning weight
    skin = skinning(bones, pts) # bs, N, B

    # Gi=sum(wbGb), V=RV+T
    Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3
    Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3
    pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    pts = pts[...,0]
    return pts, skin

def blend_skinning_bw(bones, rts_fw, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids indicating bone coordinates
    rts: bs,B,3,4   - B rigid transforms, applied to the bone coordinate
    pts: bs,N,3     - N 3d points
    """
    B = rts_fw.shape[-3]
    N = pts.shape[-2]
    bones = bones.view(-1,B,10)
    rts_fw = rts_fw.view(-1,B,3,4)
    pts = pts.view(-1,N,3)
    
    bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
    rts_bw = rts_invert(rts_fw)
    pts, skin = blend_skinning(bones_dfm, rts_bw, pts)
    return pts, skin, bones_dfm


def lbs(bones, embedding_time, xyz, frameid):
    B = bones.shape[-2]
    time_embedded = embedding_time(frameid.long()) 
    time_embedded = time_embedded.view(-1,B,7)# B,7
    #time_embedded = model_rts(time_embedded)[:,:,:-1]
    rquat=time_embedded[:,:,:4]
    tmat= time_embedded[:,:,4:7] * 0.1

    rquat[:,:,0]+=10
    rquat=F.normalize(rquat,2,2)
    rmat=transforms.quaternion_to_matrix(rquat) 

    # original orientation
    #bones=torch.cat([bones[:,:4], 
    #                torch.zeros(B,6).to(bones.device)],-1)
    # no bone rotation
    #rmat=torch.eye(3).to(rquat.device).view(1,1,3,3).repeat(rquat.shape[0],B,1,1)

    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    xyz, skin, bones_dfm = blend_skinning_bw(bones, rts_fw, xyz)
    return xyz, skin, bones_dfm

def obj_to_cam(in_verts, Rmat, Tmat):
    """
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    Rmat = Rmat.view(-1,3,3).permute(0,2,1) # left multiply
    Tmat = Tmat.view(-1,1,3)
    
    verts =  verts.matmul(Rmat) + Tmat 
    verts = verts.reshape(in_verts.shape)
    return verts
    
def K2mat(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = K[:,0]
    Kmat[:,1,1] = K[:,1]
    Kmat[:,0,2] = K[:,2]
    Kmat[:,1,2] = K[:,3]
    Kmat[:,2,2] = 1
    return Kmat

def mat2K(Kmat):
    """
    Kmat: ...,3,3
    """
    Kmat = Kmat.view(-1,3,3)
    device = Kmat.device
    bs = Kmat.shape[0]

    K = torch.zeros(bs, 4, device=device)
    K[:,0] = Kmat[:,0,0]
    K[:,1] = Kmat[:,1,1]
    K[:,2] = Kmat[:,0,2]
    K[:,3] = Kmat[:,1,2]
    return K

def Kmatinv(Kmat):
    """
    Kmat: ...,3,3
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    return Kmatinv

def K2inv(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = 1./K[:,0]
    Kmat[:,1,1] = 1./K[:,1]
    Kmat[:,0,2] = -K[:,2]/K[:,0]
    Kmat[:,1,2] = -K[:,3]/K[:,1]
    Kmat[:,2,2] = 1
    return Kmat

def pinhole_cam(in_verts, K):
    """
    verts: ...,N,3
    K:     ...,4
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    K = K.view(-1,4)

    Kmat = K2mat(K)
    Kmat = Kmat.permute(0,2,1)

    verts = verts.matmul(Kmat)
    verts[:,:,:2] /= verts[:,:,2:3]
    verts = verts.reshape(in_verts.shape)
    return verts

def render_color(renderer, in_verts, faces, colors, texture_type='vertex'):
    """
    verts in ndc
    in_verts: ...,N,3/4
    faces: ...,N,3
    rendered: ...,4,...
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    if texture_type=='vertex':  colors = colors.view(-1,colors.shape[1],3)
    elif texture_type=='surface': colors = colors.view(-1,colors.shape[1],colors.shape[2],3)
    device=verts.device

    offset = torch.Tensor( renderer.transform.transformer._eye).to(device)[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset
    verts_pre[:,:,1] = -1*verts_pre[:,:,1]  # pre-flip
    rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type=texture_type))
    return rendered

def render_flow(renderer, verts, faces, verts_n):
    """
    verts in ndc
    verts: ...,N,3/4
    verts_n: ...,N,3/4
    faces: ...,N,3
    """
    verts = verts.view(-1,verts.shape[1],3)
    verts_n = verts_n.view(-1,verts_n.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    device=verts.device

    rendered_ndc_n = render_color(renderer, verts, faces, verts_n)
    _,_,h,w = rendered_ndc_n.shape
    rendered_sil = rendered_ndc_n[:,-1]

    ndc = np.meshgrid(range(w), range(h))
    ndc = torch.Tensor(ndc).to(device)[None]
    ndc[:,0] = ndc[:,0]*2 / (w-1) - 1
    ndc[:,1] = ndc[:,1]*2 / (h-1) - 1

    flow = rendered_ndc_n[:,:2] - ndc
    flow = flow.permute(0,2,3,1) # x,h,w,2
    flow = torch.cat([flow, rendered_sil[...,None]],-1)

    flow[rendered_sil<1]=0.
    flow[...,-1]=0. # discard the last channel
    return flow

def force_type(varlist):
    for i in range(len(varlist)):
        varlist[i] = varlist[i].type(varlist[0].dtype)
    return varlist

def raycast(xys, Rmat, Tmat, Kinv, bound=1.5):
    """
    xys: bs, N, 3
    Rmat:bs, ...,3,3 
    Tmat:bs, ...,3 
    Kinv:bs, ...,3,3 
    """
    Rmat, Tmat, Kinv, xys = force_type([Rmat, Tmat, Kinv, xys])
    Rmat = Rmat.view(-1,3,3)
    Tmat = Tmat.view(-1,1,3)
    Kinv = Kinv.view(-1,3,3)
    nsamples = xys.shape[1]

    xy1s = torch.cat([xys, torch.ones_like(xys[:,:,:1])],2)
    xyz3d = xy1s.matmul(Kinv.permute(0,2,1))
    ray_directions = xyz3d.matmul(Rmat)  # transpose -> right multiply
    ray_origins = -Tmat.matmul(Rmat) # transpose -> right multiply
    znear =Tmat[:,:,-1:].repeat(1,nsamples,1)-bound
    zfar = Tmat[:,:,-1:].repeat(1,nsamples,1)+bound
    ray_origins = ray_origins.repeat(1,nsamples,1)

    rays = torch.cat([ray_origins, ray_directions, znear, zfar],-1)
    rays = rays.float()
    return rays

def sample_xy(img_size, bs, nsample, device, return_all=False):
    xygrid = np.meshgrid(range(img_size), range(img_size))  # w,h->hxw
    xygrid = torch.Tensor(xygrid).to(device)  # (x,y)
    xygrid = xygrid.permute(1,2,0).reshape(1,-1,2).repeat(bs,1,1) # bs,..., 3
    
    if return_all:
        rand_inds=xygrid.clone()
        xys = xygrid
    else:
        rand_inds = [np.random.choice(img_size**2, size=nsample, replace=False)\
                     for i in range(bs)]
        rand_inds = torch.LongTensor(rand_inds).to(device) # bs, ns
        xys = torch.stack([xygrid[i][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
    
    return rand_inds, xys
