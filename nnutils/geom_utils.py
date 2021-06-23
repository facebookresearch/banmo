import pdb
import numpy as np
import torch
import soft_renderer as sr

def obj_to_cam(in_verts, Rmat, Tmat):
    """
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
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
    verts: ...,N,3/4
    faces: ...,N,3
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    if texture_type=='vertex':  colors = colors.view(-1,colors.shape[1],3)
    elif texture_type=='surface': colors = colors.view(-1,colors.shape[1],colors.shape[2],3)
    device=verts.device

    offset = torch.Tensor( renderer.transform.transformer._eye).to(device)[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset
    rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type=texture_type))
    return rendered

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
