import pdb
import numpy as np
from pytorch3d import transforms
import torch
import torch.nn as nn
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

def vec_to_sim3(vec):
    """
    vec:      ...,10
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:10].exp()
    return center, orient, scale
    

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
   
    center, orient, scale = vec_to_sim3(bones) 
    orient = orient.permute(0,1,3,2) # transpose R

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

def lbs(bones, rts_fw, xyz_in, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,7       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    bones = bones.view(-1,B,10)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,7)# B,7
    rquat=rts_fw[:,:,:4]
    rmat=transforms.quaternion_to_matrix(rquat) 
    tmat= rts_fw[:,:,4:7]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)

    if backward:
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz, skin = blend_skinning(bones_dfm, rts_bw, xyz_in)
    else:
        xyz, skin = blend_skinning(bones.repeat(bs,1,1), rts_fw, xyz_in)
        bones_dfm = bone_transform(bones, rts_fw) # bone coordinates after deform
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
    shape=Kmat.shape[:-2]
    Kmat = Kmat.view(-1,3,3)
    device = Kmat.device
    bs = Kmat.shape[0]

    K = torch.zeros(bs, 4, device=device)
    K[:,0] = Kmat[:,0,0]
    K[:,1] = Kmat[:,1,1]
    K[:,2] = Kmat[:,0,2]
    K[:,3] = Kmat[:,1,2]
    K = K.view(shape+(4,))
    return K

def Kmatinv(Kmat):
    """
    Kmat: ...,3,3
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.view(Kmat.shape)
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
    in_verts: ...,N,3
    K:        ...,4
    verts:    ...,N,3 in (x,y,Z)
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    K = K.view(-1,4)

    Kmat = K2mat(K)
    Kmat = Kmat.permute(0,2,1)

    verts = verts.matmul(Kmat)
    verts_z = verts[:,:,2:3]
    verts_xy = verts[:,:,:2] / (1e-6+verts_z) # deal with neg z
    
    verts = torch.cat([verts_xy,verts_z],-1)
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

def tensor2array(tdict):
    adict={}
    for k,v in tdict.items():
        adict[k] = v.detach().cpu().numpy()
    return adict

def raycast(xys, Rmat, Tmat, Kinv, near_far):
    """
    xys: bs, N, 3
    Rmat:bs, ...,3,3 
    Tmat:bs, ...,3, camera to root coord transform 
    Kinv:bs, ...,3,3 
    """
    Rmat, Tmat, Kinv, xys = force_type([Rmat, Tmat, Kinv, xys])
    Rmat = Rmat.view(-1,3,3)
    Tmat = Tmat.view(-1,1,3)
    Kinv = Kinv.view(-1,3,3)
    bs,nsample,_ = xys.shape
    device = Rmat.device

    xy1s = torch.cat([xys, torch.ones_like(xys[:,:,:1])],2)
    xyz3d = xy1s.matmul(Kinv.permute(0,2,1))
    ray_directions = xyz3d.matmul(Rmat)  # transpose -> right multiply
    ray_origins = -Tmat.matmul(Rmat) # transpose -> right multiply

    if near_far is not None:
        znear= torch.ones(bs,nsample,1).to(device) *  near_far[0]
        zfar = torch.ones(bs,nsample,1).to(device) *  near_far[1]
    else:
        #TODO need a better way to bound raycast
        lbound, ubound=[-1.5,1.5]
        #lbound, ubound=[-5, 5]

        znear= Tmat[:,:,-1:].repeat(1,nsample,1)+lbound
        zfar = Tmat[:,:,-1:].repeat(1,nsample,1)+ubound
        znear[znear<1e-5]=1e-5

    ray_origins = ray_origins.repeat(1,nsample,1)

    rmat_vec = Rmat.reshape(-1,1,9)
    tmat_vec = Tmat.reshape(-1,1,3)
    kinv_vec = Kinv.reshape(-1,1,9)
    rtk_vec = torch.cat([rmat_vec, tmat_vec, kinv_vec],-1) # x,21
    rtk_vec = rtk_vec.repeat(1,nsample,1)

    rays={'rays_o': ray_origins, 
          'rays_d': ray_directions,
          'near': znear,
          'far': zfar,
          'rtk_vec': rtk_vec,
          'nsample': nsample,
          'bs': bs,
          }
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

def chunk_rays(rays,start,delta):
    """
    rays: a dictionary
    """
    rays_chunk = {}
    for k,v in rays.items():
        if torch.is_tensor(v):
            v = v.view(-1, v.shape[-1])
            rays_chunk[k] = v[start:start+delta]
    return rays_chunk
        

def generate_bones(num_bones_x, num_bones, bound, device):
    """
    num_bones_x: bones along one direction
    bones: x**3,9
    """
    center =  torch.linspace(-bound, bound, num_bones_x).to(device)
    center =torch.meshgrid(center, center, center)
    center = torch.stack(center,0).permute(1,2,3,0).reshape(-1,3)
    center = center[:num_bones]
    
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)
    return bones

def reinit_bones(model, mesh):
    """
    num_bones: number of bones on the surface
    mesh: trimesh
    """
    from kmeans_pytorch import kmeans
    num_bones = model.num_bones
    device = model.device
    points = torch.Tensor(mesh.vertices).to(device)
    rthead = model.nerf_bone_rts[1].rgb
    
    num_bones = num_bones//2  # reduce bones
    
    # reinit
    num_in = rthead[0].weight.shape[1]
    rthead = nn.Sequential(nn.Linear(num_in, 7*num_bones)).to(device)
    torch.nn.init.xavier_uniform_(rthead[0].weight, gain=0.5)
    torch.nn.init.zeros_(rthead[0].bias)

    _, center = kmeans(X=points, num_clusters=num_bones, 
                       distance='euclidean', device=device)
    center=center.to(device)
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)
    
    model.bones = nn.Parameter(bones)
    model.nerf_models['bones'] = model.bones
    model.num_bones = num_bones
    model.nerf_bone_rts[1].rgb = rthead
    return
            
def warp_bw(opts, model, rt_dict, query_xyz_chunk, frameid):
    """
    only used in mesh extraction
    """
    chunk = query_xyz_chunk.shape[0]
    query_time = torch.ones(chunk,1).to(model.device)*frameid
    query_time = query_time.long()
    if opts.flowbw:
        # flowbw
        xyz_embedded = model.embedding_xyz(query_xyz_chunk)
        time_embedded = model.embedding_time(query_time)[:,0]
        xyztime_embedded = torch.cat([xyz_embedded, time_embedded],1)

        flowbw_chunk = model.nerf_flowbw(xyztime_embedded)
        query_xyz_chunk += flowbw_chunk
    elif opts.lbs:
        # backward skinning
        bones = model.bones
        query_xyz_chunk = query_xyz_chunk[:,None]
        bone_rts_fw = model.nerf_bone_rts(query_time)

        query_xyz_chunk,_,bones_dfm = lbs(bones, 
                                      bone_rts_fw,
                                      query_xyz_chunk)

        query_xyz_chunk = query_xyz_chunk[:,0]
        rt_dict['bones'] = bones_dfm 
    return query_xyz_chunk, rt_dict
        
def warp_fw(opts, model, rt_dict, vertices, frameid):
    """
    only used in mesh extraction
    """
    num_pts = vertices.shape[0]
    query_time = torch.ones(num_pts,1).long().to(model.device)*frameid
    pts_can=torch.Tensor(vertices).to(model.device)
    if opts.flowbw:
        # forward flow
        pts_can_embedded = model.embedding_xyz(pts_can)
        time_embedded = model.embedding_time(query_time)[:,0]
        ptstime_embedded = torch.cat([pts_can_embedded, time_embedded],1)

        pts_dfm = pts_can + model.nerf_flowfw(ptstime_embedded)
    elif opts.lbs:
        # forward skinning
        bones = model.bones
        pts_can = pts_can[:,None]
        bone_rts_fw = model.nerf_bone_rts(query_time)

        pts_dfm,_,bones_dfm = lbs(bones, bone_rts_fw, pts_can,backward=False)
        pts_dfm = pts_dfm[:,0]
        rt_dict['bones'] = bones_dfm
    vertices = pts_dfm.cpu().numpy()
    return vertices, rt_dict
    
def canonical2ndc(model, dp_canonical_pts, rtk, kaug, frameid):
    """
    dp_canonical_pts: 5004,3, pts in the canonical space of each video
    dp_px: bs, 5004, 3
    """
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    Kinv = Kmatinv(Kaug.matmul(Kmat))
    K = mat2K(Kmatinv(Kinv))
    bs = Kinv.shape[0]
    npts = dp_canonical_pts.shape[0]

    # projection
    dp_canonical_pts = dp_canonical_pts[None]
    if model.opts.flowbw:
        time_embedded = model.embedding_time(frameid)
        time_embedded = time_embedded.repeat(1,npts, 1)
        dp_canonical_embedded = model.embedding_xyz(dp_canonical_pts)[None]
        dp_canonical_embedded = dp_canonical_embedded.repeat(bs,1,1)
        dp_canonical_embedded = torch.cat([dp_canonical_embedded, time_embedded], -1)
        dp_deformed_flo = model.nerf_flowfw(dp_canonical_embedded)
        dp_deformed_pts = dp_canonical_pts + dp_deformed_flo
    else:
        dp_deformed_pts = dp_canonical_pts.repeat(bs,1,1)
    dp_cam_pts = obj_to_cam(dp_deformed_pts, Rmat, Tmat) 
    dp_px = pinhole_cam(dp_cam_pts,K)
    return dp_px 
