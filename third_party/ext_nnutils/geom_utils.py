"""
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import pdb
import soft_renderer as sr

def sample_textures(texture_flow, images):
    """
    texture_flow: B x F x T x T x 2
    (In normalized coordinate [-1, 1])
    images: B x 3 x N x N

    output: B x F x T x T x 3
    """
    # Reshape into B x F x T*T x 2
    T = texture_flow.size(-2)
    F = texture_flow.size(1)
    flow_grid = texture_flow.view(-1, F, T * T, 2)
    # B x 3 x F x T*T
    samples = torch.nn.functional.grid_sample(images, flow_grid, align_corners=True)
    # B x 3 x F x T x T
    samples = samples.view(-1, 3, F, T, T)
    # B x F x T x T x 3
    return samples.permute(0, 2, 3, 4, 1)

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def orthographic_proj(X, cam):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    return scale * X_rot[:, :, :2] + trans

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    #pdb.set_trace()
    #import kornia
    #X_rot = kornia.quaternion_to_rotation_matrix(torch.cat((quat[:,1:],quat[:,:1]),1)).matmul(X.permute(0,2,1)).permute(0,2,1)
    #assert((X_rot-quat_rotate(X, quat)).abs().mean()<1e-6)
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z

    return torch.cat((proj_xy, proj_z), 2)


def cross_product(qa, qb):
    """Cross product of va by vb.

    Args:
        qa: B X N X 3 vectors
        qb: B X N X 3 vectors
    Returns:
        q_mult: B X N X 3 vectors
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]

    # See https://en.wikipedia.org/wiki/Cross_product
    q_mult_0 = qa_1*qb_2 - qa_2*qb_1
    q_mult_1 = qa_2*qb_0 - qa_0*qb_2
    q_mult_2 = qa_0*qb_1 - qa_1*qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2], dim=-1)


def hamilton_product(qa, qb):
    """Multiply qa by qb.

    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    
    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0
    
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

    
def quat_rotate(X, q):
    """Rotate points by quaternions.

    Args:
        X: B X N X 3 points
        q: B X 4 quaternions

    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)
    
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


# project w/ intrinsicss
def pinhole_cam(verts,pp,fl):
    n_hypo = verts.shape[0] // pp.shape[0]
    pp = pp.clone().reshape(-1,2)
    fl = fl.clone().reshape(-1,2)
    verts = verts.clone()
    verts[:,:,1] = pp[:,1:2]+verts[:, :, 1].clone()*fl[:,1:2]/ verts[:,:,2].clone()
    verts[:,:,0] = pp[:,0:1]+verts[:, :, 0].clone()*fl[:,0:1]/ verts[:,:,2].clone()
#    verts[:,:,2] = ( (verts[:,:,2]-verts[:,:,2].min())/(verts[:,:,2].max()-verts[:,:,2].min())-0.5).detach()
    return verts

def orthographic_cam(verts,pp,fl):
    n_hypo = verts.shape[0] // pp.shape[0]
    pp = pp.clone()[:,None].repeat(1,n_hypo,1).view(-1,2)
    fl = fl.clone()[:,None].view(-1,2)
    verts = verts.clone()
    verts[:,:,1] = pp[:,1:2]+verts[:, :, 1].clone()*fl[:,1:2]
    verts[:,:,0] = pp[:,0:1]+verts[:, :, 0].clone()*fl[:,0:1]
    return verts
            
def raycast(opts,ppoint,scale,Rmat,Tmat,bs):
    from nerf import meshgrid_xy
    ii, jj = meshgrid_xy(torch.arange(opts.img_size),torch.arange(opts.img_size))
    homogrid = torch.cat([ii.reshape(1,-1)/(opts.img_size-1)*2-1,
                          jj.reshape(1,-1)/(opts.img_size-1)*2-1]).cuda() # x,y,1
    homogrid = homogrid[None,None] # bs, hypo, 3,N
    homogrid = homogrid - ppoint[:,None,:,None] # bs,2
    homogrid = homogrid / scale[:,:,None,None]  # x/z, y/z, 1 -> x,y,z
    homogrid = torch.cat([homogrid, torch.ones_like(homogrid[:,:,:1])],2)
    nerf_rmat = Rmat.view(bs,opts.n_hypo,-1,3,3)[:,:,:1].view(-1,3,3) # bs*hypo
    #nerf_rmat = Rmat.view(bs,opts.n_hypo,-1,3,3)[:,:,:1].view(-1,3,3).permute(0,2,1) # bs*hypo
    nerf_tmat = Tmat.view(bs,-1,3)[:,:1].reshape(-1,3,1) # bs
    ray_directions = nerf_rmat.matmul(homogrid.view(-1,3,opts.img_size**2)).view(bs,opts.n_hypo,3,-1)  # bs, hypo, 3,N
    ray_origins = -nerf_rmat.matmul(nerf_tmat.repeat(opts.n_hypo,1,1)).repeat(1,1,opts.img_size**2).view(bs,opts.n_hypo,3,-1)
    ray_origins = ray_origins.permute(0,1,3,2)
    ray_directions = ray_directions.permute(0,1,3,2)
    return ray_origins, ray_directions, nerf_tmat[:,-1]

def obj_to_cam(verts, Rmat, Tmat,nmesh,n_hypo,skin,tocam=True,scores=None):
    verts = verts.clone()
    Rmat = Rmat.clone()
    Tmat = Tmat.clone()
    #verts = verts[:,None].repeat(1,n_hypo,1,1).view(-1,verts.shape[1],3)
    verts = verts.view(-1,verts.shape[1],3)

    #if scores is not None:
    #    scores = scores.softmax(0)[None,:,None,None].double().detach()
    #    scores = scores/scores.max()
    #    verts = (verts.view(-1,n_hypo,verts.shape[1],3)*scores + verts.detach().view(-1,n_hypo,verts.shape[1],3)*(1-scores)).view(-1,verts.shape[1],3).float()
    bodyR = Rmat[::nmesh].clone()
    bodyT = Tmat[::nmesh].clone()
    #Rmat = torch.eye(3).repeat(Rmat.shape[0],1,1).cuda()
    if nmesh>1:
        vs = []
        for k in range(nmesh-1):
            partR = Rmat[k+1::nmesh].clone()
            partT = Tmat[k+1::nmesh].clone()
            vs.append( (verts.matmul(partR) + partT)[:,np.newaxis] )
        vs = torch.cat(vs,1) # N, K, Nv, 3
        vs = (vs * skin).sum(1)
    else:
        vs = verts
    
    if tocam:
        vs =  vs.clone().matmul(bodyR) + bodyT
    else:
        vs = vs.clone()
    return vs
        
# render camera multiplex
def render_multiplex(pred_v, faces, tex, Rmat, Tmat, skin, ppoint, scale, renderer, n_hypo, n_mesh, texture_type='vertex'):
    Rmat_tex = Rmat.view(-1,3,3)
    Tmat_tex = Tmat[:,np.newaxis, :]
    verts_tex = obj_to_cam(pred_v, Rmat_tex, Tmat_tex, n_mesh, n_hypo,skin)
    verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
    verts_tex = pinhole_cam(verts_tex, ppoint, scale)
    offset = torch.Tensor( renderer.transform.transformer._eye ).cuda()[None,None]
    verts_pre = verts_tex[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    renderer.rasterizer.background_color = [1,1,1]
    self_rd = renderer.render_mesh(sr.Mesh(verts_pre, faces, textures=tex,  texture_type=texture_type))
    return self_rd

def rotation_multiplex(num_pts, hemisphere=True):
    # The golden spiral method        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    from numpy import pi, cos, sin, arccos, arange
    indices = arange(0, num_pts, dtype=float) + 0.5
    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    hypo_normals = np.vstack([cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)]).T
    if hemisphere: 
        hypo_normals = hypo_normals[num_pts//2:]
    if hemisphere:
        rotations = np.asarray([R_2vect([1,0,0], i) for i in hypo_normals])
    else:
        rotations = np.asarray([R_2vect([0,0,1], i) for i in hypo_normals])
    return rotations

def R_2vect(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """
    from math import acos, atan2, cos, pi, sin
    from numpy import array, cross, dot, float64, hypot, zeros
    from numpy.linalg import norm
    R = np.zeros((3,3))

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)
    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R

def label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray(
[[151, 119, 169],
 [ 43, 215,  34],
 [251, 250,  67],
 [ 80, 246, 250],
 [ 30,  49, 108],
 [224,  25,  53],
 [224, 210, 228],
 [209,  55, 250],
 [ 55,  85, 242],
 [113,  90,  39],
 [ 28, 175, 141],
 [130, 227, 105],
 [204, 140,  80],
 [ 82,  28, 168],
 [176, 220,  33],
 [111, 154, 243],
 [235,  35, 163],
 [252, 176, 153],
 [ 72, 145,  81],
 [179, 241, 168],
 [150,  42,  93],
 [187, 127, 236],
 [ 54, 229, 180],
 [153,  37, 209],
 [250, 167,  27],
 [ 84, 253,  64],
 [135, 206, 212],
 [148, 177,  66],
 [106,  79, 124],
 [ 65, 145, 187],
 [244,  95, 105],
 [227, 143, 195],
 [ 24, 102, 161],
 [102,  38, 251],
 [ 43, 124,  25],
 [195, 195, 117],
 [ 26, 174, 251],
 [ 38, 243, 109],
 [177, 107, 115],
 [ 60,  30,  64],
 [113, 178, 163],
 [105,  85, 206],
 [166, 181, 249],
 [ 38,  37, 213],
 [252, 247, 206],
 [140,  51, 156],
 [195,  98,  40],
 [251, 156, 245],
 [122, 246, 185],
 [129, 122,  81],]
 #    [[0, 0, 0],
 #     [120, 120, 120],
 #     [180, 120, 120],
 #     [6, 230, 230],
 #     [80, 50, 50],
 #     [4, 200, 3],
 #     [120, 120, 80],
 #     [140, 140, 140],
 #     [204, 5, 255],
 #     [230, 230, 230],
 #     [4, 250, 7],
 #     [224, 5, 255],
 #     [235, 255, 7],
 #     [150, 5, 61],
 #     [120, 120, 70],
 #     [8, 255, 51],
 #     [255, 6, 82],
 #     [143, 255, 140],
 #     [204, 255, 4],
 #     [255, 51, 7],
 #     [204, 70, 3],
 #     [0, 102, 200],
 #     [61, 230, 250],
 #     [255, 6, 51],
 #     [11, 102, 255],
 #     [255, 7, 71],
 #     [255, 9, 224],
 #     [9, 7, 230],
 #     [220, 220, 220],
 #     [255, 9, 92],
 #     [112, 9, 255],
 #     [8, 255, 214],
 #     [7, 255, 224],
 #     [255, 184, 6],
 #     [10, 255, 71],
 #     [255, 41, 10],
 #     [7, 255, 255],
 #     [224, 255, 8],
 #     [102, 8, 255],
 #     [255, 61, 6],
 #     [255, 194, 7],
 #     [255, 122, 8],
 #     [0, 255, 20],
 #     [255, 8, 41],
 #     [255, 5, 153],
 #     [6, 51, 255],
 #     [235, 12, 255],
 #     [160, 150, 20],
 #     [0, 163, 255],
 #     [140, 140, 140],
 #     [250, 10, 15],
 #     [20, 255, 0],
 #     [31, 255, 0],
 #     [255, 31, 0],
 #     [255, 224, 0],
 #     [153, 255, 0],
 #     [0, 0, 255],
 #     [255, 71, 0],
 #     [0, 235, 255],
 #     [0, 173, 255],
 #     [31, 0, 255],
 #     [11, 200, 200],
 #     [255, 82, 0],
 #     [0, 255, 245],
 #     [0, 61, 255],
 #     [0, 255, 112],
 #     [0, 255, 133],
 #     [255, 0, 0],
 #     [255, 163, 0],
 #     [255, 102, 0],
 #     [194, 255, 0],
 #     [0, 143, 255],
 #     [51, 255, 0],
 #     [0, 82, 255],
 #     [0, 255, 41],
 #     [0, 255, 173],
 #     [10, 0, 255],
 #     [173, 255, 0],
 #     [0, 255, 153],
 #     [255, 92, 0],
 #     [255, 0, 255],
 #     [255, 0, 245],
 #     [255, 0, 102],
 #     [255, 173, 0],
 #     [255, 0, 20],
 #     [255, 184, 184],
 #     [0, 31, 255],
 #     [0, 255, 61],
 #     [0, 71, 255],
 #     [255, 0, 204],
 #     [0, 255, 194],
 #     [0, 255, 82],
 #     [0, 10, 255],
 #     [0, 112, 255],
 #     [51, 0, 255],
 #     [0, 194, 255],
 #     [0, 122, 255],
 #     [0, 255, 163],
 #     [255, 153, 0],
 #     [0, 255, 10],
 #     [255, 112, 0],
 #     [143, 255, 0],
 #     [82, 0, 255],
 #     [163, 255, 0],
 #     [255, 235, 0],
 #     [8, 184, 170],
 #     [133, 0, 255],
 #     [0, 255, 92],
 #     [184, 0, 255],
 #     [255, 0, 31],
 #     [0, 184, 255],
 #     [0, 214, 255],
 #     [255, 0, 112],
 #     [92, 255, 0],
 #     [0, 224, 255],
 #     [112, 224, 255],
 #     [70, 184, 160],
 #     [163, 0, 255],
 #     [153, 0, 255],
 #     [71, 255, 0],
 #     [255, 0, 163],
 #     [255, 204, 0],
 #     [255, 0, 143],
 #     [0, 255, 235],
 #     [133, 255, 0],
 #     [255, 0, 235],
 #     [245, 0, 255],
 #     [255, 0, 122],
 #     [255, 245, 0],
 #     [10, 190, 212],
 #     [214, 255, 0],
 #     [0, 204, 255],
 #     [20, 0, 255],
 #     [255, 255, 0],
 #     [0, 153, 255],
 #     [0, 41, 255],
 #     [0, 255, 204],
 #     [41, 0, 255],
 #     [41, 255, 0],
 #     [173, 0, 255],
 #     [0, 245, 255],
 #     [71, 0, 255],
 #     [122, 0, 255],
 #     [0, 255, 184],
 #     [0, 92, 255],
 #     [184, 255, 0],
 #     [0, 133, 255],
 #     [255, 214, 0],
 #     [25, 194, 194],
 #     [102, 255, 0],
 #     [92, 0, 255],]
  )
#  colormap = np.zeros((256, 3), dtype=np.uint8)
#  colormap[0] = [128, 64, 128]
#  colormap[1] = [255, 0, 0]
#  colormap[2] = [0, 255, 0]
#  colormap[3] = [250, 250, 0]
#  colormap[4] = [0, 215, 230]
#  colormap[5] = [190, 153, 153]
#  colormap[6] = [250, 170, 30]
#  colormap[7] = [102, 102, 156]
#  colormap[8] = [107, 142, 35]
#  colormap[9] = [152, 251, 152]
#  colormap[10] = [70, 130, 180]
#  colormap[11] = [220, 20, 60]
#  colormap[12] = [0, 0, 230]
#  colormap[13] = [0, 0, 142]
#  colormap[14] = [0, 0, 70]
#  colormap[15] = [0, 60, 100]
#  colormap[16] = [0, 80, 100]
#  colormap[17] = [244, 35, 232]
#  colormap[18] = [119, 11, 32]
#  return colormap
