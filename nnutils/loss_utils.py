import pdb
import trimesh
import cv2
import numpy as np
import torch
from nnutils.nerf import evaluate_mlp
from nnutils.geom_utils import rot_angle, mat2K, Kmatinv, obj_to_cam, \
                                pinhole_cam, lbs, skinning, mlp_skinning
import torch.nn.functional as F

def nerf_gradient(mlp, embed, pts, use_xyz=False,code=None, sigma_only=False):
    """
    gradient of mlp params wrt pts
    """
    pts.requires_grad_(True)
    pts_embedded = embed(pts)
    if use_xyz: xyz=pts
    else: xyz=None
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts.shape[0], 
            xyz=xyz,code=code,sigma_only=sigma_only)
        
    # get gradient for each size-1 output
    gradients = []
    for i in range(y.shape[-1]):
        y_sub = y [...,i:i+1]
        d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y_sub,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradients.append( gradient[...,None] )
    gradients = torch.cat(gradients,-1) # ...,input-dim, output-dim
    return gradients

def eikonal_loss(mlp, embed, bound, nsample=1000):
    device = next(mlp.parameters()).device
    # Sample points for the eikonal loss
    pts = torch.rand(1,nsample,3)*2*bound-bound
    pts= pts.to(device)

    g = nerf_gradient(mlp, embed, pts, sigma_only=True)
    g = g[...,0]
    eikonal_loss = ((g.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss

def elastic_loss(mlp, embed, xyz, time_embedded):
    xyz = xyz.detach().clone()
    time_embedded = time_embedded.detach().clone()
    g = nerf_gradient(mlp, embed, xyz, use_xyz=mlp.use_xyz,code=time_embedded)
    jacobian = g+torch.eye(3)[None,None].to(g.device)

    sign, log_svals = jacobian.slogdet()
    log_svals = log_svals.clone()
    log_svals[sign<=0] = 0.
    elastic_loss = log_svals**2
    return elastic_loss
    

def bone_density_loss(mlp, embed, bones):
    pts = bones[:,:3] 
    pts_embedded = embed(pts)
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], sigma_only=True)
    return bone_density_loss

def visibility_loss(mlp, embed, xyz_pos, w_pos, bound, chunk):
    """
    w_pos: num_points x num_samples, visibility returns from nerf
    bound: scalar, used to sample negative samples
    """
    device = next(mlp.parameters()).device
    xyz_pos = xyz_pos.detach().clone()
    w_pos = w_pos.detach().clone()
    
    # negative examples
    nsample = w_pos.shape[0]*w_pos.shape[1]
    xyz_neg = torch.rand(1,nsample,3)*2*bound-bound
    xyz_neg = xyz_neg.to(device)
    xyz_neg_embedded = embed(xyz_neg)
    vis_neg_pred = evaluate_mlp(mlp, xyz_neg_embedded, chunk=chunk)[...,0]
    vis_loss_neg = -F.logsigmoid(-vis_neg_pred).sum()*0.1/nsample
      
    # positive examples
    xyz_pos_embedded = embed(xyz_pos)
    vis_pos_pred = evaluate_mlp(mlp, xyz_pos_embedded, chunk=chunk)[...,0]
    vis_loss_pos = -(F.logsigmoid(vis_pos_pred) * w_pos).sum()/nsample

    vis_loss = vis_loss_pos + vis_loss_neg
    return vis_loss

def rtk_loss(rtk, rtk_raw, aux_out):
    rot_pred = rtk[:,:3,:3]
    rot_gt = rtk_raw[:,:3,:3]
    rot_loss = rot_angle(rot_pred.matmul(rot_gt.permute(0,2,1))).mean()
    rot_loss = 0.01*rot_loss

    trn_pred = rtk[:,:3,3]
    trn_gt = rtk_raw[:,:3,3]
    trn_loss = (trn_pred - trn_gt).pow(2).sum(-1).mean()
    total_loss = rot_loss + trn_loss
    aux_out['rot_loss'] = rot_loss
    aux_out['trn_loss'] = trn_loss
    return total_loss

def rtk_cls_loss(scores, grid, rtk_raw, aux_out):
    """
    scores, bs, N
    grid,   bs, N, 3,3
    rtk_raw bs, 4,4
    """
    ##TODO self correlation
    #pdb.set_trace()
    #drot = grid[None].matmul(grid[:,None].permute(0,1,3,2))
    #drot = rot_angle(drot)

    bs,num_score = scores.shape
    rot_gt = rtk_raw[:,None,:3,:3].repeat(1,num_score,1,1)
    grid = grid[None].repeat(bs,1,1,1)
    drot = rot_gt.matmul(grid.permute(0,1,3,2)) # bs, N, 3,3

    ## softmax ce loss
    #drot = rot_angle(drot)
    #target = drot.argmin(1)
    #sce_loss = 0.01*F.cross_entropy(scores, target)
    #
    #total_loss = sce_loss 
    #aux_out['sce_loss'] = sce_loss

    # bce loss
    # positive examples
    w_pos = (-2*rot_angle(drot).pow(2)).exp()
    bce_loss_pos = -(F.logsigmoid(scores) * w_pos).mean()
    
    # negative examples
    w_neg = (1-w_pos)
    w_neg = w_neg/w_neg.mean() * w_pos.mean()
    bce_loss_neg = -(F.logsigmoid(-scores) * w_neg).mean()

    bce_loss = bce_loss_pos + bce_loss_neg
    total_loss = bce_loss 
    aux_out['bce_loss'] = bce_loss

    softmax = torch.softmax(scores, 1)
    aux_out['cls_entropy'] = (-softmax*softmax.log()).sum(1).mean(0)
    return total_loss



def feat_match_loss(nerf_feat, embedding_xyz, feats, pts, pts_prob, bound, 
        is_training=True):
    """
    feats:    bs, ns, num_feat
    pts:      bs, ns, ndepth, 3
    pts_prob: bs, ns, ndepth
    loss:     bs, ns, 1
    """
    # part1: matching
    pts_pred = feat_match(nerf_feat, embedding_xyz, feats, 
            bound,is_training=is_training)

    # part2: compute loss
    bs     = pts_prob.shape[0]
    ndepth = pts_prob.shape[-1]
    pts =           pts.view(-1, ndepth,3)
    pts_prob = pts_prob.view(-1, ndepth,1)
    
    # compute expected pts
    pts_prob = pts_prob.detach()
    pts_prob = pts_prob/(1e-9+pts_prob.sum(1)[:,None])
    pts_exp = (pts * pts_prob).sum(1)

    # loss
    # evaluate against model's opacity distirbution along the ray with soft target
    feat_err = (pts_pred - pts_exp).norm(2,-1) # n,ndepth

    # rearrange outputs
    pts_pred  = pts_pred.view(bs,-1,3)
    pts_exp   = pts_exp .view(bs,-1,3)
    feat_err = feat_err .view(bs,-1,1)
    return pts_pred, pts_exp, feat_err
    
def kp_reproj_loss(pts_pred, xys, models, embeddings, rays):
    """
    pts_pred,   bs, ...,3
    xys,        bs,n,2
    kp reprojection loss is only used to update root/body pose and skinning weights
    """
    bs,ns,_ = xys.shape
    N = bs*ns
    xyz_coarse_sampled = pts_pred.view(-1,1,3)
    # detach grad since reproj-loss would not benefit feature learning 
    # (due to ambiguity)
    xyz_coarse_sampled = xyz_coarse_sampled.detach() 
    xys = xys.view(-1,1,2)

    # TODO wrap flowbw and lbs into the same module
    # TODO include loss for flowbw
    rtk_vec =  rays['rtk_vec']    .view(N,-1) # bs, ns, 21
    if 'bones' in models.keys():
        bone_rts_fw = rays['bone_rts'].view(N,-1) # bs, ns,-1
        if 'nerf_skin' in models.keys():
            nerf_skin = models['nerf_skin']
        else: nerf_skin = None
        bones = models['bones']
        skin_aux = models['skin_aux']
        rest_pose_code = models['rest_pose_code']
        embedding_xyz = embeddings['xyz']

        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones.device))
        rest_pose_code = rest_pose_code[None].repeat(N, 1,1)
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        dskin_fwd = mlp_skinning(nerf_skin, rest_pose_code, xyz_coarse_embedded)
        skin_forward = skinning(bones, xyz_coarse_sampled, 
                            dskin_fwd, skin_aux=skin_aux)
        xyz_coarse_sampled,_ = lbs(bones, bone_rts_fw,
                          skin_forward, xyz_coarse_sampled, backward=False)

    Rmat = rtk_vec[:,0:9]  .view(N,1,3,3)
    Tmat = rtk_vec[:,9:12] .view(N,1,3)
    Kinv = rtk_vec[:,12:21].view(N,1,3,3)
    K = mat2K(Kmatinv(Kinv))

    xyz_coarse_sampled = obj_to_cam( xyz_coarse_sampled, Rmat, Tmat) 
    xyz_coarse_sampled = pinhole_cam(xyz_coarse_sampled,K)
    
    proj_err = (xys - xyz_coarse_sampled[...,:2]).norm(2,-1)
    proj_err = proj_err.view(pts_pred.shape[:-1]+(1,))
    return proj_err
    
    
def feat_match(nerf_feat, embedding_xyz, feats, bound, 
        is_training=True):
    """
    feats:    bs, ns, num_feat
    """
    if is_training: 
        chunk_pts = 8*1024
    else:
        chunk_pts = 1024
    chunk_pix = 200
    grid_size = 20
    bs,N,num_feat = feats.shape
    device = feats.device
    nsample = bs*N
    feats = feats.view(nsample,num_feat)
    feats = F.normalize(feats,2,-1)

    # sample model on a regular 3d grid, and correlate with feature, nkxkxk
    p1d = np.linspace(-bound, bound, grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(p1d, p1d, p1d), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).to(device).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    cost_vol = []
    for i in range(0,grid_size**3,chunk_pts):
        query_xyz_chunk = query_xyz[i:i+chunk_pts]
        xyz_embedded = embedding_xyz(query_xyz_chunk)[:,None] # (N,1,...)
        vol_feat_chunk = evaluate_mlp(nerf_feat, xyz_embedded)[:,0] # (chunk, num_feat)
        # normalize vol feat
        vol_feat_chunk = F.normalize(vol_feat_chunk,2,-1)

        cost_chunk = []
        for j in range(0,nsample,chunk_pix):
            feats_chunk = feats[j:j+chunk_pix] # (chunk pix, num_feat)
            # cpix, cpts
            # distance metric
            cost_subchunk = (vol_feat_chunk[None] * \
                    feats_chunk[:,None]).sum(-1) * (nerf_feat.beta.abs()+1e-9)
            cost_chunk.append(cost_subchunk)
        cost_chunk = torch.cat(cost_chunk,0) # (nsample, cpts)
        cost_vol.append(cost_chunk)
    cost_vol = torch.cat(cost_vol,-1) # (nsample, k**3)
    prob_vol = cost_vol.softmax(-1)

    # regress to the true location, n,3
    if not is_training: torch.cuda.empty_cache()
    pts_pred = (prob_vol[...,None] * query_xyz[None]).sum(1)
    return pts_pred


def grad_update_bone(bones,embedding_xyz, nerf_vis, learning_rate):
    """
    #TODO need to update bones locally
    """
    device = bones.device
    bones_data = bones.data.detach()
    bones_data.requires_grad_(True)
    bone_xyz_embed = embedding_xyz(bones_data[:,None,:3])
    sdf_at_bone = evaluate_mlp(nerf_vis, bone_xyz_embed)
    bone_loc_loss = F.relu(-sdf_at_bone).mean()
    
    # compute gradient wrt bones
    d_output = torch.ones_like(bone_loc_loss, requires_grad=False, device=device)
    gradient = torch.autograd.grad(
        outputs=bone_loc_loss,
        inputs=bones_data,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    bones.data = bones.data-gradient*learning_rate

    return bone_loc_loss
