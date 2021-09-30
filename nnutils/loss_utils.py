import pdb
import torch
from nnutils.nerf import evaluate_mlp
from nnutils.geom_utils import rot_angle
import torch.nn.functional as F

def nerf_gradient(mlp, embed, pts, use_xyz=False,code=None, sigma_only=False):
    """
    gradient of mlp params wrt pts
    """
    pts.requires_grad_(True)
    pts_embedded = embed(pts)
    if use_xyz: xyz=pts
    else: xyz=None
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], 
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
