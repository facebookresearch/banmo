import pdb
import torch
from nnutils.nerf import evaluate_mlp

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
