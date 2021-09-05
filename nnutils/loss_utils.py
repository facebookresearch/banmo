import pdb
import torch
from nnutils.nerf import evaluate_mlp

def nerf_gradient(mlp, embed, pts):
    """
    gradient of mlp params wrt pts
    """
    pts.requires_grad_(True)
    pts_embedded = embed(pts)
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], sigma_only=True)
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=pts,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients

def eikonal_loss(mlp, embed, bound, nsample=1000):
    device = next(mlp.parameters()).device
    # Sample points for the eikonal loss
    pts = torch.rand(1,nsample,3)*2*bound-bound
    pts= pts.to(device)

    g = nerf_gradient(mlp, embed, pts)
    eikonal_loss = ((g.norm(2, dim=-1) - 1) ** 2).mean()
    return eikonal_loss

def bone_density_loss(mlp, embed, bones):
    pts = bones[:,:3] 
    pts_embedded = embed(pts)
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], sigma_only=True)
    return bone_density_loss
