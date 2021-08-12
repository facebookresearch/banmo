# adopted from nerf-pl
import pdb
import torch
import torch.nn.functional as F
from pytorch3d import transforms
from torchsearchsorted import searchsorted

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        # a hacky way to ensures prob. sum up to 1     
        # while the prob. of last bin does not correspond with the values
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*F.softplus(sigmas+noise)) # (N_rays, N_samples_)
        #alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights, sigmas

        # compute final weighted outputs
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights, sigmas


    # Extract models from lists
    model_coarse = models['coarse']
    embedding_xyz = embeddings['xyz']
    embedding_dir = embeddings['dir']

    # Decompose the inputs
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']  # both (N_rays, 3)
    near = rays['near']
    far = rays['far']  # both (N_rays, 1)
    N_rays = rays_d.shape[0]

    # Embed direction
    rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]
    dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # produce points in the root body space
    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    # root space point correspondence in t2
    xyz_coarse_target = xyz_coarse_sampled.clone()

    ##TODO: produce backward 3d flow and warp to canonical space.
    def evaluate_mlp(model, embedded, code=None, sigma_only=False):
        B,nbins,_ = embedded.shape
        out_chunks = []
        for i in range(0, B, chunk):
            if code is not None:
                embedded = torch.cat([embedded[i:i+chunk],
                           code[i:i+chunk].repeat(1,nbins,1)], -1)
            out_chunks += [model(embedded, sigma_only=sigma_only)]

        out = torch.cat(out_chunks, 0)
        return out

    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
        
        # cycle loss
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, code=time_embedded)
        frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
        # rigidity loss
        frame_disp3d = flow_fw.norm(2,-1)

        ## cycle loss: canodical deformed canonical
        #bound=1 #TODO modif this based on size of canonical volume
        #xyz_can_sampled = torch.rand(xyz_coarse_sampled.shape)*2*bound-bound
        #xyz_can_sampled = xyz_can_sampled.to(xyz_coarse_embedded.device)
        #xyz_can_embedded = embedding_xyz(xyz_can_sampled)
        #sigma_can = evaluate_mlp(model_coarse, xyz_can_embedded, sigma_only=True)
        #weights_sample = sigma_can.sigmoid()[...,0]
        #flow_fw_can = evaluate_mlp(model_flowfw, xyz_can_embedded, code=time_embedded)

        #xyz_dfm_sampled = xyz_can_sampled + flow_fw_can
        #xyz_dfm_embedded = embedding_xyz(xyz_dfm_sampled)
        #flow_bw_dfm = evaluate_mlp(model_flowbw, xyz_dfm_embedded, code=time_embedded)
        #frame_cyc_dis = (flow_bw_dfm+flow_fw_can).norm(2,-1)
        #
        ## rigidity loss
        #frame_disp3d = flow_fw_can.norm(2,-1)

        if "time_embedded_target" in rays.keys():
            time_embedded_target = rays['time_embedded_target'][:,None]
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                    code=time_embedded_target)
            xyz_coarse_target=xyz_coarse_sampled + flow_fw

    elif 'bones' in models.keys():
        # backward skinning
        bones = models['bones']
        bone_rts_fw = rays['bone_rts']
        xyz_coarse_frame = xyz_coarse_sampled.clone()
        xyz_coarse_sampled, skin, bones_dfm = lbs(bones, 
                                                  bone_rts_fw, 
                                                  xyz_coarse_sampled)
        # cycle loss
        xyz_coarse_frame_cyc,_,_ = lbs(bones, bone_rts_fw,
                                       xyz_coarse_sampled,backward=False)
        frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
        
        # rigidity loss
        frame_disp3d = (xyz_coarse_frame_cyc - xyz_coarse_sampled).norm(2,-1)
        
        ## cycle loss: canodical deformed canonical
        #bound=1 #TODO modif this based on size of canonical volume
        #xyz_can_sampled = torch.rand(xyz_coarse_sampled.shape)*2*bound-bound
        #xyz_can_sampled = xyz_can_sampled.to(xyz_coarse_sampled.device)
        #xyz_can_embedded = embedding_xyz(xyz_can_sampled)
        #sigma_can = evaluate_mlp(model_coarse, xyz_can_embedded, sigma_only=True)
        #weights_sample = sigma_can.sigmoid()[...,0]
        #xyz_dfm_sampled,_,_ = lbs(bones, bone_rts_fw, xyz_can_sampled,backward=False)
        #xyz_can_cyc,_,_     = lbs(bones, bone_rts_fw, xyz_dfm_sampled)
        #frame_cyc_dis = (xyz_can_sampled - xyz_can_cyc).norm(2,-1)
        #
        ## rigidity loss
        #frame_disp3d = (xyz_can_sampled - xyz_dfm_sampled).norm(2,-1)
            
        if 'bone_rts_target' in rays.keys():
            bone_rts_target = rays['bone_rts_target']
            xyz_coarse_target,_,_ = lbs(bones, bone_rts_target, 
                                    xyz_coarse_sampled,backward=False)

    if test_time:
        weights_coarse, sigmas = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'sil_coarse': weights_coarse.sum(1)}
    else:
        rgb_coarse, depth_coarse, weights_coarse, sigmas = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'img_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'sil_coarse': weights_coarse.sum(1),
                 }

    # compute correspondence: root space to target view space
    # RT: root space to camera space
    rtk_vec_target =  rays['rtk_vec_target']
    Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
    Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
    Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
    K = mat2K(Kmatinv(Kinv))

    xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
    xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)
                  
    result['xyz_coarse_target'] = xyz_coarse_target
    result['weights_coarse'] = weights_coarse
        
    if 'flowbw' in models.keys() or  'bones' in models.keys():
        result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
        result['frame_disp3d'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
        
        #result['frame_cyc_dis'] = 0.01*(frame_cyc_dis * weights_sample.detach()).sum(-1)
        #result['frame_disp3d'] =  0.01*(frame_disp3d  * weights_sample.detach()).sum(-1)
        
        ### script to plot sigmas/weights
        #from matplotlib import pyplot as plt
        #plt.ioff()
        #plt.plot(weights_coarse[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
        #plt.savefig('weights.png')
        #plt.cla()
        #plt.plot(sigmas[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
        #plt.savefig('sigmas.png')


    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models['fine']
        rgb_fine, depth_fine, weights_fine, sigmas = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['img_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['sil_fine'] = weights_fine.sum(1)

    return result
