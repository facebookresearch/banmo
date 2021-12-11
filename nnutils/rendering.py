# adopted from nerf-pl
import pdb
import torch
import torch.nn.functional as F
from pytorch3d import transforms

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, gauss_mlp_skinning
from nnutils.nerf import evaluate_mlp
from nnutils.loss_utils import elastic_loss, visibility_loss, feat_match_loss,\
                                kp_reproj_loss

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

    inds = torch.searchsorted(cdf, u, right=True)
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
                obj_bound=None,
                use_fine=False,
                img_size=None,
                progress=None,
                opts=None,
                render_vis=False,
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

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    if use_fine: N_samples = N_samples//2 # use half samples to importance sample

    # Extract models from lists
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

    # zvals are not optimized
    # produce points in the root body space
    xyz_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if use_fine: # sample points for fine model
        # output: 
        #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
        #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
        #  not loss:   'depth_rnd', 'joint_render', 'pts_pred', 'pts_exp'
        with torch.no_grad():
            _, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                              chunk, N_samples,
                              N_rays, embedding_xyz, rays_d, noise_std,
                              obj_bound, dir_embedded, z_vals,
                              img_size, progress,opts,fine_iter=False)

        # reset N_importance
        N_importance = N_samples
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        N_samples = N_samples + N_importance # get back to original # of samples
    
    result, _ = inference_deform(xyz_sampled, rays, models, 
                          chunk, N_samples,
                          N_rays, embedding_xyz, rays_d, noise_std,
                          obj_bound, dir_embedded, z_vals,
                          img_size, progress,opts,render_vis=render_vis)

    return result
    
def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, weights_only=False, clip_bound = None, vis_pred=None,
        symm_shape=False):
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
    xyz_input = xyz_.view(N_rays,N_samples,3)
    if symm_shape:
        ##TODO set to x-symmetric here
        xyz_input = torch.cat([xyz_input[...,:1].abs(), xyz_input[...,1:3]],-1)
    out = evaluate_mlp(model, xyz_input, 
            embed_xyz = embedding_xyz,
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
            code=env_code,
            chunk=chunk//N_samples, sigma_only=weights_only).view(B,-1)

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
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(model.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas))
    sigmas = sigmas * ibetas

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

    #set out-of-bound and nonvisible alphas to zero
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
    weights = alphas * alpha_prod # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1
    if weights_only:
        return weights

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    return rgb_final, depth_final, weights, visibility
    
def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress,opts, fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """
    is_training = models['coarse'].training
    if 'sim3_j2c' in rays.keys():
        # similarity transform to the joint canoical space
        sim3_j2c = rays['sim3_j2c'][:,None]
        Tmat_j2c, Rmat_j2c, Smat_j2c = vec_to_sim3(sim3_j2c)
        Smat_j2c = Smat_j2c.mean(-1)[...,None]
        Rmat_c2j, Tmat_c2j = rtmat_invert(Rmat_j2c, Tmat_j2c)
    
        xyz_coarse_sampled = xyz_coarse_sampled / Smat_j2c
        xyz_coarse_sampled = obj_to_cam(xyz_coarse_sampled, Rmat_c2j, Tmat_c2j)

    # root space point correspondence in t2
    xyz_coarse_target = xyz_coarse_sampled.clone()
    xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()

    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
       
        if fine_iter:
            # cycle loss (in the joint canonical space)
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # rigidity loss
            frame_disp3d = flow_fw.norm(2,-1)

            if "time_embedded_target" in rays.keys():
                time_embedded_target = rays['time_embedded_target'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            if "time_embedded_dentrg" in rays.keys():
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw


    elif 'bones' in models.keys():
        bones = models['bones']
        bone_rts_fw = rays['bone_rts']
        skin_aux = models['skin_aux']
        
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        time_embedded = rays['time_embedded'][:,None]
        bones_dfm = bone_transform(bones, bone_rts_fw) # coords after deform
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, bones_dfm,
                            time_embedded,  nerf_skin, skin_aux=skin_aux)

        # backward skinning
        xyz_coarse_sampled, bones_dfm = lbs(bones, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )

        if fine_iter:
            rest_pose_code =  models['rest_pose_code']
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones.device))
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, bones,
                                rest_pose_code,  nerf_skin, skin_aux=skin_aux)

            # cycle loss (in the joint canonical space)
            xyz_coarse_frame_cyc,_ = lbs(bones, bone_rts_fw,
                              skin_forward, xyz_coarse_sampled, backward=False)
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
            
            # rigidity loss
            num_bone = bones.shape[0] 
            bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
            bone_trn = bone_fw_reshape[:,:,9:12]
            bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
            frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
            
            if 'bone_rts_target' in rays.keys():
                bone_rts_target = rays['bone_rts_target']
                xyz_coarse_target,_ = lbs(bones, bone_rts_target, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            if 'bone_rts_dentrg' in rays.keys():
                bone_rts_dentrg = rays['bone_rts_dentrg']
                xyz_coarse_dentrg,_ = lbs(bones, bone_rts_dentrg, 
                                   skin_forward, xyz_coarse_sampled,backward=False)

    # nerf shape/rgb
    model_coarse = models['coarse']
    if 'env_code' in rays.keys():
        env_code = rays['env_code']
    else:
        env_code = None

    # set out of bounds weights to zero
    if render_vis: 
        clip_bound = obj_bound
        xyz_embedded = embedding_xyz(xyz_coarse_sampled)
        vis_pred = evaluate_mlp(models['nerf_vis'], 
                               xyz_embedded, chunk=chunk)[...,0].sigmoid()
    else:
        clip_bound = None
        vis_pred = None
    rgb_coarse, depth_rnd, weights_coarse, vis_coarse = \
        inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
                weights_only=False, env_code=env_code, 
                clip_bound=clip_bound, vis_pred=vis_pred, symm_shape=opts.symm_shape)
    sil_coarse =  weights_coarse[:,:-1].sum(1)
    result = {'img_coarse': rgb_coarse,
              'depth_rnd': depth_rnd,
              'sil_coarse': sil_coarse,
             }
  
    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)

    if fine_iter:
        xyz_joint = xyz_coarse_sampled
        if 'nerf_dp' in models.keys():
            # render densepose surface
            nerf_dp = models['nerf_dp']
            xyz_joint_embedded = embedding_xyz(xyz_joint)
            flow_dp = evaluate_mlp(nerf_dp, xyz_joint_embedded, chunk=chunk//N_samples)
            xyz_joint= xyz_joint + flow_dp
        result['joint_render'] = torch.sum(weights_coarse.unsqueeze(-1)*xyz_joint, -2)
   
        if 'sim3_j2c' in rays.keys():
            # similarity transform to the video canoical space
            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat_j2c, Tmat_j2c)
            xyz_coarse_target = xyz_coarse_target * Smat_j2c

        if 'rtk_vec_target' in rays.keys():
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)
            
        result['xyz_coarse_frame']   = xyz_coarse_frame 

        if 'rtk_vec_dentrg' in rays.keys():
            if 'sim3_j2c_dentrg' in rays.keys():
                # similarity transform to the video canoical space
                sim3_j2c_dt = rays['sim3_j2c_dentrg'][:,None]
                Tmat_j2c_dt, Rmat_j2c_dt, Smat_j2c_dt = vec_to_sim3(sim3_j2c_dt)
                Smat_j2c_dt = Smat_j2c_dt.mean(-1)[...,None]
                xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat_j2c_dt, Tmat_j2c_dt)
                xyz_coarse_dentrg = xyz_coarse_dentrg * Smat_j2c_dt

            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat) 
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
            
            
            
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
            if 'flowbw' in models.keys():
                result['frame_rigloss'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
                # only evaluate at with_grad mode
                if xyz_coarse_frame.requires_grad:
                    # elastic energy
                    result['elastic_loss'] = elastic_loss(model_flowbw, embedding_xyz, 
                                      xyz_coarse_frame, time_embedded)
            else:
                result['frame_rigloss'] =  (frame_rigloss).mean(-1)

            
            ### script to plot sigmas/weights
            #from matplotlib import pyplot as plt
            #plt.ioff()
            #plt.plot(weights_coarse[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('weights.png')
            #plt.cla()
            #plt.plot(sigmas[weights_coarse.sum(-1)==1][:].T.cpu().numpy(),'*-')
            #plt.savefig('sigmas.png')

        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled, vis_coarse, obj_bound, chunk)

        # render flow 
        xys = rays['xys']
        if 'rtk_vec_target' in rays.keys():
            flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                xys, img_size)
            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

        if 'rtk_vec_dentrg' in rays.keys():
            fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                xyz_coarse_dentrg, xys, img_size)
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid
        
        # viser feature matching
        if 'feats_at_samp' in rays.keys():
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            if progress<opts.warmup_init_steps:
                xyz_coarse_sampled = xyz_coarse_sampled.detach()
                weights_coarse = weights_coarse.detach()
            pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz,
                       feats_at_samp, xyz_coarse_sampled, weights_coarse,
                       obj_bound, is_training=is_training)


            # 3d-2d projection
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays)
            proj_err = proj_err/img_size * 2
            
            result['pts_pred'] = pts_pred
            result['pts_exp']  = pts_exp
            result['feat_err'] = feat_err # will be used as loss
            result['proj_err'] = proj_err # will be used as loss

        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        
        if 'img_at_samp' in rays.keys():
            # compute other losses
            img_at_samp = rays['img_at_samp']
            sil_at_samp = rays['sil_at_samp']
            flo_at_samp = rays['flo_at_samp']
            cfd_at_samp = rays['cfd_at_samp']

            # img loss
            img_loss_samp = (rgb_coarse - img_at_samp).pow(2)
            
            # sil loss, weight sil loss based on # points
            if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                pos_wt = sil_at_samp.numel()/    sil_at_samp.sum()
                neg_wt = sil_at_samp.numel()/(1-sil_at_samp).sum()
                sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
            else: sil_balance_wt = 1
            sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
               
            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1)
            # hard-threshold cycle error
            sil_at_samp_flo = (sil_at_samp>0)\
                     & (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
       
            result['img_at_samp']   = img_at_samp
            result['sil_at_samp']   = sil_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp'] = img_loss_samp 
            result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp

    return result, weights_coarse
