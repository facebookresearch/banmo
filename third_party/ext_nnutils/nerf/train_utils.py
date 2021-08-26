import pdb
import torch

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field
local_chunksize=131072

def run_network(network_fn, pts, viewdirs, chunksize, embed_fn, embeddirs_fn, code=None):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if code is not None:
        embedded = torch.cat([embedded, code[:,None].repeat(1,pts.shape[1], 1).view(-1,code.shape[-1])],1)
    if embeddirs_fn is not None:
        viewdirs = viewdirs[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    is_train=True,
    radiance_field_noise_std=0.2,
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        64,
        dtype=ro.dtype,
        device=ro.device,
    )
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([num_rays, 64])

    if is_train:
        # noise
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch[..., -3:],
        local_chunksize,
        encode_position_fn,
        encode_direction_fn,
    )

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd/rd.norm(2,-1).unsqueeze(-1),
        radiance_field_noise_std=radiance_field_noise_std,
        white_background=False,
    )

    # fine pass
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],
        64,
        det=(not is_train),
    )
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    # pts -> (N_rays, N_samples + N_importance, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_fine,
        pts,
        ray_batch[..., -3:],
        local_chunksize,
        encode_position_fn,
        encode_direction_fn,
    )
    rgb_fine, disp_fine, acc_fine, _, depth_fine = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd/rd.norm(2,-1).unsqueeze(-1),
        radiance_field_noise_std=radiance_field_noise_std,
        white_background=False,
    )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_coarse, depth_fine


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    depth,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    is_train=True,
    radiance_field_noise_std=0.2,
):
    # Provide ray directions as input
    viewdirs = ray_directions
    #viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
    viewdirs = viewdirs.reshape((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
    ro = ray_origins.reshape((-1, 3))
    rd = ray_directions.reshape((-1, 3))
    near = (depth-1).reshape(-1,1)
    far =  (depth+1).reshape(-1,1)
    rays = torch.cat((ro, rd, near, far), dim=-1)
    rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=local_chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            is_train=is_train,
            radiance_field_noise_std=radiance_field_noise_std,
        )
        for batch in batches
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)
