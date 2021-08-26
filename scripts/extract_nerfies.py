# @title Define imports and utility functions.

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
jax_config.enable_omnistaging() # Linen requires enabling omnistaging

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython
import tempfile
import imageio
from IPython.display import display, HTML
from base64 import b64encode


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint


def show_image(image, fmt='png'):
    image = image_utils.image_to_uint8(image)
    f = BytesIO()
    PIL.Image.fromarray(image).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def show_video(frames, fps=30):
  with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
    with imageio.get_writer(f.name, fps=fps) as writer:
      for frame in frames:
        writer.append_data(frame)

    with open(f.name,'rb') as f:
      data_url = "data:video/mp4;base64," + b64encode(f.read()).decode()
    display(HTML("""
    <video controls autoplay loop>
      <source src="%s" type="video/mp4">
    </video>
    """ % data_url))


import os
os.chdir('third_party/nerfies-0.1/')

# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.

import sys
sys.path.insert(0,'third_party/nerfies-0.1/')
from pathlib import Path
from pprint import pprint
import gin
from IPython.display import display, Markdown

from nerfies import configs


dataname = sys.argv[1]
# @markdown The working directory where the trained model is.
train_dir = './logs/cat_501/'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = 'dataset/%s/'%dataname  # @param {type: "string"}

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_path = '%s/config.gin'%(train_dir)
with open(config_path, 'r') as f:
  logging.info('Loading config from %s', config_path)
  config_str = f.read()
gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
  logging.info('Saving config to %s', config_path)
  f.write(config_str)

exp_config = configs.ExperimentConfig()
model_config = configs.ModelConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

# @title Create datasource and show an example.

from nerfies import datasets
from nerfies import image_utils

datasource_spec = exp_config.datasource_spec
if datasource_spec is None:
    datasource_spec = {
        'type': exp_config.datasource_type,
        'data_dir': data_dir,
    }
datasource = datasets.from_config(
  datasource_spec,
  #image_scale=exp_config.image_scale,
  image_scale=1,
  use_appearance_id=model_config.use_appearance_metadata,
  use_camera_id=model_config.use_camera_metadata,
  use_warp_id=model_config.use_warp,
  random_seed=exp_config.random_seed)

# @title Initialize model
# @markdown Defines the model and initializes its parameters.

from flax.training import checkpoints
from nerfies import models
from nerfies import model_utils
from nerfies import schedules
from nerfies import training


rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.host_id())
devices = jax.devices()

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
elastic_loss_weight_sched = schedules.from_config(
    train_config.elastic_loss_weight_schedule)

rng, key = random.split(rng)
params = {}
model, params['model'] = models.nerf(
    key,
    model_config,
    batch_size=train_config.batch_size,
    num_appearance_embeddings=len(datasource.appearance_ids),
    num_camera_embeddings=len(datasource.camera_ids),
    num_warp_embeddings=len(datasource.warp_ids),
    near=datasource.near,
    far=datasource.far,
    use_warp_jacobian=train_config.use_elastic_loss,
    use_weights=train_config.use_elastic_loss)

optimizer_def = optim.Adam(learning_rate_sched(0))
optimizer = optimizer_def.create(params)
state = model_utils.TrainState(
    optimizer=optimizer,
    warp_alpha=warp_alpha_sched(0))
scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    background_loss_weight=train_config.background_loss_weight)
logging.info('Restoring checkpoint from %s', checkpoint_dir)
#state = checkpoints.restore_checkpoint(checkpoint_dir, state,step=10000)
step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices)
del params

# @title Define pmapped render function.

import functools
from nerfies import evaluation
from importlib import reload  

devices = jax.devices()


def _model_fn(key_0, key_1, params, rays_dict, alpha):
  out = model.apply({'params': params},
                    rays_dict,
                    warp_alpha=alpha,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    # key0, key1, params, rays_dict, alpha
    in_axes=(0, 0, 0, 0, 0),
    devices=devices,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)


# @title Load cameras.

from nerfies import utils


test_camera_paths = datasource.glob_cameras(Path(data_dir, 'camera'))
#test_camera_paths = datasource.glob_cameras('dataset/syn-eagled-/camera/')
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

from jax import tree_util
import time
from importlib import reload  
reload(models)

model, _ = models.nerf_mesh(
    key,
    model_config,
    batch_size=train_config.batch_size,
    num_appearance_embeddings=len(datasource.appearance_ids),
    num_camera_embeddings=len(datasource.camera_ids),
    num_warp_embeddings=len(datasource.warp_ids),
    near=datasource.near,
    far=datasource.far,
    use_warp_jacobian=train_config.use_elastic_loss,
    use_weights=train_config.use_elastic_loss)

def _model_fn(key_0, key_1, params, rays_dict, alpha):
  out = model.apply({'params': params},
                    rays_dict,
                    warp_alpha=alpha,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    # key0, key1, params, rays_dict, alpha
    in_axes=(0, 0, 0, 0, 0),
    devices=devices,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

device_count=2
chunk=eval_config.chunk
model_fn = pmodel_fn


_, key_0, key_1 = jax.random.split(rng, 3)
key_0 = jax.random.split(key_0, device_count)
key_1 = jax.random.split(key_1, device_count)
host_id = jax.host_id()

def extract_mesh(bs_pts, chunk, rays_dict, device_count, key_0, key_1, state):
    occ = []
    for i in range(0, bs_pts, chunk):
        #logging.info('\tRendering ray batch: %d/%d', i, num_rays)
        # pylint: disable=cell-var-from-loop
        chunk_slice_fn = lambda x: x[i:i + chunk]
        chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, rays_dict)
        num_chunk_rays = chunk_rays_dict['query_xyz'].shape[0]
        remainder = num_chunk_rays % device_count
        if remainder != 0:
          padding = device_count - remainder
          # pylint: disable=cell-var-from-loop
          chunk_pad_fn = lambda x: jnp.pad(x, ((0, padding), (0, 0)), mode='edge')
          chunk_rays_dict = tree_util.tree_map(chunk_pad_fn, chunk_rays_dict)
        else:
          padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        per_host_rays = num_chunk_rays // jax.host_count()
        chunk_rays_dict = tree_util.tree_map(
            lambda x: x[(host_id * per_host_rays):((host_id + 1) * per_host_rays)],
            chunk_rays_dict)
        chunk_rays_dict = utils.shard(chunk_rays_dict, device_count)

        model_out = model_fn(
            key_0,
            key_1,
            state.optimizer.target['model'],
            chunk_rays_dict,
            state.warp_alpha)
        occ.append(utils.unshard(model_out['alpha'][0], padding))
    occ = jnp.concatenate(occ, axis=0)

    vol_o = np.asarray(occ.reshape((grid_size, grid_size, grid_size)))
    return vol_o


import trimesh
import mcubes
import cv2
import shutil
save_data=True
seqname='nerfies_%s'%dataname
logname='nerfies-%s'%seqname
bound=(datasource.far-datasource.near)/2

frame_len=len(test_cameras)
#render_len=15
render_len=frame_len
#grid_size = 128*2
grid_size = 12
threshold = 20

if save_data:
    target_dir='logdir/%s/'%logname
    try: shutil.rmtree(target_dir)
    except: pass
    os.mkdir(target_dir)

    rgb_save_path = '../vid2shape/database/DAVIS/JPEGImages/Full-Resolution/%s'%seqname
    sil_save_path = '../vid2shape/database/DAVIS/Annotations/Full-Resolution/%s'%seqname
    try:
        shutil.rmtree(rgb_save_path)
        shutil.rmtree(sil_save_path)
    except: pass
    os.mkdir(rgb_save_path)
    os.mkdir(sil_save_path)

pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
query_yxz = query_yxz.reshape(-1, 3)
query_xyz = np.concatenate([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)
#query_xyz[:,-1] *= -1

bs_pts = query_xyz.shape[0]
points = query_xyz[:,None]

for tidx in range(0,render_len):
    tidxf = int(tidx/render_len*frame_len)
    print(tidxf)
    metadata={}
    metadata['appearance'] = tidxf* np.ones((points.shape[0],1)).astype(int)
    metadata['warp'] = tidxf*np.ones((points.shape[0],1)).astype(int)
    rays_dict={'query_xyz': query_xyz, 'metadata': metadata}

    vol_o = extract_mesh(bs_pts, chunk, rays_dict, device_count, key_0, key_1, state)

    print('fraction occupied:', (vol_o > threshold).astype(float).mean())
    vertices, triangles = mcubes.marching_cubes(vol_o, threshold)
    vertices = (vertices - grid_size/2)/grid_size*2
    vertices = vertices * bound

    mesh = trimesh.Trimesh(vertices, triangles)
    camtxt = np.zeros((4,4))
    camtxt[:3,:3] = test_cameras[tidxf].orientation
    camtxt[:3,3] = camtxt[:3,:3].dot(-test_cameras[tidxf].position[:,None])[:,0]
    camtxt[3,:2] = [test_cameras[tidxf].focal_length, 
                    test_cameras[tidxf].focal_length]
    camtxt[3,2:] = test_cameras[tidxf].principal_point
    
    if save_data:
        mesh.export('%s/%s-mesh-%05d.obj'%(target_dir, seqname,tidx))
        np.savetxt( '%s/%s-cam-%05d.txt'%(target_dir,seqname,tidx), camtxt)
        input_rgb = datasource.load_rgb(datasource.train_ids[tidxf])
        input_sil = 1-datasource.load_sil(datasource.train_ids[tidxf])
        cv2.imwrite('%s/%05d.jpg'%(rgb_save_path,tidx), 255*input_rgb[:,:,::-1])
        cv2.imwrite('%s/%05d.png'%(sil_save_path,tidx), 128*input_sil)

print(camtxt[-1])
print(datasource.near)
print(datasource.far)
