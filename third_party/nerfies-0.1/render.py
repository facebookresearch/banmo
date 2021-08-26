# @title Configure notebook runtime
# @markdown If you would like to use a GPU runtime instead, change the runtime type by going to `Runtime > Change runtime type`. 
# @markdown You will have to use a smaller batch size on GPU.
import pdb
import jax
runtime_type = 'gpu'  # @param ['gpu', 'tpu']
if runtime_type == 'tpu':
  import jax.tools.colab_tpu
  jax.tools.colab_tpu.setup_tpu()

print('Detected Devices:', jax.devices())


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
import tempfile
import imageio
from base64 import b64encode


# Monkey patch logging.
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint



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


# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.


from pathlib import Path
from pprint import pprint
import gin

from nerfies import configs


# @markdown The working directory where the trained model is.
train_dir = './logs'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
data_dir = './dataset/toby-sit'  # @param {type: "string"}
config_path = 'configs/gpu_quarterhd_4gpu.gin'

with open(config_path, 'r') as f:
  logging.info('Loading config from %s', config_path)
  config_str = f.read()
gin.parse_config(config_str)

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)


exp_config = configs.ExperimentConfig()
model_config = configs.ModelConfig()
eval_config = configs.EvalConfig()
train_config = configs.TrainConfig()


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
  image_scale=exp_config.image_scale,
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
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices)
del params

# @title Define pmapped render function.

import functools
from nerfies import evaluation

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

extract_fn = functools.partial(evaluation.extract_mesh,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)

# @title Load cameras.

from nerfies import utils


test_camera_paths = datasource.glob_cameras(Path(data_dir, 'camera-paths/orbit-mild'))
test_cameras = utils.parallel_map(datasource.load_camera, test_camera_paths, show_pbar=True)

# @title Render video frames.
from nerfies import visualization as viz


rng = rng + jax.host_id()  # Make random seed separate across hosts.
keys = random.split(rng, len(devices))

  
# mesh
import pdb
batch = datasets.camera_to_rays(test_cameras[0])
batch['metadata'] = {
    'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
    'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
}
pred_color, pred_depth, pred_depth_med, pred_acc = extract_fn(state, batch, rng=rng)

results = []
for i in range(len(test_cameras)):
  print(f'Rendering frame {i+1}/{len(test_cameras)}')
  camera = test_cameras[i]
  batch = datasets.camera_to_rays(camera)
  batch['metadata'] = {
      'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
      'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
  }

  pred_color, pred_depth, pred_depth_med, pred_acc = render_fn(state, batch, rng=rng)
  results.append((pred_color, pred_depth))
  pred_depth_viz = viz.colorize(pred_depth.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
import pdb;pdb.set_trace()
import cv2
cv2.imwrite('./0.png', pred_color)

