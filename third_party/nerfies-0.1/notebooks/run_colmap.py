import pdb
import shutil
import sys
sys.path.insert(0,'third_party/nerfies-0.1/')
sys.path.insert(0,'third_party/nerfies-0.1/third_party/pycolmap/')
# @title Define Scene Manager.
from absl import logging
from typing import Dict
import numpy as np
from nerfies.camera import Camera
import pycolmap
from pycolmap import Quaternion
import os
os.chdir('third_party/nerfies-0.1/notebooks/')
from importlib import reload
# @title Configure dataset directories
from pathlib import Path
# @markdown The base directory for all captures. This can be anything if you're running this notebook on your own Jupyter runtime.
save_dir = '../dataset/'  # @param {type: 'string'}
# @markdown The name of this capture. The working directory will be `$save_dir/$capture_name`. **Make sure you change this** when processing a new video.
seqname= sys.argv[1]
capture_name = seqname  # @param {type: 'string'}
# The root directory for this capture.
root_dir = Path(save_dir, capture_name)

try:    shutil.rmtree(root_dir)
except: pass
# Where to save RGB images.
rgb_dir = root_dir / 'rgb'
mask_dir= root_dir / 'mask'
rgb_raw_dir = root_dir / 'rgb-raw'
mask_raw_dir=root_dir / 'mask-raw'
# Where to save the COLMAP outputs.
colmap_dir = root_dir / 'colmap'
colmap_db_path = colmap_dir / 'database.db'
colmap_out_path = colmap_dir / 'sparse'


colmap_out_path.mkdir(exist_ok=True, parents=True)
rgb_raw_dir.mkdir(exist_ok=True, parents=True)

camera_dir = root_dir / 'camera'
camera_dir.mkdir(exist_ok=True, parents=True)

print(f"""Directories configured:
  root_dir = {root_dir}
  rgb_raw_dir = {rgb_raw_dir}
  rgb_dir = {rgb_dir}
  mask_dir = {mask_dir}
  colmap_dir = {colmap_dir}
  camera_dir = {camera_dir}
""")


# @title Flatten into images.
import os
import cv2
import glob


target_num_frames = 100 # @param {type: 'number'}

# @markdown Check this if you want to reprocess the frames.
overwrite = True  # @param {type:'boolean'}

if (rgb_dir / '1x').exists() and not overwrite:
  raise RuntimeError(
      f'The RGB frames have already been processed. Check `overwrite` and run again if you really meant to do this.')
else:
  tmp_rgb_raw_dir = 'rgb-raw'
  tmp_mask_raw_dir = 'mask-raw'
  try:    shutil.rmtree(tmp_rgb_raw_dir)
  except: pass
  try:    shutil.rmtree(tmp_mask_raw_dir)
  except: pass
  os.makedirs(tmp_rgb_raw_dir, exist_ok=True)
  os.makedirs(tmp_mask_raw_dir, exist_ok=True)

  for i in glob.glob('/private/home/gengshany/data/DAVIS/JPEGImages/Full-Resolution/%s/*'%seqname):
        img = cv2.imread(i)
        imgmask = cv2.imread(i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png'),0)>0
        use_lasr=False
        if use_lasr:
            img[~imgmask]=0.
            #img[imgmask]=255.
        cv2.imwrite('%s/%s.png'%(tmp_rgb_raw_dir, i.split('/')[-1][:-4]), img)
        cv2.imwrite('%s/%s.png.png'%(tmp_mask_raw_dir, i.split('/')[-1][:-4]), (~imgmask).astype(int)*255)

  try:   shutil.rmtree(rgb_raw_dir)
  except: pass
  try:   shutil.rmtree(mask_raw_dir)
  except: pass
  shutil.copytree(tmp_rgb_raw_dir,  rgb_raw_dir)
  shutil.copytree(tmp_mask_raw_dir, mask_raw_dir)



# @title Resize images into different scales.
# @markdown Here we save the input images at various resolutions (downsample by a factor of 1, 2, 4, 8). We use area relation interpolation to prevent moire artifacts.
import concurrent.futures
import numpy as np
import cv2
import imageio
from PIL import Image


def save_image(path, image: np.ndarray) -> None:
  print(f'Saving {path}')
  if not path.parent.exists():
    path.parent.mkdir(exist_ok=True, parents=True)
  with path.open('wb') as f:
    image = Image.fromarray(np.asarray(image))
    image.save(f, format=path.suffix.lstrip('.'))


def image_to_uint8(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint8 array."""
  if image.dtype == np.uint8:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
  """Trim the image if not divisible by the divisor."""
  height, width = image.shape[:2]
  if height % divisor == 0 and width % divisor == 0:
    return image

  new_height = height - height % divisor
  new_width = width - width % divisor

  return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Downsamples the image by an integer factor to prevent artifacts."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  if height % scale > 0 or width % scale > 0:
    raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                     f' scale ({scale}).')
  out_height, out_width = height // scale, width // scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized



image_scales = "1,2,4,8"  # @param {type: "string"}
image_scales = [int(x) for x in image_scales.split(',')]

tmp_rgb_dir = rgb_dir
try:shutil.rmtree(tmp_rgb_dir)
except: pass
os.makedirs(tmp_rgb_dir, exist_ok=True)

for image_path in Path(rgb_raw_dir).glob('*.png'):
  image = make_divisible(imageio.imread(image_path), max(image_scales))
  for scale in image_scales:
    os.makedirs('%s/%dx'%(tmp_rgb_dir,scale), exist_ok=True)
    save_image(
        tmp_rgb_dir / f'{scale}x/{image_path.stem}.png',
        image_to_uint8(downsample_image(image, scale)))
  

# !rsync -av "$tmp_rgb_dir/" "$rgb_dir/"

# @title Example frame.
# @markdown Make sure that the video was processed correctly.
# @markdown If this gives an exception, try running the preceding cell one more time--sometimes uploading to Google Drive can fail.

from pathlib import Path
import imageio
from PIL import Image

image_paths = list((rgb_dir / '1x').iterdir())
Image.open(image_paths[0])

# @title Extract features.
# @markdown Computes SIFT features and saves them to the COLMAP DB.
share_intrinsics = True  # @param {type: 'boolean'}
assume_upright_cameras = True  # @param {type: 'boolean'}

# @markdown This sets the scale at which we will run COLMAP. A scale of 1 will be more accurate but will be slow.
colmap_image_scale =   1# @param {type: 'number'}
colmap_rgb_dir = rgb_dir / f'{colmap_image_scale}x'
colmap_mask_dir = mask_raw_dir

# @markdown Check this if you want to re-process SfM.
overwrite = True  # @param {type: 'boolean'}

if overwrite and colmap_db_path.exists():
  colmap_db_path.unlink()

cmd = 'colmap feature_extractor \
--SiftExtraction.use_gpu 0 \
--SiftExtraction.upright %d \
--ImageReader.camera_model OPENCV \
--ImageReader.single_camera %d \
--database_path %s \
--image_path %s \
--ImageReader.mask_path %s'%(assume_upright_cameras, 
                                                   share_intrinsics,
                                                   colmap_db_path,
                                                   colmap_rgb_dir,
                                                   colmap_mask_dir)
os.system(cmd)


# @title Match features.
# @markdown Match the SIFT features between images. Use `exhaustive` if you only have a few images and use `vocab_tree` if you have a lot of images.

cmd= 'colmap exhaustive_matcher \
    --SiftMatching.use_gpu 0 \
    --database_path %s'%(colmap_db_path)
os.system(cmd)

# @title Reconstruction.
# @markdown Run structure-from-motion to compute camera parameters.

refine_principal_point = True  #@param {type:"boolean"}
min_num_matches =   30# @param {type: 'number'}
filter_max_reproj_error = 2  # @param {type: 'number'}
tri_complete_max_reproj_error = 2  # @param {type: 'number'}

cmd = 'colmap mapper \
  --Mapper.ba_refine_principal_point %d \
  --Mapper.filter_max_reproj_error %d \
  --Mapper.tri_complete_max_reproj_error %d \
  --Mapper.min_num_matches %d \
  --database_path %s \
  --image_path %s \
  --output_path %s'%(refine_principal_point,
                                           filter_max_reproj_error,
                                           tri_complete_max_reproj_error,
                                           min_num_matches,
                                           colmap_db_path,
                                           colmap_rgb_dir,
                                           colmap_out_path)
os.system(cmd)

# @title Verify that SfM worked.

if not colmap_db_path.exists():
  raise RuntimeError(f'The COLMAP DB does not exist, did you run the reconstruction?')
elif not (colmap_dir / 'sparse/0/cameras.bin').exists():
  raise RuntimeError("""
SfM seems to have failed. Try some of the following options:
 - Increase the FPS when flattenting to images. There should be at least 50-ish images.
 - Decrease `min_num_matches`.
 - If you images aren't upright, uncheck `assume_upright_cameras`.
""")
else:
  print("Everything looks good!")



