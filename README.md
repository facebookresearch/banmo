# BANMo 
[[Project page]](https://banmo-www.github.io/) [[Paper]](https://banmo-www.github.io/banmo-cvpr.pdf) [[Colab for NVS]](https://colab.research.google.com/drive/1dQJn1vsuz0DkyRZbOA1SulkVQ0V1kMUP?usp=sharing)

This repo provides scripts to reproduce experiments in the paper. For the latest updates on the software, please check out [lab4d](https://github.com/lab4d-org/lab4d).

### Changelog
- **11/21**: Remove eikonal loss to align with paper results, [#36](https://github.com/facebookresearch/banmo/issues/36)
- **08/09**: Fix eikonal loss that regularizes surface (resulting in smoother mesh).
- **06/18**: Add a colab demo for novel view synthesis.
- **04/11**: Replace matching loss with feature rendering loss; Fix bugs in LBS; Stablize optimization.
- **03/20**: Add mesh color option (canonical mappihg vs radiance) during surface extraction. See `--ce_color` flag.
- **02/23**: Improve NVS with fourier light code, improve uncertainty MLP, add long schedule, minor speed up.
- **02/17**: Add adaptation to a new video, optimization with known root poses, and pose code visualization.
- **02/15**: Add motion-retargeting, quantitative evaluation and synthetic data generation/eval.

## Install
### Build with conda
We provide two versions. 
<details><summary>[A. torch1.10+cu113 (1.4x faster on V100)]</summary>

```
# clone repo
git clone git@github.com:facebookresearch/banmo.git --recursive
cd banmo
# install conda env
conda env create -f misc/banmo-cu113.yml
conda activate banmo-cu113
# install pytorch3d (takes minutes), kmeans-pytorch
pip install -e third_party/pytorch3d
pip install -e third_party/kmeans_pytorch
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
</details>

<details><summary>[B. torch1.7+cu110]</summary>

```
# clone repo
git clone git@github.com:facebookresearch/banmo.git --recursive
cd banmo
# install conda env
conda env create -f misc/banmo.yml
conda activate banmo
# install kmeans-pytorch
pip install -e third_party/kmeans_pytorch
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```
</details>

### Data
We provide two ways to obtain data. 
The easiest way is to download and unzip the pre-processed data as follows.
<details><summary>[Download pre-processed data]</summary>

We provide preprocessed data for cat and human.
Download the pre-processed `rgb/mask/flow/densepose` images as follows
```
# (~8G for each)
bash misc/processed/download.sh cat-pikachiu
bash misc/processed/download.sh human-cap
```
</details>

<details><summary>[Download raw videos]</summary>

Download raw videos to `./raw/` folder
```
bash misc/vid/download.sh cat-pikachiu
bash misc/vid/download.sh human-cap
bash misc/vid/download.sh dog-tetres
bash misc/vid/download.sh cat-coco
```
</details>

**To use your own videos, or pre-process raw videos into banmo format, 
please follow the instructions [here](./preprocess).**

### PoseNet weights
<details><summary>[expand]</summary>

Download pre-trained PoseNet weights for human and quadrupeds
```
mkdir -p mesh_material/posenet && cd "$_"
wget $(cat ../../misc/posenet.txt); cd ../../
```
</details>


## Demo
This example shows how to reconstruct a cat from 11 videos and a human from 10 videos.
For more examples, see [here](./scripts/README.md).

<details><summary>Hardware/time for running the demo</summary>

The [short schedule](./scripts/template-short.sh) takes 4 hours on 2 V100 GPUs (+SSD storage).
To reach higher quality, the [full schedule](./scripts/template.sh) takes 12 hours.
We provide a [script](./scripts/template-accu.sh) that use gradient accumulation
 to support experiments on fewer GPUs / GPU with lower memory.
</details>

<details><summary>Setting good hyper-parameter for videos with various length</summary>

When optimizing videos with different lengths, we found it useful to scale batchsize with the number of frames.
A rule of thumb is to set "num gpus" x "batch size" x "accu steps" ~= num frames.
This means more video frames needs more GPU memory but the same optimization time.
</details>

<details><summary>Try pre-optimized models</summary>

We provide [pre-optimized models](https://www.dropbox.com/sh/5ue6tpsqmt6gstw/AAB9FD6on0UZDnThr6GEde46a?dl=0) 
and scripts to run novel view synthesis and mesh extraction (results saved at `tmp/*all.mp4`). Also see this [Colab for NVS](https://colab.research.google.com/drive/1dQJn1vsuz0DkyRZbOA1SulkVQ0V1kMUP?usp=sharing).
 
```
# download pre-optimized models
mkdir -p tmp && cd "$_"
wget https://www.dropbox.com/s/qzwuqxp0mzdot6c/cat-pikachiu.npy
wget https://www.dropbox.com/s/dnob0r8zzjbn28a/cat-pikachiu.pth
wget https://www.dropbox.com/s/p74aaeusprbve1z/opts.log # flags used at opt time
cd ../

seqname=cat-pikachiu
# render novel views
bash scripts/render_nvs.sh 0 $seqname tmp/cat-pikachiu.pth 5 0
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: path to the weights
# argv[4]: video id used for pose traj
# argv[5]: video id used for root traj

# Extract articulated meshes and render
bash scripts/render_mgpu.sh 0 $seqname tmp/cat-pikachiu.pth \
        "0 5" 64
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: weights path
# argv[4]: video id separated by space
# argv[5]: resolution of running marching cubes (use 256 to get higher-res mesh)
```

</details>

#### 1. Optimization
<details><summary>[cat-pikachiu]</summary>

```
seqname=cat-pikachiu
# To speed up data loading, we store images as lines of pixels). 
# only needs to run it once per sequence and data are stored
python preprocess/img2lines.py --seqname $seqname

# Optimization
bash scripts/template.sh 0,1 $seqname 10001 "no" "no"
# argv[1]: gpu ids separated by comma 
# args[2]: sequence name
# args[3]: port for distributed training
# args[4]: use_human, pass "" for human cse, "no" for quadreped cse
# args[5]: use_symm, pass "" to force x-symmetric shape

# Extract articulated meshes and render
bash scripts/render_mgpu.sh 0 $seqname logdir/$seqname-e120-b256-ft2/params_latest.pth \
        "0 1 2 3 4 5 6 7 8 9 10" 256
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: weights path
# argv[4]: video id separated by space
# argv[5]: resolution of running marching cubes (256 by default)
```

https://user-images.githubusercontent.com/13134872/154554031-332e2355-3303-43e3-851c-b5812699184b.mp4


</details>

<details><summary>[human-cap]</summary>

```
seqname=adult7
python preprocess/img2lines.py --seqname $seqname
bash scripts/template.sh 0,1 $seqname 10001 "" ""
bash scripts/render_mgpu.sh 0 $seqname logdir/$seqname-e120-b256-ft2/params_latest.pth \
        "0 1 2 3 4 5 6 7 8 9" 256
```

https://user-images.githubusercontent.com/13134872/154554210-3bb0a439-fe46-4ea3-a058-acecf5f8dbb5.mp4
  
</details>

#### 2. Visualization tools
<details><summary>[Tensorboard]</summary>

```
# You may need to set up ssh tunneling to view the tensorboard monitor locally.
screen -dmS "tensorboard" bash -c "tensorboard --logdir=logdir --bind_all"
```
</details>

<details><summary>[Root pose, rest mesh, bones]</summary>

To draw root pose trajectories (+rest shape) over epochs
```
# logdir
logdir=logdir/$seqname-e120-b256-init/
# first_idx, last_idx specifies what frames to be drawn
python scripts/visualize/render_root.py --testdir $logdir --first_idx 0 --last_idx 120
```
Find the output at `$logdir/mesh-cam.gif`. 
During optimization, the rest mesh and bones at each epoch are saved at `$logdir/*rest.obj`.

https://user-images.githubusercontent.com/13134872/154553887-1871fdea-24f4-4a79-8689-86ff6af7fa52.mp4

</details>

<details><summary>[Correspondence/pose code]</summary>

To visualize 2d-2d and 2d-3d matchings of the latest epoch weights
```
# 2d matches between frame 0 and 100 via 2d->feature matching->3d->geometric warp->2d
bash scripts/render_match.sh $logdir/params_latest.pth "0 100" "--render_size 128"
```
2d-2d matches will be saved to `tmp/match_%03d.jpg`. 
2d-3d feature matches of frame 0 will be saved to `tmp/match_line_pred.obj`.
2d-3d geometric warps of frame 0 will be saved to `tmp/match_line_exp.obj`.
near-plane frame 0 will be saved to `tmp/match_plane.obj`.
Pose code visualization will be saved at `tmp/code.mp4`.

https://user-images.githubusercontent.com/13134872/154553652-c93834db-cce2-4158-a30a-21680ab46a63.mp4


</details>

<details><summary>[Render novel views]</summary>

Render novel views at the canonical camera coordinate
```
bash scripts/render_nvs.sh 0 $seqname logdir/$seqname-e120-b256-ft2/params_latest.pth 5 0
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: path to the weights
# argv[4]: video id used for pose traj
# argv[5]: video id used for root traj
```
Results will be saved at `logdir/$seqname-e120-b256-ft2/nvs*.mp4`.
  
https://user-images.githubusercontent.com/13134872/155441493-38bf7a02-a6ee-4f2f-9dc5-0cf98a4c7c45.mp4

</details>

<details><summary>[Render canonical view over iterations]</summary>

Render depth and color of the canonical view over optimization iterations
```
bash scripts/visualize/nvs_iter.sh 0 logdir/$seqname-e120-b256-init/
# argv[1]: gpu id
# argv[2]: path to the logdir
```
Results will be saved at `logdir/$seqname-e120-b256-init/vis-iter*.mp4`.
  

https://user-images.githubusercontent.com/13134872/162283256-49f9de87-0bce-4f7f-9376-651a170e8879.mp4



https://user-images.githubusercontent.com/13134872/162283257-7636462b-c698-4411-9084-57f7a0bb89e8.mp4



</details>

### Common install issues
<details><summary>[expand]</summary>

* Q: pyrender reports `ImportError: Library "GLU" not found.`
    * install `sudo apt install freeglut3-dev`
* Q: ffmpeg reports `libopenh264.so.5` not fund
    * resinstall ffmpeg in conda `conda install -c conda-forge ffmpeg`
</details>

### Note on arguments
<details><summary>[expand]</summary>

- use `--use_human` for human reconstruction, otherwise it assumes quadruped animals
- use `--full_mesh` to disable visibility check at mesh extraction time
- use `--noce_color` at mesh extraction time to assign radiance instead canonical mapping as vertex colors.
- use `--queryfw` at mesh extraction time to extract forward articulated meshes, which only needs to run marching cubes once.
- use `--use_cc` maintains the largest connected component for rest mesh in order to set the object bounds and near-far plane (by default turned on). Turn it off with `--nouse_cc` for disconnected objects such as hands.
- use `--debug` to print out the rough time each component takes.
</details>

### Acknowledgement
<details><summary>[expand]</summary>

Volume rendering code is borrowed from [Nerf_pl](https://github.com/kwea123/nerf_pl).
Flow estimation code is adapted from [VCN-robust](https://github.com/gengshan-y/rigidmask).
Other external repos:
- [Detectron2](https://github.com/facebookresearch/detectron2) (modified)
- [SoftRas](https://github.com/ShichenLiu/SoftRas) (modified, for synthetic data generation)
- [Chamfer3D](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) (for evaluation)
</details>

### License
<details><summary>[expand]</summary>

- code: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
- dataset
  - [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/): cat-pikachiu, cat-coco, dog-tetres, human-cap
  - [Pexels free license](https://www.pexels.com/license/): penguin
  - [Turbosquid license](https://blog.turbosquid.com/turbosquid-3d-model-license/): [hands](https://www.turbosquid.com/3d-models/hand-hdri-shader-3d-model-1311775), [eagle](https://www.turbosquid.com/3d-models/eagle-rigged-fbx-free/1045001)
    - the final dataset is modified from those 3D assets. 
  - [AMA](https://people.csail.mit.edu/drdaniel/mesh_animation/) comes without a license
  - We thank the artists for sharing theirs videos and 3D assets.
</details>
