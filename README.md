# BANMo

## Build with conda
```
conda env create -f banmo.yml
conda activate banmo
# install softras
cd third_party/softras; python setup.py install; cd -;
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
# install kmeans-pytorch
cd third_party; git clone https://github.com/subhadarship/kmeans_pytorch; 
cd kmeans_pytorch; pip install --editable .; cd ../../
```

## Data preparation
### Download proprocessed data
Download data (~8G) as follows. 
Optionally you could download each file separately [here](https://www.dropbox.com/sh/2owtqkmyfhnn6qh/AADzpTrY50UGUk_qs40DTNa-a?dl=0) maually.
```
mkdir -p database/DAVIS/
curl -L https://www.dropbox.com/sh/2owtqkmyfhnn6qh/AADzpTrY50UGUk_qs40DTNa-a?dl=1 > ./database/cat-pikachiu.zip
curl -L https://www.dropbox.com/sh/invde3okcyum8e9/AABaJulvEoXBePV6gLLxgsM4a?dl=1 > ./database/human-cap.zip
unzip "database/*.zip" -d database/DAVIS/; rm database/*.zip
unzip "database/DAVIS/*.zip" -d database/DAVIS/; rm database/DAVIS/*.zip
```
To speed up data loading, we store images as lines
```
python preprocess/img2lines.py --seqname sfm-mcats10
python preprocess/img2lines.py --seqname adult7
```
### Preprocess videos from scratch (optional, don't use for now)
Instead using preprocessing, you provide scripts to process raw videos. See [here](./preprocess).

## Example: Cat
Create tmp dirs and run optimization monitor.
You may need to set up ssh tunneling to view the tensorboard monitor locally.
```
mkdir tmp; mkdir logdir
screen -dmS "tensorboard" bash -c "tensorboard --logdir=logdir --bind_all"
```
Run optimization on cat-pichahiu
```
bash scripts/template-sfm-cats10-line.sh 0,1,2,3,4,5,6,7
```
Extract animated meshes and render
```
bash scripts/render_vids.sh sfm-mcats10 logdir/sfm-mcats10-lbs-rkopt-90-b128-init/params_89.pth \
    "0 1 2 3 4 5 6 7 8 9" \
    "--sample_grid3d 128 --root_opt --lbs --full_mesh  --nouse_corresp --nouse_viser --nouse_proj --render_size 2 --ndepth 2 --nouse_human --queryfw --num_bones 25"
```

## Example: Human
Run optimization on human-cap
```
bash scripts/template-adult7-line.sh 0,1,2,3,4,5,6,7
```
Extract animated meshes and render
```
bash scripts/render_vids.sh adult7 logdir/adult7-lbs-rkopt-120-b128-init/params_115.pth \
    "0 1 2 3 4 5 6 7 8 9" \
    "--sample_grid3d 128 --root_opt --lbs --full_mesh  --nouse_corresp --nouse_viser --nouse_proj --render_size 2 --ndepth 2 --use_human --queryfw --num_bones 64 --symm_shape"
``` 

## Example: AMA-human (not tested)
Download swing and samba sequences from [here](https://people.csail.mit.edu/drdaniel/mesh_animation/) 
and convert into our DAVIS format with `scripts/ama-process/ama2davis.py`.
Then run the preprocessing scripts following [readme](preprocess/README.md).
To optimize, run 
```
bash scripts/template-ama-female.sh 0,1,2,3,4,5,6,7
```

## Example: synthetic data (not tested)
To render synthetic eagle/hands, see `scripts/render_eagle.sh` and `scripts/render_hands.sh`.
To optimize, see `scripts/template-cams.sh`

## Visualization scripts
To draw root pose trajectories (+rest shape) over time
```
python scripts/render_root.py --testdir logdir/sfm-mcats10-lbs-rkopt-90-b128-init/
```
To visualize matchings between frame 0 and 200.
```
bash scripts/render_match.sh sfm-mcats10 logdir/sfm-mcats10-lbs-rkopt-90-b128-init/params_89.pth "0 200" "--root_opt --lbs"
```

## Evaluation
To evaluate swing
```
python render_vis.py --testdir logdir/T_swing1-lbs-rkopt-ft2/ --outpath logdir/T_swing1-lbs-rkopt-ft2/T_swing1-eval --seqname T_swing1 --test_frames "{0}" --vp 0 --gtdir ~/data/AMA/T_swing/meshes/
```
To evaluate eagle
```
python render_vis.py --testdir logdir/baseline-a-eagle-1/ --outpath logdir/baseline-a-eagle-1/a-eagle-1-eval --seqname a-eagle-1 --test_frames "{0}" --vp 0 --gtdir database/DAVIS/Meshes/Full-Resolution/a-eagle-1/ --gt_pmat ''
```

## PoseNet Training
To traing pose predictor
```
bash scripts/template-mgpu.sh 0 test T_samba_small 10001 --num_epochs 30 --lbs --root_opt --ks_opt --nopreload --use_human --warmup_pose_ep 10
```

## deprecated
Adaptation to another set of videos
```
bash scripts/template-mgpu.sh 0 shiba_100-lbs-rkopt-ft1 shiba_100 10002 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbs-rkopt-ft5new/params_30.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --proj_end 2 --noanneal_freq --freeze_proj --nouse_resize --freeze_shape --freeze_cvf
```
To re-render nerfies meshes
```
bash scripts/render_nerfies.sh cat_905 logdir/baseline-cat_905/ {0}
```

## Additional Notes
- wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl

- use --full_mesh at extraction time to extract a complete surface (disable visibility check)

- use a surface-like mlp for mlp-based deformation field --init_beta 1./100 --sil_wt 1.

- need to set --perturb 0 at test

- Usage of preload (deprecated)
```
python preload.py --seqname sfm-mcats10
```
this will generate records for forward pairs of seqs under sfm-mcats10
at training time, add --preload to the script

### Human reconstruction
use --use_human
