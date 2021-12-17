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
Run optimization on cats
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
Run optimization on cats
```
bash scripts/template-adult7-line.sh 0,1,2,3,4,5,6,7
```
Extract animated meshes and render
```
bash scripts/render_vids.sh adult7 logdir/adult7-lbs-rkopt-120-b128-init/params_115.pth \
    "0 1 2 3 4 5 6 7 8 9" \
    "--sample_grid3d 128 --root_opt --lbs --full_mesh  --nouse_corresp --nouse_viser --nouse_proj --render_size 2 --ndepth 2 --use_human --queryfw --num_bones 64 --symm_shape"
``` 

## Visualization scripts
To draw root pose trajectories (+rest shape) over time
```
python scripts/render_root.py --testdir logdir/sfm-mcats10-lbs-rkopt-90-b128-init/
```
To visualize matchings between frame 0 and 200.
```
bash scripts/render_match.sh sfm-mcats10 logdir/sfm-mcats10-lbs-rkopt-90-b128-init/params_89.pth "0 200" "--root_opt --lbs"
```

## PoseNet Training
To traing pose predictor
```
bash scripts/template-mgpu.sh 0 test T_samba_small 10001 --num_epochs 30 --lbs --root_opt --ks_opt --nopreload --use_human --warmup_pose_ep 10
```


## Deprecated commands
To render synthetic objects
```
python scripts/render_synthetic.py --outdir syn-eagle-100 --model eagle --nframes 100 --alpha 1
python scripts/render_synthetic.py --outdir syn-eagled-15h --model eagle --nframes 15 --alpha 0.5 --xspeed 1
python scripts/render_synthetic.py --outdir syn-eagled-15q-vp2 --model eagle --nframes 15 --alpha 0.25 --xspeed 1 --init_a 0.5
python scripts/render_synthetic.py --outdir syn-hands-20h --model hands --nframes 20 --alpha 0.5 --xspeed 1 --init_a 0.25 --focal 5 --img_size 2048
python scripts/render_synthetic.py --outdir a-eagle-1 --model eagle --nframes 150 --alpha 0.1 --init_a 0 --xspeed 1
```
To optimize
```
bash scripts/template.sh logname dataname address --flowbw
bash scripts/template.sh syn-eagled-15h-lbs-corresp-can15 syn-eagled-15h 10034 --lbs --num_epochs 30 --use_corresp
bash scripts/template.sh syn-eagled-15h-lbs-corresp-root syn-eagled-15h 10045 --num_epochs 30 --use_corresp --root_opt --nouse_cam --lbs
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 cat_601-lbs-correspd-root-cnn-8gpu cat_601 10041 --num_epochs 30 --use_corresp --lbs --root_opt --flow_dp --nouse_cam --cnn_root
```
Newer ones
```
bash scripts/template-mgpu.sh 0 nerfies_cat_807-lbs-rkopt nerfies_cat_807 10001 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
bash scripts/template-mgpu.sh 0 sfm-mcats10-lbs-rkopt-b16-2ff sfm-mcats10 10001 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --batch_size 16 --nsample 64
bash scripts/template-mgpu.sh 0 sfm-mcats10-lbs-rkopt-ft4 sfm-mcats10 10010 --num_epochs 100 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbsn-correspdv-rkopt-ft-100ep-stage3/params_100.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize
bash scripts/template-mgpu.sh 0 ama-female-lbs-rkopt-b16 ama-female 10004 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-T_samba_small/cnn-params_10.pth --batch_size 16 --nsample 64 --use_human
bash scripts/template-mgpu.sh 0 ama-female-lbs-rkopt-ft1 ama-female 10060 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-T_samba_small/cnn-params_10.pth --model_path logdir/ama-female1-lbs-rkopt-b16-ftcse1e504/params_26.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --use_human
bash scripts/template-mgpu.sh 0 a-eagle-lbs-rkopt-b16-cam-cse a-eagle 10001 --num_epochs 30 --lbs --root_opt --ks_opt --batch_size 16 --nsample 64 --ft_cse
bash scripts/template-mgpu.sh 0 a-eagle-lbs-rkopt-noise30-ft1 a-eagle 10011 --num_epochs 30 --lbs --root_opt --ks_opt --model_path logdir/a-eagle-lbs-rkopt-b16-noise30-cse/params_30.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --ftcse_steps 0.2 --mtcse_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --ft_cse
```
Stage-wise finetuning
```
python submit.py scripts/template-stage.sh 0,1,2,3,4,5,6,7 sfm-mcats10 1000
```
Adaptation to another set of videos
```
bash scripts/template-mgpu.sh 0 shiba_100-lbs-rkopt-ft1 shiba_100 10002 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbs-rkopt-ft5new/params_30.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --proj_end 2 --noanneal_freq --freeze_proj --nouse_resize --freeze_shape --freeze_cvf
```
To evaluate swing
```
python render_vis.py --testdir logdir/T_swing1-lbs-rkopt-ft2/ --outpath logdir/T_swing1-lbs-rkopt-ft2/T_swing1-eval --seqname T_swing1 --test_frames "{0}" --vp 0 --gtdir ~/data/AMA/T_swing/meshes/
```
To evaluate eagle
```
python render_vis.py --testdir logdir/baseline-a-eagle-1/ --outpath logdir/baseline-a-eagle-1/a-eagle-1-eval --seqname a-eagle-1 --test_frames "{0}" --vp 0 --gtdir database/DAVIS/Meshes/Full-Resolution/a-eagle-1/ --gt_pmat ''
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

### Usage of preload (deprecated)
```
python preload.py --seqname sfm-mcats10
```
this will generate records for forward pairs of seqs under sfm-mcats10
at training time, add --preload to the script

### Human reconstruction
use --use_human
