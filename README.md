# vid2shape

## Installation
### Build with conda
```
conda env create -f vid2shape.yml
conda activate vid2shape
# install softras
cd third_party/softras; python setup.py install; cd -;
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
# marching cubes
pip install --upgrade PyMCubes
# clone detectron2 repo to ../
# install healpy
```

## Preprocess
Extract frames and masks
```
bash scripts/preprocess.sh video-folder dir-name
```

```
bash scripts/colmap_to_data.sh seqname
```

## Run
Create tmp dirs
```
ln -s /private/home/gengshany/data/ database
mkdir tmp
wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl
mv geodists_sheep_5004.pkl ./mesh_material
```

To render objects
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
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 nerfies_cat_807-lbs-rkopt nerfies_cat_807 10001 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 sfm-mcats10-lbs-rkopt-b16-2ff sfm-mcats10 10001 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --batch_size 16 --nsample 64
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 sfm-mcats10-lbs-rkopt-ft4 sfm-mcats10 10010 --num_epochs 100 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbsn-correspdv-rkopt-ft-100ep-stage3/params_100.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 ama-female1-lbs-rkopt-b16 ama-female1 10004 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-T_samba_small/cnn-params_10.pth --batch_size 16 --nsample 64 --use_human
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 ama-female-lbs-rkopt-ft1-100ep ama-female 10060 --num_epochs 100 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-T_samba_small/cnn-params_10.pth --model_path logdir/ama-female1-lbs-rkopt-b16-ftcse1e504/params_26.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --use_human
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 a-eagle-lbs-rkopt-b16-cam-cse a-eagle 10001 --num_epochs 30 --lbs --root_opt --ks_opt --batch_size 16 --nsample 64 --ft_cse
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 a-eagle-lbs-rkopt-noise30-ft1-2 a-eagle 10011 --num_epochs 30 --lbs --root_opt --ks_opt --model_path logdir/a-eagle-lbs-rkopt-b16-noise30-cse/params_30.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --ftcse_steps 0.2 --mtcse_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize
bash scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 ama-female-lbs-rkopt-ft1 ama-female 10060 --num_epochs 30 --lbs --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-T_samba_small/cnn-params_10.pth --model_path logdir/ama-female1-lbs-rkopt-b16-hf/params_30.pth --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --ftcse_steps 0.2 --mtcse_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --use_human
```



Adaptation to another set of videos
```
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 nerfies_cat_807-lbsn-correspv-rkopt-ft-noreinitb-fix nerfies_cat_807 10031 --num_epochs 30 --lbs --use_corresp --use_viser --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbsn-correspv-rkopt-100ep-b32-2/params_100.pth --use_proj --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --preload
```

Stage-wise finetuning
```
python submit.py scripts/template-stage.sh 0,1,2,3,4,5,6,7 sfm-mcats10 1000
```

To traing pose predictor
```
bash scripts/template-mgpu.sh 0 test T_samba_small 10001 --num_epochs 30 --lbs --root_opt --ks_opt --nopreload --use_human --warmup_pose_ep 10
```

To evaluate
```
python render_vis.py --testdir logdir/T_swing1-lbs-rkopt-ft2/ --outpath logdir/T_swing1-lbs-rkopt-ft2/T_swing1-eval --seqname T_swing1 --test_frames "{0}" --vp 0 --gtdir ~/data/AMA/T_swing/meshes/
```

To visualize matchings
```
CUDA_VISIBLE_DEVICES=1 bash scripts/render_match.sh sfm-mcats10 logdir/sfm-mcats10-lbsn-correspdv01d-rkopt-u-100ep/params_100.pth "0 200" "--queryfw --root_opt --lbs"
```

To draw root pose trajectory
```
python scripts/render_root.py --testdir logdir/syn-eagle-15h-hr-lbs-corresp-root-nowarmup-64-16-olr/
```

To re-render meshes
```
CUDA_VISIBLE_DEVICES=1 bash scripts/render_vids.sh sfm-mcats10 logdir/sfm-mcats10-lbs-rkopt-ft4/params_30.pth "8 9" "--sample_grid3d 128 --queryfw --root_opt --lbs --render_size 16 --ndepth 16 --nouse_viser --nouse_corresp --nouse_proj"
```
## Additional Notes
- wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl

- use --full_mesh at extraction time to extract a complete surface (disable visibility check)

- use a surface-like mlp for mlp-based deformation field --init_beta 1./100 --sil_wt 1.

- need to set --perturb 0 at test

### Usage of preload
```
python preload.py --seqname sfm-mcats10
```
this will generate records for forward pairs of seqs under sfm-mcats10
at training time, add --preload to the script


### Human reconstruction
use --use_human
