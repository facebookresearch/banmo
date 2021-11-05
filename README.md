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
```
To optimize
```
bash scripts/template.sh logname dataname address --flowbw
python submit.py scripts/template.sh syn-eagled-15h-lbs-corresp-can15 syn-eagled-15h 10034 --lbs --num_epochs 30 --use_corresp
bash scripts/template.sh syn-eagled-15h-lbs-corresp-root syn-eagled-15h 10045 --num_epochs 30 --use_corresp --root_opt --nouse_cam --lbs
python submit.py scripts/template.sh ncat-509-correspdp-lbs-ropt-10f nerfies_cat_509 10170 --num_epochs 40 --use_corresp --lbs --flow_dp
python submit.py scripts/template.sh redo-mcat-6-corresp-lbs-flowdp-rot1 mcat_6 10168 --num_epochs 40 --use_corresp --lbs --flow_dp --rot_angle 1
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 cat_601-lbs-correspd-root-cnn-8gpu cat_601 10041 --num_epochs 30 --use_corresp --lbs --root_opt --flow_dp --nouse_cam --cnn_root
bash scripts/template-mgpu.sh 0 test-sfm cat_905 10010 --num_epochs 30 --use_corresp --root_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth  --lbs
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 sfm-mcats4-lbsn-correspdv-ropt3 sfm-mcats4 10002 --num_epochs 30 --lbs --use_corresp --flow_dp --use_viser --root_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 sfm-mcats10-lbsn-correspdv-rkopt-ft-100ep-stage3 sfm-mcats10 10001 --num_epochs 100 --lbs --use_corresp --use_viser --flow_dp --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth  --model_path logdir/sfm-mcats10-lbsn-correspdv-rkopt-ft-100ep2/params_100.pth --use_proj --warmup_init_steps 0 --reinit_bone_steps 0 --noanneal_freq --warmup_steps 0 --freeze_proj --noflow_dp
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 sfm-mcats4-lbsn-correspdv-rkopt-100ep-from-807 sfm-mcats4 10001 --num_epochs 100 --lbs --use_corresp --use_viser --flow_dp --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth  --model_path logdir/nerfies_cat_807-lbsn-correspdv-ropt-100ep-005noise/params_100.pth --use_proj --warmup_init_steps 0 --reinit_bone_steps 0 --noanneal_freq --warmup_steps 0 --freeze_proj --noflow_dp --dskin_steps 0.5 --nf_reset 0 --nouse_resize
```

Adaptation to another set of videos
```
python submit.py scripts/template-mgpu.sh 0,1,2,3,4,5,6,7 nerfies_cat_807-lbsn-correspv-rkopt-ft-noreinitb-fix nerfies_cat_807 10031 --num_epochs 30 --lbs --use_corresp --use_viser --root_opt --ks_opt --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth --model_path logdir/sfm-mcats10-lbsn-correspv-rkopt-100ep-b32-2/params_100.pth --use_proj --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize --preload
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
bash scripts/render_result.sh syn-eagled-15h logdir/syn-eagled-15h-test3/params_20.pth --lbs --sample_grid3d 128 --queryfw
bash scripts/render_result.sh mcat_6 logdir/mcat-6-corresp-lbs-flowdp-lr10-rot1-rmdp/params_11.pth --sample_grid3d 128 --queryfw --lbs
CUDA_VISIBLE_DEVICES=1 bash scripts/render_result.sh mcats logdir/mcats5-lbs-correspd-ropt-100ep/params_100.pth 9 --sample_grid3d 16 --queryfw --use_corresp --nonerf_vis
CUDA_VISIBLE_DEVICES=1 bash scripts/render_result.sh mcats logdir/mcats-lbs-correspd-root-exp-8gpu/params_100.pth 50 --sample_grid3d 128 --queryfw --lbs --root_opt --nouse_cam --explicit_root
```
## Additional Notes
wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl

### don't use visibility filtering at extraction time --noner_vis

### use a surface-like mlp for mlp-based deformation field --init_beta 1./100 --sil_wt 1.

### for root body optimization: set --lbs_reinit_epochs 5

### need to set --perturb 0 at test

### Usage of preload
```
python preload.py --seqname sfm-mcats10
```
this will generate records for forward pairs of seqs under sfm-mcats10
at training time, add --preload to the script
