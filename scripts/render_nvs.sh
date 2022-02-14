# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
seqname=$2
model_path=$3
vidid=$4   # pose traj
rootid=$5  # root traj

save_prefix=tmp/nvs
maxframe=90
scale=1
sample_grid3d=256
add_args="--sample_grid3d ${sample_grid3d} --full_mesh \
  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"

# extrat meshes
testdir=${model_path%/*} # %: from end
CUDA_VISIBLE_DEVICES=$dev python extract.py --flagfile=$testdir/opts.log \
                  --model_path $model_path \
                  --test_frames $rootid \
                  $add_args

# re-render the trained sequence
prefix=$testdir/$seqname-{$vidid}
trgpath=$prefix-vgray
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/render_vis.py --testdir $testdir \
                     --outpath $trgpath --vp 0 \
                     --seqname $seqname \
                     --test_frames {$vidid} \
                     --root_frames {$rootid} \
                     --gray_color

rootdir=$trgpath-ctrajs-
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0\
  --rootdir $rootdir --nvs_outpath $save_prefix-traj
