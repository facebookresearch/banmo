# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
dev=$1
model_path=$2
save_prefix=tmp/nvs
maxframe=200
blt=24
scale=1

# re-render the trained sequence
seqname=cat-socks-sub
vidid=0
root_frames={0}  # camera traj

testdir=${model_path%/*} # %: from end
prefix=$testdir/$seqname-{$vidid}
trgpath=$prefix-vgray
rootdir=$trgpath-ctrajs-

## run mesh extraction
#bash scripts/render_vids.sh $seqname  $model_path "0" \
#"--sample_grid3d 64 --root_opt --lbs --full_mesh  --nouse_corresp --nouse_viser --nouse_proj --render_size 2 --ndepth 2 --nouse_human --queryfw"
#
## raw video
#python render_vis.py --testdir $testdir \
#                     --outpath $trgpath-vid \
#                     --seqname $seqname \
#                     --test_frames {$vidid} \
#                     --append_img yes \
#                     --append_render no
#
## masks
#python render_vis.py --testdir $testdir \
#                     --outpath $trgpath --vp 0 \
#                     --seqname $seqname \
#                     --test_frames {$vidid} \
#                     --root_frames $root_frames \
#                     --gray_color
#                     #--outpath $prefix-fgray --vp -1 \
# render
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0\
  --frame_code --env_fourier \
  --rootdir $rootdir --nvs_outpath $save_prefix-traj

# merge
ffmpeg -y -i $trgpath-vid.mp4 \
          -i $trgpath.mp4 \
          -i $save_prefix-traj-rgb.mp4 \
-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
-map "[v]" \
$save_prefix-traj-all.mp4

ffmpeg -y -i $save_prefix-traj-all.mp4 -vf "scale=iw/2:ih/2" $save_prefix-traj-all.gif
