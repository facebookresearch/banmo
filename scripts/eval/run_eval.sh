# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

## T_swing
#gtdir=database/T_swing/meshes/
#gt_pmat=database/T_swing/calibration/Camera1.Pmat.cal
#seqname=ama-female
#seqname_eval=T_swing1

## T_samba
#gtdir=database/T_samba/meshes/
#gt_pmat=database/T_samba/calibration/Camera1.Pmat.cal
#seqname=ama-female
#seqname_eval=T_samba1

## eagle
#gtdir=database/DAVIS/Meshes/Full-Resolution/a-eagle-1/
#gt_pmat=canonical
#seqname=a-eagle-1
#seqname_eval=$seqname

# hands
gtdir=database/DAVIS/Meshes/Full-Resolution/a-hands-1/
gt_pmat=canonical
seqname=a-hands-1
seqname_eval=$seqname

dev=$1
testdir=$2
#seqname=$3
#num_bones=$3
model_path=$testdir/params_latest.pth

test_frames={0}
symm_shape=no
frame_code=

#CUDA_VISIBLE_DEVICES=$dev python extract.py --seqname $seqname \
#                  --model_path $model_path \
#                  --test_frames $test_frames \
#  --sample_grid3d 256 --root_opt --lbs --full_mesh  --queryfw \
#  --nouse_corresp --nouse_viser --nouse_proj --render_size 2 --ndepth 2 \
#  --${symm_shape}symm_shape --${frame_code}frame_code \
##  --num_bones $num_bones

outfile=`cut -d/ -f2 <<<"${testdir}"`
CUDA_VISIBLE_DEVICES=$dev python render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-pred \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} \
 > tmp/$outfile.txt
CUDA_VISIBLE_DEVICES=$dev python render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval-gt \
 --seqname $seqname_eval --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat ${gt_pmat} --vis_gtmesh

ffmpeg -y -i $testdir/$seqname-eval-gt.mp4 \
          -i $testdir/$seqname-eval-pred.mp4 \
-filter_complex "[0:v][1:v]vstack=inputs=2[v]" \
-map "[v]" \
$testdir/$seqname-all.mp4
