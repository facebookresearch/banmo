# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Use precomputed root body poses
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
num_epochs=120
batch_size=256

model_prefix=known-cam-$seqname-e$num_epochs-b$batch_size

# mode: line load
# difference from template.sh
# 1) remove pose_net path flag
# 2) add use_rtk_file flag
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --use_rtk_file \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --eikonal_wt 0.001 --nsample 4 \
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
loadname=${model_prefix}-init
savename=${model_prefix}-ft1
num_epochs=$((num_epochs/4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1\
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: fine tune with active+fine samples, large rgb loss wt and reset beta
loadname=${model_prefix}-ft1
savename=${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --eikonal_wt 0.1 --nsample 4 \
  --${use_symm}symm_shape \
  --${use_human}use_human
