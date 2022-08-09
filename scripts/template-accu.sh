# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This scripts runs a single 8G memory GPU using 4x gradient accumulation (~32 hours).
# see --accu_steps flag.
# A rule of thumb is to set "num gpus" x "batch size" x "accu steps" ~= num frames (default number 512 suits for cat-pikachiu and human-hap)
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
num_epochs=120
batch_size=128
accu_steps=4

model_prefix=$seqname-e$num_epochs-b$batch_size
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

# mode: line load
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --accu_steps $accu_steps \
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
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1\
  --accu_steps $accu_steps \
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: fine tune with active+fine samples, large rgb loss wt and reset beta
loadname=${model_prefix}-ft1
savename=${model_prefix}-ft2
num_epochs=$((num_epochs*4))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --freeze_root --use_unc --img_wt 1 --reset_beta \
  --accu_steps $accu_steps \
  --eikonal_wt 0.1 --nsample 4 \
  --${use_symm}symm_shape \
  --${use_human}use_human
