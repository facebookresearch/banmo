# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# fine-tunning from a prior model
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
prior_model=$6
num_epochs=30
batch_size=256

model_prefix=prior-$seqname-e$num_epochs-b$batch_size
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

# mode: line load
# difference from template.sh: 
# 1) load prior model
# 2) disable warmup shape flag
# 3) update near-far plane and bound from begining
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --model_path $prior_model \
  --pose_cnn_path $pose_cnn_path \
  --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --nf_reset 0 --bound_reset 0 \
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
loadname=${model_prefix}-init
savename=${model_prefix}-ft1
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 1 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1\
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: fine tunning without pose correction
loadname=${model_prefix}-ft1
savename=${model_prefix}-ft2
num_epochs=$((num_epochs/2))
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --${use_symm}symm_shape \
  --${use_human}use_human

# mode: final tunning with larger rgb loss wt and reset beta
loadname=${model_prefix}-ft2
savename=${model_prefix}-ft3
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_latest.pth \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 0 \
  --dskin_steps 0 --fine_steps 0 --noanneal_freq \
  --img_wt 1 --reset_beta --eikonal_loss \
  --${use_symm}symm_shape \
  --${use_human}use_human
