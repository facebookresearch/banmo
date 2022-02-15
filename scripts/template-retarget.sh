# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=$2 # 630 frames
addr=$3
use_human=$4
use_symm=$5
driven_model=$6
num_epochs=120
batch_size=256

model_prefix=driver-$seqname-e$num_epochs-b$batch_size
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

# mode: pose correction
# 0-80% body pose with proj loss, 80-100% gradually add all loss
# freeze shape/feature etc
savename=${model_prefix}
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --pose_cnn_path $pose_cnn_path \
  --model_path $driven_model \
  --lineload --batch_size $batch_size \
  --warmup_steps 0 --nf_reset 0 --bound_reset 1 \
  --dskin_steps 0 --fine_steps 1 --noanneal_freq \
  --freeze_proj --proj_end 1 --frzroot_start 0.4 --frzbody_end 0.2\
  --warmup_rootmlp \
  --${use_symm}symm_shape \
  --${use_human}use_human
