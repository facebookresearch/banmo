# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
num_epochs=120
batch_size=256

model_prefix=short-$seqname-e$num_epochs-b$batch_size
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
  --eikonal_wt 0.001 --nsample 4 \
  --${use_symm}symm_shape \
  --${use_human}use_human
