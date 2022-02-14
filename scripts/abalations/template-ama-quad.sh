# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=ama-female-cam # 2600
#seqname=ama-female # 2600
num_epochs=120
addname=b64-quad
addr=10005
use_human=

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

# mode: line load
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --lineload --batch_size 64 --nsample 4 \
  --warmup_shape_ep 5 \
  --nounc_filter \
  --${use_human}use_human
  #--pose_cnn_path $pose_cnn_path \
