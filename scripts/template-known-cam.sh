# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
gpus=$1
seqname=$2
addr=$3
use_human=$4
use_symm=$5
num_epochs=120
batch_size=256

# mode: line load
# nouse_cc is enabled for disconnected objects such as hands
savename=known-cam-${model_prefix}
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs \
  --use_rtk_file \
  --warmup_shape_ep 5 --warmup_rootmlp \
  --lineload --batch_size $batch_size\
  --nouse_cc \
  --${use_symm}symm_shape \
  --${use_human}use_human
