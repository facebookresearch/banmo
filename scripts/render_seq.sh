# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# running mesh extraction in parallel
seqname=cat-socks
model_path=logdir/cat-socks-e120-b256-ft3/params_latest.pth
sample_grid3d=64
cmd=`echo bash scripts/render_mgpu.sh 0 $seqname $model_path '"0"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 1 $seqname $model_path '"1"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 2 $seqname $model_path '"2"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 3 $seqname $model_path '"3"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
