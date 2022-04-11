# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
sample_grid3d=256
seqname=shiba-haru-1
model_path=logdir/$seqname-e120-b256-final-ft1/params_latest.pth
cmd=`echo bash scripts/render_mgpu.sh 0 $seqname $model_path '"0 1 2 3 4 5"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 1 $seqname $model_path '"6 7 8 9 10 11"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"

seqname=cat-pikachiu
model_path=logdir/$seqname-e120-b256-final-ft1/params_latest.pth
cmd=`echo bash scripts/render_mgpu.sh 2 $seqname $model_path '"0 1 2 3 4 5"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 3 $seqname $model_path '"6 7 8 9 10"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"

seqname=adult7
model_path=logdir/$seqname-e120-b256-final-ft1/params_latest.pth
cmd=`echo bash scripts/render_mgpu.sh 4 $seqname $model_path '"0 1 2 3 4"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 5 $seqname $model_path '"5 6 7 8 9"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"

seqname=cat-coco
model_path=logdir/$seqname-e120-b256-final-ft1/params_latest.pth
cmd=`echo bash scripts/render_mgpu.sh 6 $seqname $model_path '"0 1 2"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"
cmd=`echo bash scripts/render_mgpu.sh 7 $seqname $model_path '"3 4 5"' $sample_grid3d`; screen -dmS "if" bash -c "$cmd"

