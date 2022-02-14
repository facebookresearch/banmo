# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#bash scripts/render_mgpu.sh 0 logdir/$seqname-ft3/params_latest.pth \
#        "0 1 2 3 4 5 6 7 8 9 10"'
## argv[1]: gpu id
## argv[2]: sequence name
## argv[3]: weights path
## argv[4]: video id separated by space
## argv[5]: resolution of running marching cubes (256 by default)

dev=$1
seqname=$2
modelpath=$3
vids=$4
sample_grid3d=$5

CUDA_VISIBLE_DEVICES=${dev} bash scripts/render_vids.sh \
  ${seqname} ${modelpath} ${vids} \
  "--sample_grid3d ${sample_grid3d} --full_mesh \
  --nouse_corresp --nouse_embed --nouse_proj --render_size 2 --ndepth 2"
