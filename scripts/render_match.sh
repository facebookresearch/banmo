# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
model_path=$1
match_frames=$2
add_args=${*: 3:$#-1}
testdir=${model_path%/*} # %: from end

python scripts/visualize/match.py --flagfile=$testdir/opts.log \
                --model_path $model_path \
                --match_frames "$match_frames" \
                  $add_args
