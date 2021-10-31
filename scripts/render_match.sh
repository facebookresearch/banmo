seqname=$1
model_path=$2
match_frames=$3
add_args=${*: 3:$#-1}

python match.py --seqname $seqname \
                --model_path $model_path \
                --match_frames "$match_frames" \
                --use_viser \
                  $add_args
