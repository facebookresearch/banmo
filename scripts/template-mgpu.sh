# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
export MKL_SERVICE_FORCE_INTEL=1
dev=$1
ngpu=`echo $dev |  awk -F '[\t,]' '{print NF-1}'`
ngpu=$(($ngpu + 1 ))
echo "using "$ngpu "gpus"

logname=$2
seqname=$3
address=$4
add_args=${*: 4:$#-1}

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch\
                    --master_port $address \
                    --nproc_per_node=$ngpu main.py \
                    --ngpu $ngpu \
                    --seqname $seqname \
                    --logname $logname \
                    $add_args
