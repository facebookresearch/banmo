export MKL_SERVICE_FORCE_INTEL=1
dev=0
ngpu=1
nepoch=20
bs=4
img_size=512

logname=$1
seqname=$2
address=$3
add_args=${*: 4:$#-1}

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch\
                    --master_port $address \
                    --nproc_per_node=$ngpu main.py \
                    --num_epochs $nepoch \
                    --batch_size $bs \
                    --img_size $img_size \
                    --seqname $seqname \
                    --logname $logname \
                    $add_args