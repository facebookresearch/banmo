export MKL_SERVICE_FORCE_INTEL=1
dev=$1
ngpu=`echo $dev |  awk -F '[\t,]' '{print NF-1}'`
ngpu=$(($ngpu + 1 ))
echo "using "$ngpu "gpus"
nepoch=20
bs=4
img_size=512

logname=$2
seqname=$3
address=$4
add_args=${*: 4:$#-1}

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch\
                    --master_port $address \
                    --nproc_per_node=$ngpu main.py \
                    --ngpu $ngpu \
                    --num_epochs $nepoch \
                    --batch_size $bs \
                    --img_size $img_size \
                    --seqname $seqname \
                    --logname $logname \
                    $add_args
