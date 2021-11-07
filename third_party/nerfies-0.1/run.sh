dev=0,1,2,3,4,5,6,7
ngpu=8
#dev=0
#ngpu=1
seqname=cat_807
logname=cat_807
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export DATASET_PATH=dataset/$seqname
export EXPERIMENT_PATH=logs/$logname
#export NCCL_DEBUG=INFO 
python train.py \
    --data_dir $DATASET_PATH \
    --exp_dir $EXPERIMENT_PATH \
    --gin_configs configs/ablation.gin
    #--gin_configs configs/gpu_quarterhd.gin
    #--gin_configs configs/gpu_quarterhd_4gpu.gin
    #--gin_configs configs/test_vrig.gin
