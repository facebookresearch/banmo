dev=0
ngpu=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export DATASET_PATH=dataset/cat_807
export EXPERIMENT_PATH=logs/cat_807

python eval.py \
    --data_dir $DATASET_PATH \
    --exp_dir $EXPERIMENT_PATH \
    --gin_configs configs/gpu_quarterhd_4gpu.gin
    #--gin_configs configs/ablation.gin
