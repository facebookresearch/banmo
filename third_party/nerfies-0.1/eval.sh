dev=0
ngpu=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export DATASET_PATH=dataset/syn-eagled-100h-cam-scale/
export EXPERIMENT_PATH=logs/syn-eagled-100h-cam-scale/

python eval.py \
    --data_dir $DATASET_PATH \
    --exp_dir $experiment_path \
    --gin_configs configs/gpu_quarterhd_4gpu.gin
    #--gin_configs configs/ablation.gin
