det_path=../detectron2/projects/DensePose
cd $det_path
#config=configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml
#weights=https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl
config=configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml
weights=https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl
#config=configs/cse/densepose_rcnn_R_50_FPN_soft_animals_finetune_4k.yaml
#weights=https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_finetune_4k/253145793/model_final_8f8ba2.pkl
data=$1
python apply_net.py show \
            $config \
            $weights \
            $1 bbox,dp_vertex -v --output ./output.png
cd -
mv $det_path/output*.png ./
