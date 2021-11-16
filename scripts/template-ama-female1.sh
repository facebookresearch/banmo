gpus=$1
seqname=ama-female1
num_epochs=90
addname=b16
addr=10003
use_human=


if [ "$use_human" = "" ]; then
  pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
else
  pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
fi
echo $pose_cnn_path

model_prefix=$seqname-lbs-rkopt-nounc-$num_epochs-$addname
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --batch_size 16 --nsample 64 \
  --pose_cnn_path $pose_cnn_path \
  --${use_human}use_human \
  --nouse_unc

model_prefix=$seqname-lbs-rkopt-nofp-$num_epochs-$addname
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --batch_size 16 --nsample 64 \
  --pose_cnn_path $pose_cnn_path \
  --${use_human}use_human \
  --nouse_viser --nouse_proj

# no corresp
#model_prefix=$seqname-lbs-rkopt-noflow-$num_epochs-$addname
#savename=${model_prefix}-init
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --batch_size 16 --nsample 64 \
#  --pose_cnn_path $pose_cnn_path \
#  --${use_human}use_human \
#  --nouse_corresp

# referene
#model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
#savename=${model_prefix}-init
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --batch_size 16 --nsample 64 \
#  --pose_cnn_path $pose_cnn_path \
#  --${use_human}use_human \

# norp
#model_prefix=$seqname-lbs-rkopt-nocam-$num_epochs-$addname
#savename=${model_prefix}-init
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --batch_size 16 --nsample 64 \
#  --${use_human}use_human \
#  --nouse_cam \
#  #--pose_cnn_path $pose_cnn_path \
