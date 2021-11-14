gpus=$1
seqname=T_samba
num_epochs=200
addname=b16
addr=10004
use_human=

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
else
  pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
fi
echo $pose_cnn_path

savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --batch_size 16 --nsample 64 \
  --${use_human}use_human