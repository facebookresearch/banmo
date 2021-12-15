gpus=$1
seqname=sfm-mcats10
num_epochs=90 # 750 frames, 128 * 8 = 1024
addname=b128-accu
addr=10004
use_human=no

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=mesh_material/posenet/human.pth
else
  pose_cnn_path=mesh_material/posenet/quad.pth
fi
echo $pose_cnn_path

savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --lineload --batch_size 128 --nsample 4 \
  --use_accu 8 \
  --${use_human}use_human

loadname=${model_prefix}-init
savename=${model_prefix}-ft1
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --bound_reset 0.2 \
  --dskin_steps 0.2 --fine_steps 0.2 --noanneal_freq\
  --lineload --batch_size 128 --nsample 4\
  --use_accu 8 \
  --${use_human}use_human
