gpus=$1
seqname=panda
num_epochs=60 # ~380 frames
addname=b16
addr=10004
use_human=no

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

loadname=${model_prefix}-init
savename=${model_prefix}-ft1
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  --${use_human}use_human