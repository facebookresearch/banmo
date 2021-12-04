gpus=$1
seqname=nerfies_cat_807
num_epochs=30
addname=b128line-xyt
addr=10004
use_human=no

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
else
  pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
fi
echo $pose_cnn_path

#TODO flodp
savename=${model_prefix}-init
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --lineload --batch_size 128 --nsample 8\
  --warmup_init_steps 0 --warmup_steps 0 \
  --fine_steps 0.0 --noanneal_freq --nouse_resize \
  --${use_human}use_human
  #--batch_size 16 --nsample 64 \
  #--flow_dp \

#loadname=${model_prefix}-init
#savename=${model_prefix}-ft1
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$loadname/params_$num_epochs.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
#  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
#  --flow_dp \
#  --${use_human}use_human