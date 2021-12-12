gpus=$1
seqname=adult7 # 630 frames
num_epochs=120
addname=b128-sm2nd01
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
  --lineload --batch_size 128 --nsample 4 --nouse_resize \
  --img_wt 0.1 --sil_wt 0.1 --proj_wt 0.02\
  --num_bones 64 \
  --symm_shape \
  --root_sm \
  --${use_human}use_human
  #--lineload --batch_size 128 --nsample 8 --nouse_resize \
  #--flow_dp \

loadname=${model_prefix}-init
savename=${model_prefix}-ft1
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_$num_epochs.pth \
  --lineload --batch_size 128 --nsample 4 \
  --img_wt 0.1 --sil_wt 0.1 --proj_wt 0.02\
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --bound_reset 0.2 \
  --dskin_steps 0.2 --fine_steps 0.2 --noanneal_freq --nouse_resize \
  --num_bones 64 \
  --symm_shape \
  --root_sm \
  --${use_human}use_human
  #--lineload --batch_size 128 --nsample 8 \
  #--flow_dp \
  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \

#loadname=${model_prefix}-ft1
#savename=${model_prefix}-ft2
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$loadname/params_$num_epochs.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --bound_reset 0.2 \
#  --dskin_steps 0.2 --fine_steps 0.2 --noanneal_freq --nouse_resize \
#  --num_bones 64 \
#  --flow_dp \
#  --${use_human}use_human
#  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
