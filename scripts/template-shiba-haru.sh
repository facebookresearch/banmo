gpus=$1
seqname=shiba-haru
num_epochs=150 # ~750 frames
addname=b16-flowdp
addr=10004
use_human=no

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
else
  pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
fi
echo $pose_cnn_path

#savename=${model_prefix}-init
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --batch_size 16 --nsample 64 \
#  --flow_dp \
#  --${use_human}use_human

#loadname=shiba-haru-lbs-rkopt-150-b16-flowdp-ft1
#savename=${model_prefix}-ft2
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs 30 --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$loadname/params_105.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --dskin_steps 0.2 \
#  --fine_steps 0.2 --noanneal_freq --nouse_resize \
#  --flow_dp \
#  --bound_reset 0.2 \
#  --${use_human}use_human
#  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \

#loadname=${model_prefix}-ft2
#savename=${model_prefix}-ft3
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs 30 --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$loadname/params_30.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --dskin_steps 0.2 \
#  --fine_steps 0.2 --noanneal_freq --nouse_resize \
#  --flow_dp \
#  --bound_reset 0.2 \
#  --${use_human}use_human
#  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \

#loadname=${model_prefix}-ft3
#savename=${model_prefix}-ft4
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs 30 --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$loadname/params_30.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --dskin_steps 0.2 \
#  --fine_steps 0.2 --noanneal_freq --nouse_resize \
#  --bound_reset 0.2 \
#  --${use_human}use_human
#  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
#  #--flow_dp \

loadname=${model_prefix}-ft4
savename=${model_prefix}-ft5
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs 30 --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$loadname/params_30.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0.2 --dskin_steps 0.2 \
  --fine_steps 0.2 --noanneal_freq --nouse_resize \
  --bound_reset 0.2 \
  --${use_human}use_human
  #--fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  #--flow_dp \
