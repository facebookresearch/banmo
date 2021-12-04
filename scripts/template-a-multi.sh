gpus=$1
seqname=a-hands-apart
#seqname=a-eagle
num_epochs=30
addname=b16-cam-noviser
addr=10005
use_human=

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
#  --${use_human}use_human \
#  --batch_size 16 --nsample 64 \
#  --noroot_opt  --noks_opt \
#  --nouse_viser --nouse_proj \
##  --pose_cnn_path $pose_cnn_path \
#
##TODO proj end
#loadname=${model_prefix}-init
#savename=${model_prefix}-ft1-2
#bash scripts/template-mgpu.sh $gpus $savename \
#    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --model_path logdir/$loadname/params_$num_epochs.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
#  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
#  --${use_human}use_human \
#  --noroot_opt  --noks_opt \
#  --nouse_viser --nouse_proj \
##  --proj_end 0.001 \
##  --ft_cse --ftcse_steps 0
##  --pose_cnn_path $pose_cnn_path \

#TODO proj end
loadname=${model_prefix}-ft1-2
savename=${model_prefix}-ft2-2
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --model_path logdir/$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  --${use_human}use_human \
  --noroot_opt  --noks_opt \
  --nouse_viser --nouse_proj \
#  --proj_end 0.001 \
#  --ft_cse --ftcse_steps 0
#  --pose_cnn_path $pose_cnn_path \