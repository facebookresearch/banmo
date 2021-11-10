gpus=$1
seqname=$2
addname=$3
addr=$4
use_human=$5

num_epochs=30
extname=lbs-rkopt
model_prefix=$seqname-$extname-$addname
pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
#pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth

#savename=init
#bash scripts/template-mgpu.sh $gpus $seqname-$extname-$addname$savename $seqname $addr \
#  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth \
#  --batch_size 16 --nsample 64

#loadname=''
#savename='-ft1'
#bash scripts/template-mgpu.sh $gpus $model_prefix$savename \
#    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
#  --pose_cnn_path $pose_cnn_path \
#  --model_path logdir/$model_prefix$loadname/params_$num_epochs.pth \
#  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
#  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
#  --${use_human}use_human

loadname='-ft1'
savename='-ft2'
bash scripts/template-mgpu.sh $gpus $model_prefix$savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$model_prefix$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  --${use_human}use_human

loadname='-ft2'
savename='-ft3'
bash scripts/template-mgpu.sh $gpus $model_prefix$savename \
    $seqname $addr --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path logdir/$model_prefix$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  --${use_human}use_human
