gpus=$1
seqname=nerfies_cat_807
num_epochs=30
addname=frz
addr=10004
use_human=no

model_prefix=$seqname-lbs-rkopt-$num_epochs-$addname
if [ "$use_human" = "" ]; then
  pose_cnn_path=logdir/pose-occ03-T_samba_small/cnn-params_10.pth
else
  pose_cnn_path=logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth
fi
echo $pose_cnn_path

#TODO freeze coarse; model_patp
model_path=logdir/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/params_90.pth

#TODO proj_end 0
savename=test-3
#savename=${model_prefix}-ft-loss
bash scripts/template-mgpu.sh $gpus $savename \
    $seqname $addr  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path $pose_cnn_path \
  --model_path $model_path \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize \
  --freeze_coarse\
  --proj_end 0.001 --loadid0 385 --loadvid 4\
  --${use_human}use_human
