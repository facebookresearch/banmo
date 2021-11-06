gpus=$1
seqname=$2
num_epochs=30
extname=lbs-rkopt
addr=$3

loadname=ft1new
savename=ft2new
bash scripts/template-mgpu.sh $gpus $seqname-$extname-$savename $seqname $addr \
  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth \
  --model_path logdir/$seqname-$extname-$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize

loadname=ft2new
savename=ft3new
bash scripts/template-mgpu.sh $gpus $seqname-$extname-$savename $seqname $addr \
  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth \
  --model_path logdir/$seqname-$extname-$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize

loadname=ft3new
savename=ft4new
bash scripts/template-mgpu.sh $gpus $seqname-$extname-$savename $seqname $addr \
  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth \
  --model_path logdir/$seqname-$extname-$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize

loadname=ft4new
savename=ft5new
bash scripts/template-mgpu.sh $gpus $seqname-$extname-$savename $seqname $addr \
  --num_epochs $num_epochs --lbs --root_opt --ks_opt \
  --pose_cnn_path logdir/pose-occ03-cat_600-lbs-corresp-ropt-8gpu/cnn-params_10.pth \
  --model_path logdir/$seqname-$extname-$loadname/params_$num_epochs.pth \
  --warmup_init_steps 0 --warmup_steps 0 --nf_reset 0 --dskin_steps 0 \
  --fine_steps 0.2 --noanneal_freq --freeze_proj --nouse_resize
