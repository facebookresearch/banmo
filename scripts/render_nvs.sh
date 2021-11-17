dev=0
seqname=tiger-100
#seqname=sfm-mcats10
model_path=logdir/tiger-100-lbs-rkopt-30-frz-ftloss-nocorresp/params_22.pth
retarget_path=logdir/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/params_90.pth
prefix=tmp/nvs
vidid=0
scale=1.5
maxframe=90
blt=24

#CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
#  --model_path $model_path \
#  --vidid $vidid \
#  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
#  --rootdir tmp/traj-freeze/ --nvs_outpath $prefix-frz
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe $maxframe --lbs --bullet_time $blt --nvs_outpath $prefix-b00 \
  --retarget_path $retarget_path
