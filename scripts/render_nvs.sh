dev=0
seqname=sfm-mcats10
model_path=logdir/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/params_90.pth
prefix=tmp/nvs
vidid=3
scale=2

CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe 90 --lbs --bullet_time -1 \
  --rootdir tmp/traj-freeze/ --nvs_outpath $prefix-frz
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe 90 --lbs --bullet_time 0 --nvs_outpath $prefix-b00
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe 90 --lbs --bullet_time 85 --nvs_outpath $prefix-b85
