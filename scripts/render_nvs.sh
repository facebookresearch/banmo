dev=0
seqname=tiger-100
#seqname=sfm-mcats10
model_path=/private/home/gengshany/data/old_checkpoints_4/tiger-100-lbs-rkopt-30-frz-ftloss-nocorresp/params_22.pth
#model_path=/private/home/gengshany/data/old_checkpoints_4/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/params_90.pth
retarget_path=/private/home/gengshany/data/old_checkpoints_4/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/params_90.pth
prefix=tmp/nvs
vidid=0
#vidid=9
scale=0.5
maxframe=90
blt=24

rootdir1=/private/home/gengshany/data/old_checkpoints_4/sfm-mcats10-lbs-rkopt-90-noaccu-b16-ft2-ftcse/sfm-mcats10-\{9\}-cgray-ctrajs-
rootdir2=/private/home/gengshany/data/old_checkpoints_4/tiger-100-lbs-rkopt-30-frz-ftloss-nocorresp/tiger-100-\{0\}-vgray-ctrajs-
rootdir3=/private/home/gengshany/data/old_checkpoints_4/tiger-100-lbs-rkopt-30-frz-ftloss-nocorresp/tiger-100-\{0\}-fgray-ctrajs-

CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
  --model_path $model_path \
  --vidid $vidid \
  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
  --chunk 2048 \
  --rootdir $rootdir3 --nvs_outpath $prefix-traj \
  --retarget_path $retarget_path

#CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
#  --model_path $model_path \
#  --vidid $vidid \
#  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
#  --chunk 2048 \
#  --rootdir $rootdir1 --nvs_outpath $prefix-traj

#CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
#  --model_path $model_path \
#  --vidid $vidid \
#  --scale $scale --maxframe $maxframe --lbs --bullet_time -1 \
#  --rootdir tmp/traj-freeze/ --nvs_outpath $prefix-frz

#CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname $seqname \
#  --model_path $model_path \
#  --vidid $vidid \
#  --scale $scale --maxframe $maxframe --lbs --bullet_time $blt --nvs_outpath $prefix-b00 \
#  --retarget_path $retarget_path
