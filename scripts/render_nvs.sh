dev=1
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname nerfies_cat_807 \
  --model_path logdir/nerfies_cat_807-lbsn-correspdv-ropt-f1x08-a04-120ep/params_116.pth \
  --scale 0.5 --maxframe 90 --lbs --bullet_time -1 --rootdir tmp/traj-freeze/ --nvs_outpath tmp/frz
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname nerfies_cat_807 \
  --model_path logdir/nerfies_cat_807-lbsn-correspdv-ropt-f1x08-a04-120ep/params_116.pth \
  --scale 0.5 --maxframe 90 --lbs --bullet_time 0 --nvs_outpath tmp/b00
CUDA_VISIBLE_DEVICES=$dev python nvs.py --seqname nerfies_cat_807 \
  --model_path logdir/nerfies_cat_807-lbsn-correspdv-ropt-f1x08-a04-120ep/params_116.pth \
  --scale 0.5 --maxframe 90 --lbs --bullet_time 85 --nvs_outpath tmp/b85
