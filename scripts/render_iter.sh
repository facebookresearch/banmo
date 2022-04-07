# render canonical model over iterations
dev=$1
testdir=$2
scale=1

save_prefix=$testdir/vis-iter
CUDA_VISIBLE_DEVICES=$dev python scripts/visualize/nvs_iter.py --flagfile=$testdir/opts.log \
  --model_path $testdir/params_latest.pth \
  --scale $scale \
  --chunk 2048 \
  --nouse_corresp --nouse_unc --perturb 0 --vidid 0 --nolbs\
  --nvs_outpath $save_prefix-iter
