# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
python scripts/synthetic/render_synthetic.py --outdir a-eagle-1 --model eagle \
    --nframes 150 --alpha 0.2 --init_a 0.0 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-eagle-2 --model eagle \
    --nframes 150 --alpha 0.2 --init_a 0.2 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-eagle-3 --model eagle \
    --nframes 150 --alpha 0.2 --init_a 0.4 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-eagle-4 --model eagle \
    --nframes 150 --alpha 0.2 --init_a 0.6 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-eagle-5 --model eagle \
    --nframes 150 --alpha 0.2 --init_a 0.8 --xspeed 1
seqname=a-eagle-1
python preprocess/compute_dp.py $seqname n
seqname=a-eagle-2
python preprocess/compute_dp.py $seqname n
seqname=a-eagle-3
python preprocess/compute_dp.py $seqname n
seqname=a-eagle-4
python preprocess/compute_dp.py $seqname n
seqname=a-eagle-5
python preprocess/compute_dp.py $seqname n
 
# flow
seqname=a-eagle-1
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-eagle-2
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-eagle-3
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-eagle-4
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-eagle-5
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -
