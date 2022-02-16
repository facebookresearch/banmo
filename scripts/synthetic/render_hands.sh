# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
python scripts/synthetic/render_synthetic.py --outdir a-hands-1 --model hands \
    --nframes 150 --alpha 0.1 --init_a 0.2 --xspeed 1 --rot_axis x
python scripts/synthetic/render_synthetic.py --outdir a-hands-2 --model hands \
    --nframes 150 --alpha 0.3 --init_a 0.3 --xspeed 1 --rot_axis x
python scripts/synthetic/render_synthetic.py --outdir a-hands-3 --model hands \
    --nframes 150 --alpha 0.3 --init_a 0.2 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-hands-4 --model hands \
    --nframes 150 --alpha 0.3 --init_a 0.7 --xspeed 1
python scripts/synthetic/render_synthetic.py --outdir a-hands-5 --model hands \
    --nframes 150 --alpha 0.3 --init_a 0.7 --xspeed 1 --rot_axis x
seqname=a-hands-1
python preprocess/compute_dp.py $seqname n
seqname=a-hands-2
python preprocess/compute_dp.py $seqname n
seqname=a-hands-3
python preprocess/compute_dp.py $seqname n
seqname=a-hands-4
python preprocess/compute_dp.py $seqname n
seqname=a-hands-5
python preprocess/compute_dp.py $seqname n

# flow
seqname=a-hands-1
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-hands-2
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-hands-3
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-hands-4
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -

seqname=a-hands-5
cd third_party/vcnplus
  bash compute_flow.sh $seqname
cd -
