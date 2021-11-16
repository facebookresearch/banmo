python scripts/render_synthetic.py --outdir a-hands-test --model hands \
    --nframes 2 --alpha 0.1 --init_a 0.2 --xspeed 1 --rot_axis x
#python scripts/render_synthetic.py --outdir a-hands-1 --model hands \
#    --nframes 150 --alpha 0.1 --init_a 0.2 --xspeed 1 --rot_axis x
#python scripts/render_synthetic.py --outdir a-hands-2 --model hands \
#    --nframes 150 --alpha 0.3 --init_a 0.3 --xspeed 1 --rot_axis x
#python scripts/render_synthetic.py --outdir a-hands-3 --model hands \
#    --nframes 150 --alpha 0.3 --init_a 0.2 --xspeed 1
#python scripts/render_synthetic.py --outdir a-hands-4 --model hands \
#    --nframes 150 --alpha 0.3 --init_a 0.7 --xspeed 1
#python scripts/render_synthetic.py --outdir a-hands-5 --model hands \
#    --nframes 150 --alpha 0.3 --init_a 0.7 --xspeed 1 --rot_axis x
#seqname=a-hands-1
#python scripts/compute_dp.py $seqname y
#seqname=a-hands-2
#python scripts/compute_dp.py $seqname y
#seqname=a-hands-3
#python scripts/compute_dp.py $seqname y
#seqname=a-hands-4
#python scripts/compute_dp.py $seqname y
#seqname=a-hands-5
#python scripts/compute_dp.py $seqname y
#
## flow
#seqname=a-hands-1
#cd third_party/vcnplus
#  bash compute_flow.sh $seqname
#cd -
#
#seqname=a-hands-2
#cd third_party/vcnplus
#  bash compute_flow.sh $seqname
#cd -
#
#seqname=a-hands-3
#cd third_party/vcnplus
#  bash compute_flow.sh $seqname
#cd -
#
#seqname=a-hands-4
#cd third_party/vcnplus
#  bash compute_flow.sh $seqname
#cd -
#
#seqname=a-hands-5
#cd third_party/vcnplus
#  bash compute_flow.sh $seqname
#cd -
#
#python preload.py --seqname a-hands
