seqname=$1
nerfies_path=/private/home/gengshany/code/vid2shape/third_party/nerfies-0.1

. activate nerfies
python third_party/nerfies-0.1/notebooks/run_colmap.py $seqname
python third_party/nerfies-0.1/notebooks/save_to_cams.py $seqname
python scripts/colmap_to_database.py $nerfies_path $seqname
. deactivate

python scripts/compute_dp.py nerfies_$seqname

cd third_party/vcnplus
bash compute_flow.sh nerfies_$seqname
