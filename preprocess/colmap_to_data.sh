seqname=$1
nerfies_path=${pwd}/third_party/nerfies-0.1

conda activate nerfies
python third_party/nerfies-0.1/notebooks/run_colmap.py $seqname
python third_party/nerfies-0.1/notebooks/save_to_cams.py $seqname
python preprocess/colmap_to_database.py $nerfies_path $seqname
conda deactivate

python preprocess/compute_dp.py nerfies_$seqname

cd third_party/vcnplus
bash compute_flow.sh nerfies_$seqname
