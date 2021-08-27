seqname=$1
nerfies_path=/private/home/gengshany/code/vid2shape/third_party/nerfies-0.1

rm -rf third_party/nerfies-0.1/dataset/$seqname
unzip ~/dropbox/viser/$seqname-nerfies.zip -d third_party/nerfies-0.1

. activate nerfies
python scripts/colmap_to_database.py $nerfies_path $seqname
. deactivate

python scripts/compute_dp.py nerfies_$seqname

. activate viser
cd third_party/vcnplus
bash compute_flow.sh nerfies_$seqname
