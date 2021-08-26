seqname=$1
nerfies_path=/private/home/gengshany/code/vid2shape/third_party/nerfies-0.1

. activate nerfies
unzip ~/dropbox/viser/$seqname-nerfies.zip -d third_party/nerfies-0.1
python $nerfies_path scripts/colmap_to_database.py $seqname
. deactivate

python scripts/compute_dp.py nerfies_$seqname

. activate viser
cd third_party/vcnplus
bash compute_flow.sh nerfies_$seqname
