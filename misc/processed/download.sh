seqname=$1
rootdir=$PWD

mkdir -p database/DAVIS/ && cd "$_"
wget $(cat $rootdir/misc/processed/processed-$seqname.txt)
unzip '*.zip'; cd ../../
