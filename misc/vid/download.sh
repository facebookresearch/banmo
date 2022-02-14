seqname=$1
rootdir=$PWD

mkdir -p raw/$seqname && cd "$_"
# download the video
wget $(cat $rootdir/misc/vid/vid-$seqname.txt)
cd ../../
