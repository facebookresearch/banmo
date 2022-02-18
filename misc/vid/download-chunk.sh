seqname=$1
rootdir=$PWD

mkdir -p raw/$seqname && cd "$_"
# download the video
wget $(cat $rootdir/misc/vid/vidchunk-$seqname.txt) -O tmp.zip
unzip tmp.zip
rm tmp.zip
cd ../../
