rootdir=/private/home/gengshany/dropbox/GengshanYang_project/test_videos/
tmpdir=/private/home/gengshany/data/tmp/
prefix=cat_5
filedir=$rootdir/$prefix
outdir=$rootdir/output
suffix=.MOV

counter=0
for infile in $filedir/*$suffix; do
  # filter videos
#  if [[ "$infile" == *"cat_5/IMG_8001.MOV"* ]]; then
  if [[ "$infile" != *"cat_5/IMG_8005.MOV"* ]]; then
    counter=$((counter+1))
    continue
  fi

  echo $infile  
  seqname=$prefix$(printf "%02d" $counter)

  # extract frames
#  rm $outdir/*
#  ffmpeg -i $infile -vf fps=10 $outdir/%05d.jpg
#
#  # prepare data for segmentation
#  todir=$tmpdir/$seqname
#  rm $todir -rf
#  mkdir $todir
#  mkdir $todir/images/
#  mkdir $todir/masks/
#  cp $outdir/* $todir/images
#  python scripts/densepose.py $seqname
#
#  # flow
#  . activate viser
#  cd /private/home/gengshany/code/viser/data_gen
#  bash auto_gen.sh $seqname
#  cd -

  cd /private/home/gengshany/code/viser/database/DAVIS/
  rm ~/dropbox/viser/$seqname.zip
  zip ~/dropbox/viser/$seqname.zip -r  */Full-Resolution/$seqname/

  counter=$((counter+1))
  cd /private/home/gengshany/dropbox/GengshanYang_project/test_videos
done
