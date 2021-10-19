rootdir=$1
tmpdir=tmp/
prefix=$2
filedir=$rootdir/$prefix
outdir=$rootdir/output
suffix=$3
fps=10

if [ "$suffix" = ".MOV" ]; then
  # rename to upper case
  cd $filedir
  for file in ./*; do mv -- "$file" "${file^^}"; done
  cd -
fi

counter=0
for infile in $filedir/*$suffix; do
  # filter videos
#  if [[ "$infile" == *"cat_5/IMG_8005.MOV"* ]]; then
##  if [[ "$infile" != *"cat_5/IMG_8001.MOV"* ]]; then
#    counter=$((counter+1))
#    continue
#  fi

  echo $infile  

  if [ "$suffix" = ".MOV" ]; then
    seqname=$prefix$(printf "%02d" $counter)
    ## process videos
    # extract frames
    rm $outdir/*
    ffmpeg -i $infile -vf fps=$fps $outdir/%05d.jpg

    # segmentation
    todir=$tmpdir/$seqname
    rm $todir -rf
    mkdir $todir
    mkdir $todir/images/
    mkdir $todir/masks/
    cp $outdir/* $todir/images
    python scripts/densepose.py $seqname
  else
    seqname=$(basename "$infile")
    seqname=${seqname::-4}
    # process segmented data
    unzip -o $infile -d database/DAVIS/
  fi

  python scripts/compute_dp.py $seqname

  # flow
  cd third_party/vcnplus
  bash compute_flow.sh $seqname
  cd -

  bash scripts/colmap_to_data.sh $seqname

  ## save to zips
  #cd database/DAVIS/
  #rm  $rootdir/$seqname.zip
  #zip $rootdir/$seqname.zip -r  */Full-Resolution/$seqname/
  #cd -

  counter=$((counter+1))
done
