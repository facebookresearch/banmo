#
# bash scripts/preprocess.sh ./raw/ Sultan .MOV no 10 
#                            folder, folder, file ext, human or not, fps
# file ext can be {.MOV, .mp4}

rootdir=$1
tmpdir=tmp/
prefix=$2
filedir=$rootdir/$prefix
maskoutdir=$rootdir/output
finaloutdir=database/DAVIS/
suffix=$3
ishuman=$4 # y/n
fps=$5
#fps=10

## rename to upper case
#if [ "$suffix" = ".MOV" ]; then
#  cd $filedir
#  for file in ./*; do mv -- "$file" "${file^^}"; done
#  cd -
#fi

counter=0
for infile in $filedir/*$suffix; do
  # filter videos
#  if [[ "$infile" == *"cat_5/IMG_8005.MOV"* ]]; then
##  if [[ "$infile" != *"cat_5/IMG_8001.MOV"* ]]; then
#    counter=$((counter+1))
#    continue
#  fi

  echo $infile  
  if [ "$suffix" = ".MOV" ] || [ "$suffix" = ".mp4" ]; then
    seqname=$prefix$(printf "%02d" $counter)
    ## process videos
    # extract frames
    rm -i $maskoutdir/*
    ffmpeg -i $infile -vf fps=$fps $maskoutdir/%05d.jpg

    # segmentation
    todir=$tmpdir/$seqname
    rm -i $todir -rf
    mkdir $todir
    mkdir $todir/images/
    mkdir $todir/masks/
    cp $maskoutdir/* $todir/images
    rm -i $finaloutdir/JPEGImages/Full-Resolution/$seqname
    rm -i $finaloutdir/Annotations/Full-Resolution/$seqname
    mkdir $finaloutdir/JPEGImages/Full-Resolution/$seqname
    mkdir $finaloutdir/Annotations/Full-Resolution/$seqname
    python scripts/mask.py $seqname $ishuman
  elif [ "$suffix" = ".zip" ]; then
    seqname=$(basename "$infile")
    seqname=${seqname::-4}
    # process segmented data
    unzip -o $infile -d $finaloutdir
  else
    echo 'directly processing sil/rgb'
    seqname=`echo $infile | sed -e 's/\/.*\///g'`
    seqname=${seqname::-4}
    echo $seqname
  fi

  rm -i $finaloutdir/DensePose/Full-Resolution/$seqname
  mkdir $finaloutdir/DensePose/Full-Resolution/$seqname
  python scripts/compute_dp.py $seqname $ishuman

  # flow
  cd third_party/vcnplus
  bash compute_flow.sh $seqname
  cd -

#  bash scripts/colmap_to_data.sh $seqname

  ## save to zips
  #cd database/DAVIS/
  #rm -i  $rootdir/$seqname.zip
  #zip $rootdir/$seqname.zip -r  */Full-Resolution/$seqname/
  #cd -

  counter=$((counter+1))
done
