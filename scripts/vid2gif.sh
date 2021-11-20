prefix=cat_5
filedir=$prefix
outdir=tmp
suffix=.MOV

counter=0
for infile in $filedir/*$suffix; do
  ffmpeg -y -i $infile -vf "scale=iw/8:ih/8" $infile.gif

  counter=$((counter+1))
done
