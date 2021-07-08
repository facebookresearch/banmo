seqname=$1
model_path=$2
add_args=${*: 3:$#-1}
prefix=/scratch/gengshany/Dropbox/output/$seqname

testdir=${model_path%/*} # %: from end
python extract.py --seqname $seqname \
                  --model_path $model_path \
                  $add_args
python render_vis.py --testdir $testdir \
                     --outpath $prefix-vid \
                     --seqname $seqname \
                     --append_img yes \
                     --append_render no
python render_vis.py --testdir $testdir \
                     --outpath $prefix-mov \
                     --seqname $seqname \
                     --freeze no
python render_vis.py --testdir $testdir \
                     --outpath $prefix-sta \
                     --seqname $seqname \
                     --freeze yes
