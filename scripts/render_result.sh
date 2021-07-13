seqname=$1
model_path=$2
add_args=${*: 3:$#-1}
prefix=/scratch/gengshany/Dropbox/output/$seqname

testdir=${model_path%/*} # %: from end
python extract.py --seqname $seqname \
                  --model_path $model_path \
                  $add_args
#python render_vis.py --testdir $testdir \
#                     --outpath $prefix-vid \
#                     --seqname $seqname \
#                     --append_img yes \
#                     --append_render no
python render_vis.py --testdir $testdir \
                     --outpath $prefix-mov0 \
                     --seqname $seqname \
                     --freeze no \
                     --vp 0 \
                     --vis_bones
#python render_vis.py --testdir $testdir \
#                     --outpath $prefix-mov1 \
#                     --seqname $seqname \
#                     --freeze no \
#                     --vp 1
#python render_vis.py --testdir $testdir \
#                     --outpath $prefix-mov2 \
#                     --seqname $seqname \
#                     --freeze no \
#                     --vp 2
#python render_vis.py --testdir $testdir \
#                     --outpath $prefix-sta \
#                     --seqname $seqname \
#                     --freeze yes
#
#ffmpeg -i $prefix-vid.mp4 \
#       -i $prefix-sta.mp4 \
#       -i $prefix-sta.mp4 \
#       -i $prefix-mov0.mp4 \
#       -i $prefix-mov1.mp4 \
#       -i $prefix-mov2.mp4 \
#-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[top];\
#[3:v][4:v][5:v]hstack=inputs=3[bottom];\
#[top][bottom]vstack=inputs=2[v]" \
#-map "[v]" \
#$prefix-all.mp4
