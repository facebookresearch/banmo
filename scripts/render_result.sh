seqname=$1
model_path=$2
testdir=${model_path%/*} # %: from end
add_args=${*: 3:$#-1}
prefix=$testdir/$seqname
#prefix=/scratch/gengshany/Dropbox/output/$seqname

python extract.py --seqname $seqname \
                  --model_path $model_path \
                  $add_args
python render_vis.py --testdir $testdir \
                     --outpath $prefix-vid \
                     --seqname $seqname \
                     --append_img yes \
                     --append_render no
python render_vis.py --testdir $testdir \
                     --outpath $prefix-bne \
                     --seqname $seqname \
                     --vp -1 \
                     --vis_bones
python render_vis.py --testdir $testdir \
                     --outpath $prefix-trj0 \
                     --seqname $seqname \
                     --vp 0 \
                     --vis_traj
python render_vis.py --testdir $testdir \
                     --outpath $prefix-mov0 \
                     --seqname $seqname \
                     --vp 0 
python render_vis.py --testdir $testdir \
                     --outpath $prefix-mov1 \
                     --seqname $seqname \
                     --vp 1
python render_vis.py --testdir $testdir \
                     --outpath $prefix-mov2 \
                     --seqname $seqname \
                     --vp 2
python render_vis.py --testdir $testdir \
                     --outpath $prefix-rst \
                     --seqname $seqname \
                     --rest
python render_vis.py --testdir $testdir \
                     --outpath $prefix-frz \
                     --seqname $seqname \
                     --freeze
python render_vis.py --testdir $testdir \
                     --outpath $prefix-err0 \
                     --seqname $seqname \
                     --vp 0  \
                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
python render_vis.py --testdir $testdir \
                     --outpath $prefix-err1 \
                     --seqname $seqname \
                     --vp 1  \
                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
python render_vis.py --testdir $testdir \
                     --outpath $prefix-err2 \
                     --seqname $seqname \
                     --vp 2  \
                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/
python render_vis.py --testdir $testdir \
                     --outpath $prefix-errs \
                     --seqname $seqname \
                     --vp -1 \
                     --gtdir database/DAVIS/Meshes/Full-Resolution/$seqname/

ffmpeg -y -i $prefix-vid.mp4 \
          -i $prefix-rst.mp4 \
          -i $prefix-frz.mp4 \
          -i $prefix-mov0.mp4 \
          -i $prefix-mov1.mp4 \
          -i $prefix-mov2.mp4 \
          -i $prefix-err0.mp4 \
          -i $prefix-err1.mp4 \
          -i $prefix-err2.mp4 \
-filter_complex "[0:v][1:v][2:v]hstack=inputs=3[top];\
[3:v][4:v][5:v]hstack=inputs=3[mid];\
[6:v][7:v][8:v]hstack=inputs=3[bottom];\
[top][mid][bottom]vstack=inputs=3[v]" \
-map "[v]" \
$prefix-all.mp4

ffmpeg -y -i $prefix-all.mp4 -vf "scale=iw/2:ih/2" $prefix-all.gif
imgcat $prefix*.mp4
imgcat $prefix-all.gif
imgcat $prefix-bne.gif
