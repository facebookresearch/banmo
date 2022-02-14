## More examples  (under construction)

## Example: Motion retargeting

### Example: AMA-human
Download swing and samba sequences from [aminated mesh animation website](https://people.csail.mit.edu/drdaniel/mesh_animation/) or 
run the following scripts
```
cd database; wget $(cat ../misc/ama.txt);
# untar files
ls *.tar | xargs -i tar xf {}
find ./T_* -type f -name "*.tgz" -execdir tar -xvzf {} \;
cd ../
```
and convert into our DAVIS format.
```
python scripts/ama-process/ama2davis.py --path ./database/T_samba
python scripts/ama-process/ama2davis.py --path ./database/T_swing
```
Then extract flow and dense appearance features (take ~1h)
```
mkdir raw/ama-female;
# write filenames in replace of .MOV files
ls -d database/DAVIS/Annotations/Full-Resolution/T_s* | xargs -i echo {} | sed 's:.*/::' | xargs -i touch raw/ama-female/{}.txt # create empty txt files
bash preprocess/preprocess.sh ama-female .txt y 10
python preprocess/img2lines.py --seqname ama-female
```
To optimize, run 
```
bash scripts/template.sh 0,1 ama-female 10001 "" "no"
```

### Evaluation on AMA
Install chamfer3D
```
cd third_party/chamfer3D/; python setup.py install; cd ../../
```
To evaluate AMA-swing
```
seqname=T_swing1
testdir=~/data/old_checkpoints_4/ama-female-lbs-rkopt-300-b16-ft2/
gtdir=~/data/AMA/T_swing/meshes/
gt_pmat=/private/home/gengshany/data/AMA/T_swing/calibration/Camera1.Pmat.cal
python render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval \
    --seqname $seqname --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat $gt_pmat
```


### Synthetic data
First, install softras
```
cd third_party/softras; python setup.py install; cd -;
```
Then download animated mesh sequences in XXX, and install mesh renderer.
```
cd third_party/softras
python setup.py install
```
Then render image data and prepare mesh ground-truth
```
bash scripts/render_eagle.sh
``` 

To run optimization
```
bash scripts/template-cam.sh 0,1,2,3,4,5,6,7 a-eagle 120
```


To evaluate eagle
```
seqname=a-eagle-1
testdir=/private/home/gengshany/data/old_checkpoints_4/a-eagle-lbs-rkopt-b16-cam-cse
gtdir=database/DAVIS/Meshes/Full-Resolution/a-eagle-1/
gt_pmat="''"
python render_vis.py --testdir $testdir  --outpath $testdir/$seqname-eval \
    --seqname $seqname --test_frames "{0}" --vp 0  --gtdir $gtdir --gt_pmat $gt_pmat
```
