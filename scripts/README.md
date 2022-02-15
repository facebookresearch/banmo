## More examples  (under construction)

### Example: Motion retargeting
We show an example of retargeting a source (driven) dog model to a driver cat video.

First download the pre-trained dog model.
```
mkdir -p tmp && cd "$_"
wget https://www.dropbox.com/s/y2phlrnh7v3vlx0/shiba-haru-1.pth; cd ../
```

Then run retargeting (optimization) on all the cat videos. This takes 5h on 2 V100 GPUs.
Refer to the main page for downloading the videos and preprocessing.
```
seqname=cat-pikachiu
# To speed up data loading, we store images as lines of pixels). 
# only needs to run it once per sequence and data are stored
python preprocess/img2lines.py --seqname $seqname

# Optimization
bash scripts/template.sh 0,1 $seqname 10001 "no" "no"
bash scripts/template-retarget.sh 0,1 $seqname 10001 "no" "no" tmp/shiba-haru-1.pth
# argv[1]: gpu ids separated by comma 
# args[2]: sequence name
# args[3]: port for distributed training
# args[4]: use_human, pass "" for human cse, "no" for quadreped cse
# args[5]: use_symm, pass "" to force x-symmetric shape
# args[6]: driven model

# Extract articulated meshes and render
bash scripts/render_nvs.sh 0 $seqname logdir/driver-$seqname-e120-b256/params_latest.pth 4 4
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: weights path
# argv[4]: video id used for pose traj
# argv[5]: video id used for root traj
```
</details>

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
