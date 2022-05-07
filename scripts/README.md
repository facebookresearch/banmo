## More examples

### Example: Motion retargeting
We show an example of retargeting a source (driven) dog model to a driver cat video.

First download the pre-trained dog model.
```
mkdir -p tmp && cd "$_"
wget https://www.dropbox.com/s/wqm7ulxx9qpu85g/shiba-haru-1.npy
wget https://www.dropbox.com/s/e21vycgoe5y7pio/shiba-haru-1.pth
cd ../
```

Then run retargeting (optimization) on a cat video. This takes 2h on 2 TitanXp GPUs.
Refer to the main page for downloading the videos and preprocessing.
```
seqname=cat-pikachiu-sub
# To speed up data loading, we store images as lines of pixels). 
# only needs to run it once per sequence and data are stored
python preprocess/img2lines.py --seqname $seqname

# Optimization
bash scripts/template-retarget.sh 0,1 $seqname 10001 "no" "no" tmp/shiba-haru-1.pth
# argv[1]: gpu ids separated by comma 
# args[2]: sequence name
# args[3]: port for distributed training
# args[4]: use_human, pass "" for human cse, "no" for quadreped cse
# args[5]: use_symm, pass "" to force x-symmetric shape
# args[6]: driven model

# Extract articulated meshes and render
bash scripts/render_nvs.sh 0 $seqname logdir/driver-$seqname-e120-b256/params_latest.pth 0 0
# argv[1]: gpu id
# argv[2]: sequence name
# argv[3]: weights path
# argv[4]: video id used for pose traj
# argv[5]: video id used for root traj
```
Results will be saved at `logdir/driver-$seqname-e120-b256/nvs*.mp4`.
</details>

### Example: Adaptation to a new video

Instead of retargeting to a new video, we can adapt a pre-optimized model 
to a new video, which usually converges faster and more stable than starting from scratch.
We show an example of adapting cat-coco to a single video of cat-socks.

First download the pre-trained cat-coco model.
```
mkdir -p tmp && cd "$_"
wget https://www.dropbox.com/s/tvc9rpef9fgb7vp/cat-coco.npy
wget https://www.dropbox.com/s/iwosbezgfwcr54w/cat-coco.pth
cd ../
```

Then download and run optimization on a video of cat-socks 
```
seqname=cat-socks
bash misc/processed/download.sh $seqname
python preprocess/img2lines.py --seqname $seqname
bash scripts/template-prior-model.sh 0,1 $seqname 10001 "no" "no" tmp/cat-coco.pth
bash scripts/render_mgpu.sh 0 $seqname logdir/prior-$seqname-e30-b256-ft2/params_latest.pth \
        "0" 256
```


### Example: Known root poses

There are two additional steps to use pre-computed camera poses wrt the object (root body poses) instead of the posenet.
- First, add `rtk_path` key to config files, see `configs/cat-pikachiu-cam.config` for an example.
- Then use `template-known-cam.sh` instead of `template.sh` when running optimization. Specifically, it replaces `--pose_cnn_path $pose_cnn_path` with `--use_rtk_file`.

Here's an example. Frist, download the precomputed rtk files.
```
mkdir -p cam-files/cse-cat-pikachiu && cd "$_"
wget https://www.dropbox.com/sh/4zw6mkhn1bbjk2s/AAAHOPz0NLInAQBSBHqFdA9ha -O tmp.zip
unzip tmp.zip; cd ../../
```
The rtk files are in the format of 
```
# xxx-%05d.txt/
# [R_3x3|T_3x1] # translation is in the same scale as unit-sized object
# [fx,fy,px,py] # in pixel
```
Then run optimization (assuming you've already run img2line.py)
```
seqname=cat-pikachiu-cam
bash scripts/template-known-cam.sh 0,1 $seqname 10001 "no" "no"
```



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
seqname=ama-female
mkdir raw/$seqname;
# write filenames in replace of .MOV files
ls -d database/DAVIS/Annotations/Full-Resolution/T_s* | xargs -i echo {} | sed 's:.*/::' | xargs -i touch raw/$seqname/{}.txt # create empty txt files
bash preprocess/preprocess.sh $seqname .txt y 10
```
To optimize, run 
```
# store as lines
python preprocess/img2lines.py --seqname $seqname 
# optimization
bash scripts/template.sh 0,1 $seqname 10001 "" "no"
# extract articulated meshes for two representative videos
bash scripts/render_mgpu.sh 0 $seqname logdir/$seqname-e120-b256-ft3/params_latest.pth \
        "0 8" 256
```

### Evaluation on AMA
Install chamfer3D
```
pip install -e third_party/chamfer3D/
```

Then download example data
```
mkdir -p tmp/banmo-swing1 && cd "$_"
wget https://www.dropbox.com/sh/n9eebife5uovg2m/AAA1BsADDzCIsTSUnJyCTRp7a -O tmp.zip
unzip tmp.zip; rm tmp.zip; cd ../../
```

To evaluate AMA-swing
```
bash scripts/eval/run_eval.sh 0 tmp/banmo-swing1/
# argv[1] gpu id
# argv[2] test dir
```
results will be saved at `tmp/banmo-swing1.txt` and video visualizations will be saved at `tmp/banmo-swing1/ama-female-all.mp4`

### Synthetic data
First install soft rasterizer
```
pip install -e third_party/softras
```

Then download animated mesh sequences
```
mkdir database/eagle && cd "$_"
wget https://www.dropbox.com/sh/xz8kckfq817ggqd/AADIhtb1syWhDQeY8xa9Brc0a -O eagle.zip
unzip eagle.zip; cd ../../
mkdir database/hands && cd "$_"
wget https://www.dropbox.com/sh/kbobvwtt51bl165/AAC4d-tbJ5PR6XQIUjbk3Qe2a -O hands.zip
unzip hands.zip; cd ../../
```

Render image data and prepare mesh ground-truth
```
bash scripts/synthetic/render_eagle.sh
bash scripts/synthetic/render_hands.sh
``` 

To run optimization on eagle
```
seqname=a-eagle
# store as lines
python preprocess/img2lines.py --seqname $seqname 
# optimization
bash scripts/template-known-cam.sh 0,1 $seqname 10001 "no" "no"
# extract articulated meshes
bash scripts/render_mgpu.sh 0 $seqname logdir/known-cam-$seqname-e120-b256/params_latest.pth \
        "0" 256
```
To run optimization on hands, turn connected components off (`--nouse_cc`). See note on the main page.

To evaluate eagle, modify the related lines in `scripts/eval/run_eval.sh` and run
```
bash scripts/eval/run_eval.sh 0 logdir/known-cam$seqname-e120-b256/
```
