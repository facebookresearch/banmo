## Pre-process raw videos into BANMo format

We provide instructions and to process raw videos into format ready-to-use for banmo.

<details><summary>[Data format]</summary>

```
DAVIS/
    JPEGImages/
        Full-Resolution/
            sequence-name/
                {%05d}.jpg
    # segmentations from detectron2
    Annotations/
        Full-Resolution/
            sequence-name/
                {%05d}.png
    # forward backward flow between every {1,2,4,8,16,32} frames from VCN-robust
    FlowBW_%d/ and FlowFw_%d/ 
        Full-Resolution/
            sequence-name/ and optionally seqname-name_{%02d}/ (frame interval)
                flo-{%05d}.pfm
                occ-{%05d}.pfm
                visflo-{%05d}.jpg
                warp-{%05d}.jpg
    # 16-dim Densepose features from CSE
    Densepose/
        Full-Resolution/
            sequence-name/
                # 112x(112*16) cropped densepose features
                feat-{%05d}.pfm 
                # [x,y,w,h] saved to warp cropped features to original coordinate
                bbox-{%05d}.txt 
                # densepose surface indices, for visualization
                {%05d}.pfm 
    # lines of pixels in order to speed up dataloading
    Pixels/  
        Full-Resolution/
            sequence-name/
                # skipped frames of flow followed by frame index
                %d-%05d/ 
```
Under each folder, there are visualizations of segmentation, flow and densepose not listed above.
</details>

### Download optical flow model
Download pre-trained VCN optical flow (`pip install gdown` first). Then run
```
mkdir ./lasr_vcn
gdown https://drive.google.com/uc?id=139S6pplPvMTB-_giI6V2dxpOHGqqAdHn -O ./lasr_vcn/vcn_rob.pth
# alternatively: wget https://www.dropbox.com/s/bgsodsnnbxdoza3/vcn_rob.pth -O ./lasr_vcn/vcn_rob.pth
```

### Run segmentation, extract features and flow
Frist, make sure you have ffmpeg installed (`sudo apt-get install ffmpeg`) and 
downloaded cat and human videos under `raw/`.  

Run the following to extract per-frame rgb, mask, flow images 
```
# argv[1]: sequence name. It points to folders under `raw/`.
# argv[2]: format of the video. If the videos end with .mp4 replace .MOV with .mp4
# argv[3]: human or not. y: human, n: quadreped.
# argv[4]: FPS. By default we extract frames at 10 fps
bash preprocess/preprocess.sh cat-pikachiu .MOV n 10 
bash preprocess/preprocess.sh human-cap .MOV y 10 
```

### How to reconstruct your own videos?
To use your own videos, save them under `./raw/$seqname/`. and run the 
command above.

The processing scripts supports `.MOV` and `.mp4` suffixes for now.


<details><summary>[Tips on video capture]</summary>

- Avoid occluding the target by the background, and keep object within the frame. Otherwise, segmentation may fail.

- Avoid zooming. Banmo assumes a constant camera parameter per video.

- Use higher-res videos. The examples uses images of 1920x1080.
</details>

