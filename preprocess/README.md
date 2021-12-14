In the root dir of banmo, create tmp dir and data dir
```
mkdir ./tmp; mkdir -p database/DAVIS/
```
Download input cat videos to `./raw`
```
mkdir -p raw/Sultan; mkdir raw/output
curl -L https://www.dropbox.com/sh/qbl88gmukegusy9/AAAi6JI-NTUtJGa_sNZ7IETza?dl=1 > ./raw/Sultan.zip
unzip "./raw/Sultan.zip" -d ./raw/Sultan/
```
Download pre-trained VCN optical flow:
```
mkdir ./lasr_vcn
gdown https://drive.google.com/uc?id=139S6pplPvMTB-_giI6V2dxpOHGqqAdHn -O ./lasr_vcn/vcn_rob.pth
```
Extract per-frame rgb, mask, flow images. 
Optionally, you can extract initial SfM camera poses by uncommenting some lines in `preprocess.sh`.
To do so, we reused [nerfies](https://github.com/google/nerfies) colmap processing code.
Please follow the instructions to install nerfies conda env.
```
bash preprocess/preprocess.sh Sultan .MOV n 10 
```
To speed up data loading, we store images as lines
```
python preprocess/img2lines.py --seqname sultan10
```
