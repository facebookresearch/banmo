# vid2shape

## Installation
### Build with conda
```
conda env create -f vid2shape.yml
conda activate vid2shape
# install softras
cd third_party/softras; python setup.py install; cd -;
# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
# marching cubes
pip install --upgrade PyMCubes
# clone detectron2 repo to ../
```

## Preprocess
Extract frames and masks
```
bash scripts/preprocess.sh video-folder
```

Run colmap scripts and save to a zip file. Then
```
bash scripts/colmap_to_data.sh seqname
```

## Run
Create tmp dirs
```
ln -s /private/home/gengshany/data/ database
mkdir tmp
wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl
mv geodists_sheep_5004.pkl ./mesh_material
```

To render objects
```
python scripts/render_synthetic.py --outdir syn-eagle-100 --model eagle --nframes 100 --alpha 1
python scripts/render_synthetic.py --outdir syn-eagled-15h --model eagle --nframes 15 --alpha 0.5 --xspeed 1
python scripts/render_synthetic.py --outdir syn-eagled-15q-vp2 --model eagle --nframes 15 --alpha 0.25 --xspeed 1 --init_a 0.5
```
To optimize
```
bash scripts/template.sh logname dataname address --flowbw
python submit.py scripts/template.sh syn-eagled-15h-lbs-corresp-can15 syn-eagled-15h 10034 --lbs --num_epochs 30 --use_corresp
python submit.py scripts/template.sh ncat-509-correspdp-lbs-ropt-10f nerfies_cat_509 10170 --num_epochs 40 --use_corresp --lbs --flow_dp
python submit.py scripts/template.sh redo-mcat-6-corresp-lbs-flowdp-rot1 mcat_6 10168 --num_epochs 40 --use_corresp --lbs --flow_dp --rot_angle 1
```
To re-render meshes
```
bash scripts/render_result.sh syn-eagled-15h logdir/syn-eagled-15h-test3/params_20.pth --lbs --sample_grid3d 128 --queryfw
bash scripts/render_result.sh mcat_6 logdir/mcat-6-corresp-lbs-flowdp-lr10-rot1-rmdp/params_11.pth --sample_grid3d 128 --queryfw --lbs
```
## Additional Notes
wget https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_sheep_5004.pkl
