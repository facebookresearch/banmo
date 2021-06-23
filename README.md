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
# install manifold remeshing
git clone --recursive -j8 git://github.com/hjwdzh/Manifold; cd Manifold; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release;make; cd ../../
```

## Run
To render objects
```
python scripts/render_synthetic.py --outdir syn-eagle-100 --model eagle --nframes 100 --alpha 1
```
To optimize
```
python -m torch.distributed.launch --master_port 19957 --nproc_per_node=1 main.py --batch_size 3 --img_size 128
```
To re-render meshes
```
python extract.py --img_size 256 --seqname syn-eagle-100
```
## Additional Notes
