# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# save preprocessed files to zips
seqname=$1
#seqname=cat-pikachiu00

cd database/DAVIS/
zip $seqname.zip -r Annotations/Full-Resolution/$seqname/ \
                            Densepose/Full-Resolution/$seqname/ \
                                Flow*/Full-Resolution/$seqname/ \
                           JPEGImages/Full-Resolution/$seqname/ 
