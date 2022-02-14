# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch

def image_grid(img, row, col):
    """
    img:     N,h,w,x
    collage: 1,.., x
    """
    bs,h,w,c=img.shape
    device = img.device
    collage = torch.zeros(h*row, w*col, c).to(device)
    for i in range(row):
        for j in range(col):
            collage[i*h:(i+1)*h,j*w:(j+1)*w] = img[i*col+j]
    return collage
