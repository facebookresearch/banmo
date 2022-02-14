# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import numpy as np
import cv2
import pdb

pmat = np.loadtxt('/private/home/gengshany/data/AMA/T_swing/calibration/Camera1.Pmat.cal')
K,R,T,_,_,_,_=cv2.decomposeProjectionMatrix(pmat)
print(K/K[-1,-1])
print(R)
print(T/T[-1])
pdb.set_trace()
