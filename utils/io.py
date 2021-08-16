import cv2
import pdb
import numpy as np
import imageio
from typing import Any, Dict, List, Tuple, Union


def save_vid(outpath, frames, suffix='.gif'):
    """
    save frames to video
    """
    # convert to 150 frames
    frame_150=[]
    for i in range(150):
        fid = int(i/150.*len(frames))
        frame = frames[fid]
        if frame.max()<=1: 
            frame=frame*255
        frame = frame.astype(np.uint8)
        if suffix=='.gif':
            h,w=frame.shape[:2]
            fxy = np.sqrt(1e5/(h*w))
            frame = cv2.resize(frame,None,fx=fxy, fy=fxy)
        frame_150.append(frame)
    imageio.mimsave('%s%s'%(outpath,suffix), frame_150, fps=30)

class visObj(object):
    """
    a class for detectron2 vis
    """
    def has(self, name: str) -> bool:
        return name in self._fields
    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]
