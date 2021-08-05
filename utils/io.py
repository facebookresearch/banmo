import cv2
import pdb
import numpy as np
import imageio

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
