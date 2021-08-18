import cv2
import pdb
import configparser
import torch
import numpy as np
import imageio
from typing import Any, Dict, List, Tuple, Union
import glob

import dataloader.vidbase as base_data


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

def config_to_dataloader(opts, is_eval=False):
    """
    from a dict of options {seqname, batch_size, ngpu} to a pytorch dataloader
    """
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts['seqname'])
    
    numvid =  len(config.sections())-1
    datalist = []
    for i in range(numvid):
        dataset = get_config_info(opts, config, 'data_%d'%i, i, is_eval=is_eval)
        datalist = datalist + dataset
    dataset = torch.utils.data.ConcatDataset(datalist)
    return dataset

def get_config_info(opts, config, name, dataid, is_eval=False):
    def load_attr(attrs, config, dataname):
        try:attrs['datapath'] = '%s'%(str(config.get(dataname, 'datapath')))
        except:pass
        try:attrs['dframe'] = [int(i) for i in config.get(dataname, 'dframe').split(',')]
        except:pass
        try:attrs['can_frame']= int(config.get(dataname, 'can_frame'))
        except:pass
        try:attrs['init_frame']=int(config.get(dataname, 'init_frame'))
        except:pass
        try:attrs['end_frame'] =int(config.get(dataname, 'end_frame'))
        except:pass
        try:attrs['rtk_path'] =config.get(dataname, 'rtk_path')
        except:pass
        return 
    
    attrs={}
    attrs['rtk_path'] = None

    load_attr(attrs, config, 'data')
    load_attr(attrs, config, name)
    datapath = attrs['datapath']
    dframe =   attrs['dframe']
    can_frame =attrs['can_frame']
    init_frame=attrs['init_frame']
    end_frame= attrs['end_frame']
    rtk_path= attrs['rtk_path']
    numvid =  len(config.sections())-1
    if numvid==1: datapath='%s/%s'%(datapath, opts['seqname'])
    
    imglist = sorted(glob.glob('%s/*'%datapath))
    try: flip=int(config.get(name, 'flip'))
    except: flip=0

    if end_frame >0:
        imglist = imglist[:end_frame]
    print('init:%d, end:%d'%(init_frame, end_frame))
    datasets = []
    for df in dframe:
        dataset = VidDataset(opts, imglist = imglist, can_frame = can_frame, 
                          dframe=df, init_frame=init_frame, 
                          dataid=dataid, numvid=numvid, flip=flip, is_eval=is_eval,
                          rtk_path=rtk_path)
        datasets.append(dataset)
    return datasets
    
class VidDataset(base_data.BaseDataset):
    '''
    '''

    def __init__(self, opts, filter_key=None, imglist=None, can_frame=0,
                    dframe=1,init_frame=0, dataid=0, numvid=1, flip=0, 
                    is_eval=False, rtk_path=None):
        super(VidDataset, self).__init__(opts, filter_key=filter_key)
        
        self.flip=flip
        self.imglist = imglist
        self.can_frame = can_frame
        self.dframe = dframe
        seqname = imglist[0].split('/')[-2]
       
        self.masklist = [i.replace('JPEGImages', 'Annotations').replace('.jpg', '.png') for i in self.imglist] 
        self.camlist =  [i.replace('JPEGImages', 'Camera').replace('.jpg', '.txt') for i in self.imglist]
      
        if dframe==1:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/flo-'%seqname) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/flo-'%seqname) for i in self.imglist]
        else:
            self.flowfwlist = [i.replace('JPEGImages', 'FlowFW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s_%02d/flo-'%(seqname,self.dframe)) for i in self.imglist]
            self.flowbwlist = [i.replace('JPEGImages', 'FlowBW').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s_%02d/flo-'%(seqname,self.dframe)) for i in self.imglist]
        self.featlist = [i.replace('JPEGImages', 'AppFeat').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/feat-'%seqname) for i in self.imglist]
        self.featlist = ['%s/feat-%06d.pfm'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in self.featlist]
        self.kplist = [i.replace('JPEGImages', 'KP').replace('.jpg', '_keypoints.json').replace('.png', '_keypoints.json') for i in self.imglist]
        self.dplist = [i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm') for i in self.imglist]
        if rtk_path is not None:
            self.rtklist =['%s-%05d.txt'%(rtk_path, i) for i in range(len(self.imglist))]
        else:
            self.rtklist =[i.replace('JPEGImages', 'Cameras').replace('.jpg', '.txt') for i in self.imglist]
        
        self.baselist = [i for i in range(len(self.imglist)-self.dframe)] +  [i+self.dframe for i in range(len(self.imglist)-self.dframe)]
        self.directlist = [1] * (len(self.imglist)-self.dframe) +  [0]* (len(self.imglist)-self.dframe)
        
        # to skip frames
        self.odirectlist = self.directlist.copy()
        len_list = len(self.baselist)//2
        self.fw_list = self.baselist[:len_list][init_frame::self.dframe]
        self.bw_list = self.baselist[len_list:][init_frame::self.dframe]
        self.dir_fwlist = self.directlist[:len_list][init_frame::self.dframe]
        self.dir_bwlist = self.directlist[len_list:][init_frame::self.dframe]

        if is_eval:
            self.baselist = self.fw_list
            self.directlist = self.dir_fwlist
        else:
            self.baselist = self.fw_list + self.bw_list
            self.directlist = self.dir_fwlist + self.dir_bwlist
            self.baselist =   [self.baselist[0]]   + self.baselist   + [self.baselist[-1]]
            self.directlist = [self.directlist[0]] + self.directlist + [self.directlist[-1]]

            fac = (opts['batch_size']*opts['ngpu']*200)//len(self.directlist) // numvid
            if fac==0: fac=1
            self.directlist = self.directlist*fac
            self.baselist = self.baselist*fac

        # Load the annotation file.
        self.num_imgs = len(self.directlist)
        self.dataid = dataid
        print('%d pairs of images' % self.num_imgs)
