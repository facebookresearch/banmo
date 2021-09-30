import os
from typing import Any, Dict, List, Tuple, Union
import cv2
import pdb
import configparser
import torch
import numpy as np
import imageio
import trimesh
import glob
import matplotlib.cm

import sys
sys.path.insert(0,'third_party')
import dataloader.vidbase as base_data
from ext_utils.flowlib import flow_to_image

def merge_dict(dict_list):
    out_dict = {}
    for k in dict_list[0].keys():
        out_dict[k] = []

    for i in range(len(dict_list)):
        for k in out_dict.keys():
            out_dict[k] += dict_list[i][k]
    return out_dict

def load_root(root_dir, cap_frame):
    """
    load all the root se(3)
    input is ...-(00000.txt)
    """
    camlist = []
    cam_path = '%s0*.txt'%(root_dir)
    all_path = sorted(glob.glob(cam_path))
    if cap_frame>0:
        all_path = all_path[:cap_frame]
    for idx,path in enumerate(all_path):
        rtk = np.loadtxt(path)
        camlist.append(rtk)
    camlist = np.asarray(camlist)
    return camlist

def draw_cams(all_cam, color='cool', axis=True,
        color_list = None):
    """
    all_cam: a list of 4x4 cameras
    """
    # scale: the scene bound
    cmap = matplotlib.cm.get_cmap(color)
    all_cam = np.asarray(all_cam)
    trans_norm = np.linalg.norm(all_cam[:,:3,3],2,-1)
    valid_cams = trans_norm>0
    trans_max = np.median(trans_norm[valid_cams])
    scale=trans_max
    traj_len = len(all_cam)
    cam_list = [] 
    if color_list is None:
        color_list = np.asarray(range(traj_len))/float(traj_len)
    for j in range(traj_len):
        cam_rot  = all_cam[j][:3,:3].T
        cam_tran = -cam_rot.dot(all_cam[j][:3,3:])[:,0]
    
        radius = 0.02*scale
        cam = trimesh.creation.uv_sphere(radius=radius,count=[2, 2])

        if axis:
            #TODO draw axis
            extents = np.asarray([radius*20, radius*10, radius*0.1])
            axis = trimesh.creation.axis(origin_size = radius, 
                                        origin_color = cmap(color_list[j]),
                                        axis_radius = radius* 0.1,
                                        axis_length = radius*5)
            #extents=extents)
            #axis.vertices[:,2] += radius * 5
            #cam = trimesh.util.concatenate([elips, axis])
            cam = axis

        #cam.vertices = cam.vertices + cam_tran
        cam.vertices = cam.vertices.dot(cam_rot.T) + cam_tran
        #cam.visual.vertex_colors = cmap(float(j)/traj_len)
        cam_list.append(cam)
    mesh_cam = trimesh.util.concatenate(cam_list)
    return mesh_cam


def save_vid(outpath, frames, suffix='.gif',upsample_frame=150., fps=30,
        is_flow=False):
    """
    save frames to video
    frames:     n,h,w,1 or n.
    """
    # convert to 150 frames
    if upsample_frame<1: upsample_frame = len(frames)
    frame_150=[]
    for i in range(int(upsample_frame)):
        fid = int(i/upsample_frame*len(frames))
        frame = frames[fid]
        if is_flow:
            frame = flow_to_image(frame)
        if frame.max()<=1: 
            frame=frame*255
        frame = frame.astype(np.uint8)
        if suffix=='.gif':
            h,w=frame.shape[:2]
            fxy = np.sqrt(4e5/(h*w))
            frame = cv2.resize(frame,None,fx=fxy, fy=fxy)
        frame_150.append(frame)
    imageio.mimsave('%s%s'%(outpath,suffix), frame_150, fps=fps)

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
    rtk_path=opts['rtk_path']
    numvid =  len(config.sections())-1
    if numvid==1 and not config.has_option(name, 'datapath'): 
        datapath='%s/%s'%(datapath, opts['seqname'])
    
    if rtk_path =='':
        rtk_path= attrs['rtk_path']
    elif not os.path.isfile('%s-00000.txt'%rtk_path):
        print('loading cameras from init-cam')
        rtk_path = '%s/%s'%(rtk_path, datapath.strip('/').split('/')[-1])
    
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
        self.featlist = [i.replace('JPEGImages', 'Densepose').replace('.jpg', '.pfm').replace('.png', '.pfm').replace('%s/'%seqname, '%s/feat-'%seqname) for i in self.imglist]
        self.featlist = ['%s/feat-%05d.pfm'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in self.featlist]
        self.bboxlist = ['%s/bbox-%05d.txt'%(i.rsplit('/',1)[0], int(i.split('feat-')[-1].split('.pfm')[0])) for i in self.featlist]
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

def str_to_frame(test_frames, data_info):
    if test_frames[0]=='{':
        # render a list of videos
        idx_render = []
        for i in test_frames[1:-1].split(','):
            vid_idx = int(i)
            idx_render += range(data_info['offset'][vid_idx]-vid_idx, 
                                data_info['offset'][vid_idx+1]-vid_idx-1)
    else:
        test_frames = int(test_frames)
        if test_frames==0: 
            test_frames = data_info['len_evalloader']-1
        # render specific number of frames
        idx_render = np.linspace(0,data_info['len_evalloader']-1,
                               test_frames, dtype=int)
    return idx_render

def extract_data_info(loader):
    data_info = {}
    dataset_list = loader.dataset.datasets
    data_offset = [0]
    impath = []
    for dataset in dataset_list:
        impath += dataset.imglist
        data_offset.append(len(dataset.imglist))
    data_info['offset'] = np.asarray(data_offset).cumsum()
    data_info['impath'] = impath
    data_info['len_evalloader'] = len(loader)
    return data_info
