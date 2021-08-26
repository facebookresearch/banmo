# GNU GPL header
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 2 as published by the Free Software Foundation.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

###########################################################################################################################
# Benjamin Biggs | bjb56@cam.ac.uk | http://mi.eng.cam.ac.uk/~bjb56/                                                      #
# Please cite `Creatures Great and SMAL: Recovering the shape and motion of animals from video' if you use this dataset   #
###########################################################################################################################
import pdb
import numpy as np
import imageio
#import scipy.misc
from random import shuffle
import json
import os
import cv2
import sys

from .joint_catalog import SMALJointInfo

BADJA_PATH = './database'

class BADJAData():
    def __init__(self, seqname):
        annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
        IGNORE_ANIMALS = [
             "bear.json",
             "camel.json",
            "cat_jump.json",
             "cows.json",
             "dog.json",
             "dog-agility.json",
             "horsejump-high.json",
             "horsejump-low.json",
             "impala0.json",
             "rs_dog.json",
            "tiger.json"
            ]
        IGNORE_ANIMALS = [i for i in IGNORE_ANIMALS if '%s.json'%seqname != i]

        self.animal_dict = {}
        self.smal_joint_info = SMALJointInfo()
        for animal_id, animal_json in enumerate(sorted(os.listdir(annotations_path))):
            if animal_json not in IGNORE_ANIMALS:
                json_path = os.path.join(annotations_path, animal_json)
                with open(json_path) as json_data:
                    animal_joint_data = json.load(json_data)
                filenames = []
                segnames = []
                joints = []
                visible = []

                for image_annotation in animal_joint_data:
                    file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
                    seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

                    if os.path.exists(file_name) and os.path.exists(seg_name):
                        filenames.append(file_name)
                        segnames.append(seg_name)
                        joints.append(np.array(image_annotation['joints']))
                        visible.append(np.array(image_annotation['visibility']))
                    elif os.path.exists(file_name):
                        print ("BADJA SEGMENTATION file path: {0} is missing".format(seg_name))
                    else:
                        print ("BADJA IMAGE file path: {0} is missing".format(file_name))
            
                self.animal_dict[animal_id] = (filenames, segnames, joints, visible)

        print ("Loaded BADJA dataset")

    def get_loader(self):
        #self.animal_dict.pop(10)
        #self.animal_dict.pop(2)
        #for idx in range(int(1e6)):
        #pdb.set_trace()
        for idx in self.animal_dict.keys():
            animal_id = idx
            #animal_id = np.random.choice(self.animal_dict.keys())
            filenames, segnames, joints, visible = self.animal_dict[animal_id]

            for image_id in range(len(filenames)):
                seg_file = segnames[image_id]
                image_file = filenames[image_id]
                
                jointstmp = joints[image_id].copy()
                jointstmp = jointstmp[self.smal_joint_info.annotated_classes]
                visibletmp = visible[image_id][self.smal_joint_info.annotated_classes]

                rgb_img = cv2.imread(image_file)[:,:,::-1]
                sil_img = imageio.imread(seg_file)

                rgb_h, rgb_w, _ = rgb_img.shape
                sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

                yield rgb_img, sil_img, jointstmp, visibletmp, image_file
