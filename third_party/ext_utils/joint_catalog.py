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

###################################################################################################################
# Annotated joints 0 -> 32 relate to positions on the SMAL model (Zuffi et al 2018) defined in `smal_CVPR2018.pkl'.
# Joints 33, 34, 35 and 36 have been added and relate to the following SMAL vertices:
#
# ------------------------------------------------
# JOINT_ID | POSITION  | SMAL_VERTEX_ID
# ------------------------------------------------
#  0 -> 32 | JOINTS IN THE SMAL MODEL
# ------------------------------------------------
#    33    |  NOSE_TIP | 1863
#    34    |    CHIN   | 26
#    35    |  LEFT EAR | 149
#    36    | RIGHT EAR | 2124
# ------------------------------------------------
#
# SMALJointCatalog comprises all joints predicted by the hourglass network in the paper `Creatures great and SMAL'.
# The joints commented out in the SMALJointCatalog have no annotations
###################################################################################################################

from enum import Enum
import numpy as np
import cv2

class SMALJointCatalog(Enum):
    # body_0 = 0
    # body_1 = 1
    # body_2 = 2
    # body_3 = 3
    # body_4 = 4
    # body_5 = 5
    # body_6 = 6
    # upper_right_0 = 7
    upper_right_1 = 8
    upper_right_2 = 9
    upper_right_3 = 10
    # upper_left_0 = 11
    upper_left_1 = 12
    upper_left_2 = 13
    upper_left_3 = 14
    neck_lower = 15
    # neck_upper = 16
    # lower_right_0 = 17
    lower_right_1 = 18
    lower_right_2 = 19
    lower_right_3 = 20
    # lower_left_0 = 21
    lower_left_1 = 22
    lower_left_2 = 23
    lower_left_3 = 24
    tail_0 = 25
    # tail_1 = 26
    # tail_2 = 27
    tail_3 = 28
    # tail_4 = 29
    # tail_5 = 30
    tail_6 = 31
    jaw = 32
    nose = 33 # ADDED JOINT FOR VERTEX 1863
    # chin = 34 # ADDED JOINT FOR VERTEX 26
    right_ear = 35 # ADDED JOINT FOR VERTEX 149
    left_ear = 36 # ADDED JOINT FOR VERTEX 2124

class SMALJointInfo():
    def __init__(self):
        # These are the 
        self.annotated_classes = np.array([
            8, 9, 10, # upper_right
            12, 13, 14, # upper_left
            15, # neck
            18, 19, 20, # lower_right
            22, 23, 24, # lower_left
            25, 28, 31, # tail
            32, 33, # head
            35, # right_ear
            36]) # left_ear

        self.annotated_markers = np.array([
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS])

        self.joint_regions = np.array([ 
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 
            2, 2, 2, 2, 
            3, 3, 
            4, 4, 4, 4, 
            5, 5, 5, 5, 
            6, 6, 6, 6, 6, 6, 6,
            7, 7, 7,
            8, 
            9])

        self.annotated_joint_region = self.joint_regions[self.annotated_classes]
        self.region_colors = np.array([
            [250, 190, 190], # body, light pink
            [60, 180, 75], # upper_right, green
            [230, 25, 75], # upper_left, red
            [128, 0, 0], # neck, maroon
            [0, 130, 200], # lower_right, blue
            [255, 255, 25], # lower_left, yellow
            [240, 50, 230], # tail, majenta
            [245, 130, 48], # jaw / nose / chin, orange
            [29, 98, 115], # right_ear, turquoise
            [255, 153, 204]]) # left_ear, pink
        
        self.joint_colors = np.array(self.region_colors)[self.annotated_joint_region]
