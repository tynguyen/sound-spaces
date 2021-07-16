"""
@Brief: Read pickle file that contains RGBD-S data generated from a scene
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os, cv2

pkl_file = "data/scene_RGBS/replica/room_0/room_0.pkl"

with open(pkl_file, "rb") as fo:
    observations = pkl.load(fo)

# Extract images
for (node, rotation_index), obs in observations.items():
    print(f"[Info] node index {node}, rotation ind {rotation_index}")
    rgb = obs["rgb"]
    cv_K = obs["cv_K"]
    cvCam2W_T = obs["cvCam2W_T"]
