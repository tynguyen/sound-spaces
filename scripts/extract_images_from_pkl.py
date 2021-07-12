"""
@Brief: extract RGB images from a pickle file dumped from get_RGBD-S_from_a_scene.py
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os, cv2

pkl_file = "data/scene_RGBS/replica/room_0/room_0.pkl"
rgb_root = "data/scene_colmaps/replica/room_0/output_images"
if not os.path.exists(rgb_root):
    os.makedirs(rgb_root, exist_ok=True)

with open(pkl_file, "rb") as fo:
    observations = pkl.load(fo)

# Extract images
for (node, rotation_index), obs in observations.items():
    print(f"[Info] node index {node}, rotation ind {rotation_index}")
    rgb = obs["rgb"]
    rgb_file = os.path.join(rgb_root, f"node_{node}_rotation_{rotation_index}.jpg")
    cv2.imwrite(rgb_file, rgb[:, :, ::-1])
