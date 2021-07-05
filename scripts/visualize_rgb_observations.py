"""
@Brief: visualze RGB images from data/scene_observations"
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt

pkl_file = "data/scene_observations/replica/room_0.pkl"
with open(pkl_file, "rb") as f:
    data = pickle.load(f)

i = 0
for pose, val in data.items():
    i += 1
    if i > 20:
        break
    print(f"[Info] pose {pose}")
    rgb_img = val["rgb"]
    dmap = val["depth"]

    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.subplot(122)
    plt.imshow(dmap, cmap="jet")
    plt.show()
