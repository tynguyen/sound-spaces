"""
@Brief: test the data extracted from Soundpaces in Colmap format
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
import os, time
from utils.transformations import rotmat2qvec
import numpy as np
import open3d as o3d
from utils.get_rgbds_options import get_args
from utils.colmap_read_write import read_model
from utils.colmap_read_write import get_single_cam_params
from utils.colmap_read_write import get_cvCam2W_transformations
from utils.colmap_read_write import read_rgbd_images
from utils.colmap_read_write import getPCLfromRGB_D
import matplotlib.pyplot as plt


def test_colmap_data(basedir, colmap_ext=".txt", rgb_ext=".jpg", depth_ext=".png"):
    """
    @Brief: test colmap data extraction from Soundspaces
    """
    print(f"-----------------------------------")
    print(f"[Info] Dataroot repo: {basedir}")
    colmap_data_root = os.path.join(basedir, "map")
    colmap_cameras, colmap_images = read_model(colmap_data_root, colmap_ext)

    # Get camera intrinsics parameters
    # TODO: handle multiple camera cases.
    # For now, assume there is only a single camera
    hwf_params = get_single_cam_params(colmap_cameras).squeeze()

    # Get OpenCV cam to world transformations
    cvCam2W_mats, near_far_distances = get_cvCam2W_transformations(
        colmap_images
    )  # (Num_poses, 4 x 4)

    # Get RGBD images
    image_names = [colmap_images[k].name for k in colmap_images]
    rgbd_images = read_rgbd_images(
        basedir, image_names, rgb_ext=rgb_ext, depth_ext=depth_ext
    )

    # Get pcl from RGB, depth images:
    K = np.array(
        [
            [hwf_params[-1], 0, hwf_params[1] / 2],
            [0, hwf_params[-1], hwf_params[0] / 2],
            [0, 0, 1],
        ]
    )
    # Visualize the pcls
    visualizer = get_o3d_visualizer()

    pcl_list = []
    for i, image_name in enumerate(image_names):
        rgb, depth = rgbd_images[image_name]
        # Depth is given in mm. Convert it to m
        # plt.subplot(121)
        # plt.imshow(depth, cmap="jet")
        depth = depth / 1000.0
        depth = depth.astype(np.float32)
        # plt.subplot(122)
        # plt.imshow(depth, cmap="jet")
        # plt.show()
        cvCam2W_T = cvCam2W_mats[i]
        o3d_rgbd_pcl = getPCLfromRGB_D(rgb, depth, K)
        # Transform this pcl
        o3d_rgbd_pcl.transform(cvCam2W_T)
        # DEBUG rotation vector
        qvec = rotmat2qvec(cvCam2W_T[:3, :3])
        print(f"[Info] Image: {image_name}")
        print(f"[Info] qvec{qvec}")
        print(f"[Info] tran vec{cvCam2W_T[:3, -1]}")
        pcl_list.append(o3d_rgbd_pcl)
        # Camera pose
        o3d_cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        o3d_cam_pose.transform(cvCam2W_T)
        pcl_list.append(o3d_cam_pose)
        # if i > 16:
        #     break

    o3d.visualization.draw_geometries(pcl_list)
    visualizer.destroy_window()


def get_o3d_visualizer():
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    return visualizer


if __name__ == "__main__":
    args = get_args("replica")
    scene_obs_dir = os.path.join(args.data_saving_root, args.scene)
    test_colmap_data(scene_obs_dir, ".txt")
