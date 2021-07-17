"""
@Brief: convert data extracted from Soundpaces from Colmap format to NeRF format
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
from builtins import breakpoint
import os
import numpy as np
from utils.get_rgbds_options import get_args
from utils.colmap_read_write import read_model
from utils.colmap_read_write import get_single_cam_params
from utils.colmap_read_write import get_cvCam2W_transformations


def gen_pose(basedir, ext=".txt"):
    """
    @Brief: generate NeRF poses.
    This is modified from the original code and this thread: https://github.com/Fyusion/LLFF/issues/10
    """
    colmap_data_root = os.path.join(basedir, "map")
    colmap_cameras, colmap_images = read_model(colmap_data_root, ext)

    # Get camera intrinsics parameters
    # TODO: handle multiple camera cases.
    # For now, assume there is only a single camera
    hwf_params = get_single_cam_params(colmap_cameras)

    # Get OpenCV cam to world transformations
    cvCam2W_mats, near_far_distances = get_cvCam2W_transformations(
        colmap_images
    )  # (Num_poses, 4 x 4)

    # Get poses in NeRF format (3x5 x num_images)
    poses = cvCam2W_mats[:, :3, :4].transpose([1, 2, 0])  # 3 x 4 x num_images
    # Concatenate poses with hwf (camera parameters)
    poses = np.concatenate(
        [poses, np.tile(hwf_params[..., np.newaxis], [1, 1, poses.shape[-1]])], 1
    )
    # Poses now is 3x5 x num_images where the first 4 columns represent the cvCam2W matrix, the
    # last column represents h,w,f

    # Swap columns to represent the transformation from
    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    # The following column swapping will swap from order 0,1,2,3,4 to 1,0,(-2),3,4.
    # This is equivalent to right multiplication of the current rotations with
    # [[0,1,0],[1,0,0],[0,0,-1]] a rotation matrix from the "weird" coord (down, right, backward) that I'm refering as NeRF coord
    # to the OpenCV coord (right, down, toward). Not sure why excluding the translation values in this swapping
    poses = np.concatenate(
        [
            poses[:, 1:2, :],
            poses[:, 0:1, :],
            -poses[:, 2:3, :],
            poses[:, 3:4, :],
            poses[:, 4:5, :],
        ],
        1,
    )

    # Flatten this matrix
    poses = poses.transpose([2, 0, 1])  # num_images x 3 x 5
    poses = poses.reshape([-1, 15])  # num_images x 15

    # Combine the two to get num_images x 15 array
    nerf_poses = np.column_stack([poses, near_far_distances])
    # Save
    np.save(os.path.join(basedir, "poses_bounds.npy"), nerf_poses)
    print(f"[Info] Saved nerf poses to {basedir}/poses_bounds.npy")


if __name__ == "__main__":
    args = get_args("replica")
    scene_obs_dir = os.path.join(args.data_saving_root, args.scene)
    gen_pose(scene_obs_dir, ".txt")
