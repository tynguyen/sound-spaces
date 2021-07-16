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


def get_single_cam_params(colmap_cameras):
    """
    @Brief: Get camera intrinsics parameters
    """
    # TODO: handle multiple camera cases.
    # For now, assume there is only a single camera
    list_of_keys = list(colmap_cameras.keys())
    cam = colmap_cameras[list_of_keys[0]]
    if cam.model == "PINHOLE":
        h, w, fx, fy = cam.height, cam.width, cam.params[0], cam.params[1]
    elif cam.model == "SIMPLE_PINHOLE":
        h, w, fx, fy = cam.height, cam.width, cam.params[0], cam.params[0]

    # TODO: handle PINHOLE camera model.
    # For now, assume fx = fy
    assert abs(fx - fy) < 1e-4, f"[Error] Assume fx = fy but your input {fx} != {fy}"
    f = fx
    return np.array([h, w, f]).reshape([3, 1])


def get_cvCam2W_transformations(colmap_images):
    """
    @Brief: get a list of transformations from world to OpenCV camera poses
    @Args:
        - colmap_images (dict): map Image ids to MyImage instances
    @Return:
        - c2w_mats (List[np.ndarray(4x4)]): list of transformations from OpenCV cam to the world
    """

    image_names = [colmap_images[k].name for k in colmap_images]
    print(f"[Info] No of images : {len(image_names)}")

    # Retrieve world to Opencv cam's transformations
    transmat_bottom_vector = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    w2c_mats = []
    near_far_distances = []
    for k in colmap_images:
        im = colmap_images[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        w2c_T = np.concatenate([np.concatenate([R, t], 1), transmat_bottom_vector], 0)
        w2c_mats.append(w2c_T)

        # Near, far distances
        near_far_distances.append([float(im.near_distance), float(im.far_distance)])

    # Convert to OpenCV cam to world transformations by inversion
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    return c2w_mats, np.array(near_far_distances)


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
