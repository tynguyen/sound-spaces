"""
@Brief: A wrapper to log colmap data to files. This is modified from read_write.py in colmap/python
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""

import os, cv2, glob
import collections
from numpy.linalg.linalg import _fastCopyAndTranspose
from skimage.filters.edges import farid

import soundfile
from soundspaces.mp3d_utils import Object
import numpy as np
import struct
import argparse
import pickle
from shutil import copyfile
from utils.transformations import qvec2rotmat
import open3d as o3d
import json

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"],)
MyBaseImage = collections.namedtuple(
    "MyImage",
    [
        "id",
        "qvec",
        "tvec",
        "camera_id",
        "name",
        "xys",
        "point3D_ids",
        "near_distance",
        "far_distance",
    ],
)


class ColmapDataWriter(Object):
    def __init__(self, data_root, audio_sample_rate=44100, ext=".txt"):
        """
        @Brief: a wrapper for Colmap data writer.
                Data will be written to images{ext}, cameras{ext}

        @Args:
            - data_root (str): absolute path to the repo we're storing the data
            - ext (str): file extension i.e: ".txt" or ".bin"
        """
        print(f"[Info] ColmapDataWriter")
        self.cameras = {}
        # Colmap Image instances
        self.images = {}

        # Rgb images
        self.rgb_images = {}
        self.rgb_root = os.path.join(data_root, "images")
        # Depth images
        self.depth_images = {}
        self.depth_root = os.path.join(data_root, "depths")
        # Audio responses
        self.audios = {}
        self.audio_root = os.path.join(data_root, "audios")
        self.audio_sample_rate = audio_sample_rate
        # Room impulse responses
        self.rirs = {}
        self.rir_root = os.path.join(data_root, "rirs")

        # Root to colmap data
        self.colmap_root = os.path.join(data_root, "map")
        self.ext = ext

        # Root to the relative transformation between each camera view and the anchor
        self.transform2anchor_root = os.path.join(data_root, "transforms2anchor")
        self.transforms2anchor = {}

        # Create repos
        os.makedirs(self.rgb_root, exist_ok=True)
        os.makedirs(self.depth_root, exist_ok=True)
        os.makedirs(self.audio_root, exist_ok=True)
        os.makedirs(self.rir_root, exist_ok=True)
        os.makedirs(self.colmap_root, exist_ok=True)
        os.makedirs(self.transform2anchor_root, exist_ok=True)

    def _form_a_camera(self, model, camera_id, width, height, params):
        """
        @Brief: return a Camera instance with the format
            Camera(
                id=camera_id, model=model, width=width, height=height, params=params
            )
        """
        return Camera(
            id=camera_id, model=model, width=width, height=height, params=params
        )

    def add_camera(self, model, camera_id, width, height, params):
        """
        @Brief: add a Camera to self.cameras given its set of values
        """
        self.cameras[camera_id] = self._form_a_camera(
            model, camera_id, width, height, params
        )
    def add_relative_transform_to_anchor(self,
                    image_id: str,
                    cvCam2anchor_T: np.ndarray,
                ) -> None:
        """
        @Brief: add a transform from camera_id to anchor to the anchor
        """
        self.transforms2anchor[image_id] = cvCam2anchor_T

    def _form_an_image(
        self,
        image_id,
        qvec,
        tvec,
        camera_id,
        image_name,
        xys,
        near_distance,
        far_distance,
    ):
        """
        @Brief: return a Image instance with the format
            MyImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    near_distance=near_distance,
                    far_distance=far_distance,
                )
        """
        return MyImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=xys,
            near_distance=near_distance,
            far_distance=far_distance,
            point3D_ids=None,
        )

    def add_colmap_image(self, image_id, colmap_image_instance):
        """
        @Brief: add an Image instance with the format
            colmap_image_instace = MyImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    near_distance=near_distance,
                    far_distance=far_distance,
                )
            to self.images
        """
        self.images[image_id] = colmap_image_instance

    def add_colmap_image_from_raw_data(
        self,
        image_id,
        qvec,
        tvec,
        camera_id,
        image_name,
        xys,
        near_distance,
        far_distance,
    ):
        """
        @Brief: add an Image instance with the format
            MyImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    near_distance=near_distance,
                    far_distance=far_distance,
                )
            to self.images
        """
        self.images[image_id] = self._form_an_image(
            image_id,
            qvec,
            tvec,
            camera_id,
            image_name,
            xys,
            near_distance,
            far_distance,
        )

    def write_colmap_data_to_files(self, ext=None):
        """
        @Brief: write colmap data to files
        """
        if self.ext is None or ext is not None:
            self.ext = ext
        if self.ext is None:
            self.ext = ".txt"
        elif self.ext not in [".txt", ".bin"]:
            raise (
                f"[Error] Given extension {self.ext} is not supported. Only support .txt or .bin"
            )

        write_model(self.cameras, self.images, self.colmap_root, self.ext)

    def write_transform2anchor_to_files(self, ext=".txt"):
        """
        @Brief: write transform2anchor data to files
        """
        file_path = os.path.join(self.transform2anchor_root, "transform2anchor" + ext)
        if ext == ".txt":
            with open(file_path, "w") as fid:
                for image_id, cvCam2anchor_T in self.transforms2anchor.items():
                    to_write = [image_id, *cvCam2anchor_T.flatten().tolist()]
                    line = " ".join([str(elem) for elem in to_write])
                    fid.write(line + "\n")

        elif ext == ".json":
            for image_id, cvCam2anchor_T in self.transforms2anchor.items():
                self.transforms2anchor[image_id] = cvCam2anchor_T.tolist()
            with open(file_path, "w") as fid:
                json.dump(self.transforms2anchor, fid)
        else:
            raise (
                f"[Error] Given extension {ext} is not supported. Only support .txt or .json"
            )

    def add_rgb_image(self, image_file, obs):
        self.rgb_images[image_file] = obs["rgb"][:, :, ::-1]  # Convert RGB to GBR

    def add_depth_image(self, image_file, obs):
        self.depth_images[image_file] = obs["depth"]

    def add_audio_response(self, audio_file, obs):
        self.audios[audio_file] = obs["audio"]

    def add_rir_file(self, rir_file, obs):
        self.rirs[rir_file] = obs["rir_file"]

    def write_rgbds_data_to_files(self):
        for image_file in self.rgb_images:
            cv2.imwrite(
                os.path.join(self.rgb_root, image_file), self.rgb_images[image_file]
            )
        for image_file in self.depth_images:
            float32_depth = self.depth_images[image_file]
            # Convert to uint16 with milimetter as the new unit
            uint16_depth = (float32_depth * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(self.depth_root, image_file), uint16_depth)

        for rir_file in self.rirs:
            copyfile(self.rirs[rir_file], os.path.join(self.rir_root, rir_file))

        for audio_file in self.audios:
            #TODO: check this if the normalization term is good

            norm_audio = self.audios[audio_file].transpose(1, 0)/10.
            soundfile.write(
                os.path.join(self.audio_root, audio_file),
                norm_audio,
                samplerate=self.audio_sample_rate,
                subtype='FLOAT' # 32 bit float
            )
            # import librosa.display
            # import  matplotlib.pyplot as plt
            # plt.subplot(211)
            # librosa.display.waveplot(self.audios[audio_file][0], sr=self.audio_sample_rate)
            # plt.subplot(212)
            # saved_audio = librosa.load(os.path.join(self.audio_root, audio_file), sr=None)[0]
            # librosa.display.waveplot(saved_audio, sr=self.audio_sample_rate)
            # plt.show()


    def write_data_to_pickle_file(self, data, scene_obs_file):
        with open(scene_obs_file, "wb") as fo:
            pickle.dump(data, fo)


class MyImage(MyBaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def read_images_text(path, colmap_format=False):
    """Read Image instances from reconstruction files given in .txt format
    By default, the file is given in a customized format which is similar to Colmap
    but does not contain some information such as 3D points.
    To read the file given in the original Colmap format, set colmap_format=True

    Args:
        path (str): path to the image file, include itself (images.txt)
        colmap_format (bool, optional): Whether the file is exactly in the Colmap fromat. Defaults to False.

    Returns:
        images [MyImage or Image]: Depend on the colmap_format boolean variable,
            this function will return a MyImage or Image instance
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = elems[0]
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                if colmap_format:
                    elems = fid.readline().split()
                    xys = np.column_stack(
                        [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                    )
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    near_distance = 0
                    far_distance = 0

                else:
                    near_distance = elems[10]
                    far_distance = elems[11]
                    xys = None
                    point3D_ids = None

                images[image_id] = MyImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                    near_distance=near_distance,
                    far_distance=far_distance,
                )
    return images


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = 0
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
                img.near_distance,
                img.far_distance,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras" + ext)) and os.path.isfile(
        os.path.join(path, "images" + ext)
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext="", colmap_format=False):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(
            os.path.join(path, "images" + ext), colmap_format=colmap_format
        )
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
    return cameras, images


def write_model(cameras, images, path, ext=".txt"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
    else:
        raise (f"[Error] Your extension is {ext}. Only Support .txt.")
    return cameras, images


def get_single_cam_params(colmap_cameras):
    """
    @Brief: Get camera intrinsics parameters from colmap data
    """
    # TODO: handle multiple camera cases.
    # For now, assume there is only a single camera
    list_of_keys = list(colmap_cameras.keys())
    cam = colmap_cameras[list_of_keys[0]]
    if cam.model == "PINHOLE":
        h, w, fx, fy = cam.height, cam.width, cam.params[0], cam.params[1]
    elif cam.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        h, w, fx, fy = cam.height, cam.width, cam.params[0], cam.params[0]

    # TODO: handle PINHOLE camera model.
    # For now, assume fx = fy
    assert abs(fx - fy) < 0.5, f"[Error] Assume fx = fy but your input {fx} != {fy}"
    f = fx
    return np.array([h, w, f]).reshape([3, 1])


def get_cvCam2W_transformations(
    colmap_images, order_poses_by_image_names=True, get_image_names=False
):
    """
    @Brief: get a list of transformations from world to OpenCV camera poses
            The order of poses that we store follows the order of image names
    @Args:
        - colmap_images (dict): map Image ids to MyImage instances
        - order_by_image_name (bool): Order poses by the corresponding image name or Not.
            By default, set it to be True because LLFF code (NeRF) reads images in a sorted order
    @Return:
        - c2w_mats (List[np.ndarray(4x4)]): list of transformations from OpenCV cam to the world
    """
    # List of (name, id) tuples
    image_names_ids = [
        (colmap_images[k].name, colmap_images[k].id) for k in colmap_images
    ]
    # Sort this tuple based on names
    sorted_image_names_ids = sorted(image_names_ids, key=lambda x: x[0])
    if order_poses_by_image_names:
        sorted_image_ids = [k[1] for k in sorted_image_names_ids]
    else:
        sorted_image_ids = [k[1] for k in image_names_ids]

    print(f"[Info] No of images : {len(sorted_image_ids)}")

    # Retrieve world to Opencv cam's transformations
    transmat_bottom_vector = np.array([0, 0, 0, 1.0]).reshape([1, 4])
    w2c_mats = []
    near_far_distances = []
    for k in sorted_image_ids:
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
    if get_image_names:
        return (
            c2w_mats,
            np.array(near_far_distances),
            [k[0] for k in sorted_image_names_ids],
        )
    return c2w_mats, np.array(near_far_distances)


def read_rgbd_images(basedir, filenames, rgb_ext=".jpg", depth_ext=".png"):
    """
    @Brief: read RGB and depth images located at {basedir}/images and {basedir}/depth correspondingly
        given the list filenames.
            Assume RGB and depth images are given in form of {image_name}.jpg and {image_name}.png

    """
    rgb_root = os.path.join(basedir, "images")
    depth_root = os.path.join(basedir, "depths")
    rgbd_images = {}
    num_rgbs = len(glob.glob(rgb_root + f"/*{rgb_ext}"))
    num_depths = len(glob.glob(depth_root + f"/*{depth_ext}"))

    assert (
        num_rgbs == num_depths
    ), f"[Error] No of RGB images must be equal to no of depth images. Given {num_rgbs} rgbs, {num_depths} depths"
    for filename in filenames:
        fileId = filename.split(".")[0]
        rgb_file = rgb_root + f"/{fileId}{rgb_ext}"
        depth_file = depth_root + f"/{fileId}{depth_ext}"
        rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file, -1)
        rgbd_images[filename] = [rgb, depth]
    return rgbd_images


def getPCLfromRGB_D(img, dmap, K, depth_scale=1, max_depth_trunc=10):
    """
    @Brief: get pointcloud from RGB and dmap
    @Args:
        - depth_scale (float): by default, just use 1.
            if using depth_scale > 1 --> make sure max_depth_trunc >= max(dmap) * depth_scale
        - max_depth_trunc (float): any value in dmap that dmap*depth_scale > max_depth_trunc will become max_depth_trunc
    """
    # dmap = dmap.astype(np.uint16)
    o3d_color = o3d.geometry.Image(img)

    o3d_depth = o3d.geometry.Image(dmap)
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=depth_scale,
        depth_trunc=max_depth_trunc,
        convert_rgb_to_intensity=False,
    )
    intrinsic_mat = o3d.camera.PinholeCameraIntrinsic(
        img.shape[1], img.shape[0], K[0, 0], K[1, 1], K[0, -1], K[1, -1]
    )
    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrinsic_mat)
    return pcl
