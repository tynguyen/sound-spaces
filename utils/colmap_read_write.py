"""
@Brief: A wrapper to log colmap data to files. This is modified from read_write.py in colmap/python
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""

import os, cv2
import collections

import soundfile
from soundspaces.mp3d_utils import Object
import numpy as np
import struct
import argparse
import pickle
from shutil import copyfile

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"],)
MyBaseImage = collections.namedtuple(
    "Image",
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

        # Create repos
        os.makedirs(self.rgb_root, exist_ok=True)
        os.makedirs(self.depth_root, exist_ok=True)
        os.makedirs(self.audio_root, exist_ok=True)
        os.makedirs(self.rir_root, exist_ok=True)
        os.makedirs(self.colmap_root, exist_ok=True)

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
            soundfile.write(
                os.path.join(self.audio_root, audio_file),
                self.audios[audio_file].transpose(1, 0),
                samplerate=self.audio_sample_rate,
            )

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


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


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


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
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
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = MyImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = MyImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
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
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            fid.write(" ".join(points_strings) + "\n")


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras" + ext)) and os.path.isfile(
        os.path.join(path, "images" + ext)
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
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
        images = read_images_text(os.path.join(path, "images" + ext))
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


def main():
    parser = argparse.ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="outut model format",
        default=".txt",
    )
    args = parser.parse_args()

    cameras, images = read_model(path=args.input_model, ext=args.input_format)

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))

    if args.output_model is not None:
        write_model(cameras, images, path=args.output_model, ext=args.output_format)


if __name__ == "__main__":
    main()
