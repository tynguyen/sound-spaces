"""
@Brief: get RGB-sound data samples from a set of viewpoints
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
from typing import Dict, Any, List, Optional
from abc import ABC
import os
import argparse
import logging
import pickle
from collections import defaultdict
from attr.setters import convert
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
import cv2

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import (
    quat_to_angle_axis,
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_coeffs,
)
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, ShortestPathPoint
from soundspaces.tasks.audionav_task import merge_sim_episode_config
from soundspaces.utils import load_metadata
from soundspaces.simulator import SoundSpacesSim
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.utils import images_to_video_with_audio
import open3d as o3d


def convert_observation_to_frame(
    observation: Dict, is_depth_normalized=False
) -> np.ndarray:
    r"""Generate image of single frame from observation

    Args:
        observation: observation returned from an environment step().
    Returns:
        generated image of a single frame.
    """
    egocentric_view_l: List[np.ndarray] = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze()
        if is_depth_normalized:
            depth_map *= 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_l.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    assert len(egocentric_view_l) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=1)

    frame = egocentric_view

    return frame


# Transformation from OpenCV cam to OpenGL cam (habitat-sim)
global_cvCam_to_openglCam_T = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1],]
)
# By default, habitat uses a different convention for cartisian cordinate system
# Transformation from the habitat-sim world to Open3D, OpenCV world
global_hat2W_T = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1],])


class CustomSim(SoundSpacesSim):
    def get_opencvCam_to_world_transformation(self, cam_position, cam_quat):
        """
        @Brief: get the transformation from the OpenCV camera to the world (in computer vision convention)
        Note that in the OpenGL camera's coord, Y is upward, X is to the right, and Z is backward
             while in the OpenCV camera's coord, Y is downward,  X is to the right, and Z is forward.
             while what we're provided from the Replica dataset is w.r.t the OpenGL camera's coord.
        @Args:
            - cam_position (np.ndarray, (3,)): transition part of the transformation from sensor to habitat-sim
            - cam_quat(np.ndarray, (4,)): rotation part of the transformation from sensor to habitat-sim.
                the (possibly non-unit norm) quaternion in scalar-last (x, y, z, w)
        """
        openglCam_rotation_mat = scipy.spatial.transform.Rotation.from_quat(
            cam_quat
        ).as_matrix()
        openglCam_2hat_T = np.column_stack((openglCam_rotation_mat, cam_position))
        openglCam_2hat_T = np.vstack((openglCam_2hat_T, (0, 0, 0, 1)))
        cvCam2W_T = global_hat2W_T @ openglCam_2hat_T @ global_cvCam_to_openglCam_T
        return cvCam2W_T

    # It really depends on ...
    def transform_rgbd_to_world_pcl_openCV_convention(
        self, cv_K, rgb, depth, cam_position=None, cam_quat=None
    ):
        """
        @Brief: This is a depricated function used to test the intrinsics matrix in OpenCV format.
        Note that in the OpenGL camera's coord, Y is upward, X is to the right, and Z is backward
             while in the OpenCV camera's coord, Y is downward,  X is to the right, and Z is forward.
             while what we're provided from the Replica dataset is w.r.t the OpenGL camera's coord.
        Transform RGBD to pointcloud in the world's coord. This function makes use of an intrinsics matrix (OpenCV format)
        Make sure depth is already aligned in the rgb frame
        @Args:
            - cv_K (np.ndarray, 4x4): intrinsics parameters in OpenCV format
            - cam_position (np.ndarray, (3,)): transition part of the transformation from sensor to habitat-sim
            - cam_quat(np.ndarray, (4,)): rotation part of the transformation from sensor to habitat-sim.
                the (possibly non-unit norm) quaternion in scalar-last (x, y, z, w)
        """
        # Get 3D points in the OpenCV camera's coord
        depth = depth.squeeze()[None]  # 1 x H x W
        img_h, img_w = rgb.shape[:2]
        xs, ys = np.meshgrid(
            np.linspace(0, img_w - 1, img_w), np.linspace(0, img_h - 1, img_h),
        )
        xs = xs.reshape(1, img_h, img_w)
        ys = ys.reshape(1, img_h, img_w)

        # Unproject (OpenCV camera's coordinate)
        xys = np.vstack((xs * depth, ys * depth, depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(cv_K), xys)

        # Visualize the points
        pcl_points = xy_c0[:3, :].T
        pcl_cam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_points))

        # Transformation from OpenCV camera to Open3D world
        cvCam2W_T = self.get_opencvCam_to_world_transformation(cam_position, cam_quat)

        # Transform the (0,0,0) coordinate frame to obtain that of the aagent w.r.t the o3d's coord system
        pcl_cam.transform(cvCam2W_T)
        return pcl_cam

    def transform_rgbd_to_world_pcl(self, rgb, depth, cam_position=None, cam_quat=None):
        """
        @Brief: transform RGBD to pointcloud in the world's coord
        Make sure depth is already aligned in the rgb frame
        @Args:
            - global_hat2W_T (np.ndarray, 4x4): transformation from habitat-sim to open3d
            - cam_position (np.ndarray, (3,)): transition part of the transformation from sensor to habitat-sim
            - cam_quat(np.ndarray, (4,)): rotation part of the transformation from sensor to habitat-sim.
                the (possibly non-unit norm) quaternion in scalar-last (x, y, z, w)
        """
        depth = depth.squeeze()[None]  # 1 x H x W
        img_h, img_w = rgb.shape[:2]
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        xs, ys = np.meshgrid(np.linspace(-1, 1, img_w), np.linspace(1, -1, img_h))
        xs = xs.reshape(1, img_h, img_w)
        ys = ys.reshape(1, img_h, img_w)

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(self.rgb_intrinsics), xys)

        # Visualize the points
        pcl_points = xy_c0[:3, :].T
        pcl_cam = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_points))
        # return pcl_cam
        # o3d.visualization.draw_geometries([pcl])

        #
        cam_rotation_mat = scipy.spatial.transform.Rotation.from_quat(
            cam_quat
        ).as_matrix()
        cam_2hat_T = np.column_stack((cam_rotation_mat, cam_position))
        cam_2hat_T = np.vstack((cam_2hat_T, (0, 0, 0, 1)))
        cam2w_T = global_hat2W_T @ cam_2hat_T
        # print(f"[Info] Angle {angle}, rot mat to habitat \n {agent2hat_T}")
        # print(f"[Info] Angle {angle}, rot mat to world \n {agent2w_T}")

        # Transform the (0,0,0) coordinate frame to obtain that of the aagent w.r.t the o3d's coord system
        pcl_cam.transform(cam2w_T)
        return pcl_cam

    def compute_rgb_intrinsics(self):
        """
        @Brief: calculate the intrinsics parameters for the RGB sensor
        """
        h = self.config.RGB_SENSOR.HEIGHT
        w = self.config.RGB_SENSOR.WIDTH
        hfov = (self.config.RGB_SENSOR.HFOV) / 180.0 * np.pi
        vfov = 2 * np.arctan(np.tan(hfov / 2) * h / float(w))
        fx = 1.0 / np.tan(hfov / 2.0)
        fy = 1.0 / np.tan(vfov / 2.0)
        K = np.array(
            [
                [fx, 0.0, 0, 0.0],
                [0.0, fy, 0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Intrinsics in OpenGL format
        self.rgb_intrinsics = K

        # Compute intrinsics parameters in OpenCV format
        self.cv_rgb_intrinsics = np.array(
            [
                [fx * w / 2.0, 0.0, w / 2, 0.0],
                [0.0, fy * h / 2.0, h / 2, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def compute_depth_intrinsics(self):
        """
        @Brief: calculate the intrinsics parameters for the depth sensor
        """
        h = self.config.DEPTH_SENSOR.HEIGHT
        w = self.config.DEPTH_SENSOR.WIDTH
        hfov = (self.config.DEPTH_SENSOR.HFOV) / 180.0 * np.pi
        vfov = 2 * np.arctan(np.tan(hfov / 2) * h / float(w))
        fx = 1.0 / np.tan(hfov / 2.0)
        fy = 1.0 / np.tan(vfov / 2.0)
        K = np.array(
            [
                [fx, 0.0, 0.0, 0.0],
                [0.0, fy, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Intrinsics in OpenGL format
        self.depth_intrinsics = K

        # Compute intrinsics parameters in OpenCV format
        self.cv_depth_intrinsics = np.array(
            [
                [fx * w / 2.0, 0.0, w / 2, 0],
                [0.0, fy * h / 2.0, h / 2, 0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def convert_position_to_index(self, position: List) -> int:
        return self._position_to_index_mapping[self.position_encoding(position)]

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:

        if self.config.USE_RENDERED_OBSERVATIONS:
            self._sim.set_agent_state(position, rotation)
        else:
            agent = self._sim.get_agent(agent_id)
            new_state = self.get_agent_state(agent_id)
            new_state.position = position
            new_state.rotation = rotation

            # NB: The agent state also contains the sensor states in _absolute_
            # coordinates. In order to set the agent's body to a specific
            # location and have the sensors follow, we must not provide any
            # state for the sensors. This will cause them to follow the agent's
            # body
            new_state.sensor_states = {}
            agent.set_state(new_state, reset_sensors)
            return True

    def get_observations_with_audiogoal_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = True,
        is_denormalize_depth: bool = False,
    ) -> Optional[Observations]:
        """
        @Brief: get observations that include RGB + depth as well sa audiogoal measurement at a specific position and
        orientation specified by
        @Args:
            pos (np.ndarray (3,))
            rot_quat (np.ndarray (4,))
            is_denormalize_depth (bool): denormalize the depth value by 10 or not.
        """
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )  # TODO: what does this mean by setting reset_sensors=True?

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position, current_state.rotation, reset_sensors=False,
                )

        # Denormalize the depth measurements (habitat-lab normalize the depth values by 10)
        if is_denormalize_depth:
            observations["depth"] *= 10

        # Make sure to update hte receiver_position_index
        self._receiver_position_index = self.convert_position_to_index(position)
        # Get rotation angle from the given quaternion "rotation"
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        rotation_angle = (
            int(
                np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(rotation))[0]))
            )
            % 360
        )
        self._rotation_angle = rotation_angle

        """
        This rotation_angle is different from the azimuth_angle which is later used to find the RIR file .
        To convert it to the azimuth_angle, use self.azimuth_angle() method
        """

        # new_state = self.get_agent_state()
        # print(f"[Info] Prev state:  \n {current_state}")
        # print(f"[Info] New state:  \n {new_state}")

        if self.config.AUDIO.HAS_DISTRACTOR_SOUND:
            # by default, does not cache for distractor sound
            audiogoal = self._compute_audiogoal()
        else:
            joint_index = (
                self._source_position_index,
                self._receiver_position_index,
                self.azimuth_angle,
            )
            print(
                f"[Info] Audio rendering| source pos index {self._source_position_index} | azimuth_angle {self.azimuth_angle}| \
                    cur pos index {self._receiver_position_index}"
            )
            if joint_index not in self._audiogoal_cache:
                self._audiogoal_cache[joint_index] = self._compute_audiogoal()
            audiogoal = self._audiogoal_cache[joint_index]
        observations["audio"] = audiogoal
        return observations

    def get_current_distance_to_goal(self):
        return self._compute_euclidean_distance_between_sr_locations()


def main(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="ss_baselines/av_nav/config/audionav/{}/train_telephone/pointgoal_rgb.yaml".format(
            dataset
        ),
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--fps", default=30, type=int, help="Simulation FPS",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="room_0",
        help="Name of a specific scene that you want to extract RGB-S. If not given, all scenes given in data/metadata will be used!",
    )
    parser.add_argument(
        "--sound_name",
        type=str,
        default="person_10",
        help="The name of the sound source file (without the extension)",
    )
    parser.add_argument(
        "--visualize_mesh", action="store_true", help="Visualize the 3D env pcl or not"
    )
    parser.add_argument(
        "--visualize_obs", action="store_true", help="Visualize the observations or not"
    )
    parser.add_argument(
        "--test_cv_K",
        action="store_true",
        help="Test intrinsics parameters given in OpenCV format",
    )
    args = parser.parse_args()

    config = get_config(args.config_path, opts=args.opts)
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]

    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False

    # For now, do this, check the config file later
    # Do nor normalize the depth by 10 (in habitat-sim)
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 720  # 256
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = 1280  # 256
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 720  # 256
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 1280  # 256
    config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE = True
    config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS = args.fps
    config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 44100
    config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"

    config.freeze()
    simulator = None
    scene_obs = defaultdict(dict)
    num_obs = 0

    # Place to dump the RGB-S data
    scene_obs_dir = "data/scene_RGBS/" + dataset
    os.makedirs(scene_obs_dir, exist_ok=True)
    metadata_dir = "data/metadata/" + dataset

    # Prepare to visualize the mesh
    if args.visualize_mesh:
        o3d_visualizer = o3d.visualization.Visualizer()
        o3d_visualizer.create_window()

    # Prepare to visualize the observations
    if args.visualize_obs:
        cv2.namedWindow("RGBD")

    for scene in os.listdir(metadata_dir):
        # Store frames and audios to write to a video
        scene_frames = []
        scene_audios = []
        print(f"-----------------------------------------------")
        if args.scene and scene != args.scene.lower():
            print(
                f"[Info] Scene {args.scene} is requested so ignore this scene {scene} in /metadata folder!"
            )
            continue
        print(f"[Info] Scene {scene} is being simulated....")
        scene_obs = dict()
        scene_metadata_dir = os.path.join(metadata_dir, scene)
        points, graph = load_metadata(scene_metadata_dir)

        if dataset == "replica":
            scene_mesh_dir = os.path.join(
                "data/scene_datasets", dataset, scene, "habitat/mesh_semantic.ply"
            )
        else:
            scene_mesh_dir = os.path.join(
                "data/scene_datasets", dataset, scene, scene + ".glb"
            )

        # Visualize the env
        if args.visualize_mesh:
            # Map pointcloud
            scene_pcl_dir = os.path.join(
                "data/scene_datasets", dataset, scene, "mesh.ply"
            )
            map_mesh = o3d.io.read_point_cloud(scene_pcl_dir)
            o3d_visualizer.add_geometry(map_mesh)

            # O3d Origin
            o3d_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=np.array([0, 0, 0])
            )
            o3d_visualizer.add_geometry(o3d_origin)

            # Agent pose
            o3d_agent_pos = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=np.array([0, 0, 0])
            )
            o3d_visualizer.add_geometry(o3d_agent_pos)

            # Instant Observation
            o3d_obs_pcl = o3d.geometry.PointCloud()
            o3d_visualizer.add_geometry(o3d_obs_pcl)

        # Set a goal location
        goal_radius = 0.00001
        goal = NavigationGoal(
            position=(4.75, -1.55, -1.91), radius=goal_radius
        )  # index: 98
        agent_start_R = quat_to_coeffs(
            quat_from_angle_axis(np.deg2rad(0), np.array([0, 1, 0]))
        ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk
        episode = NavigationEpisode(
            goals=[goal],
            episode_id=str(0),
            scene_id=scene_mesh_dir,
            start_position=(-0.25, -1.55, 0.59),  # index: 8
            start_rotation=agent_start_R,
            info={"sound": args.sound_name},
        )

        # Simulation configs including RGB sensor, depth sensor ...
        episode_sim_config = merge_sim_episode_config(
            config.TASK_CONFIG.SIMULATOR, episode
        )
        if simulator is None:
            # simulator = Sim(episode_sim_config)
            simulator = CustomSim(episode_sim_config)
            simulator.reconfigure(episode_sim_config)
            # Compute sensors' intrinsics
            simulator.compute_rgb_intrinsics()
            simulator.compute_depth_intrinsics()
        for node in graph.nodes():
            print(f"-------------------------------------------")
            # All rotation and position here are w.r.t the habitat's coordinate system which is
            # different from that of O3d. We will transform them into the o3d's later for visualization
            agent_position = graph.nodes()[node]["point"]  # (3,)
            print(f"[Info] --> Agent pos: {agent_position}")
            print(
                f"[Info] --> Agent pos index: {simulator.convert_position_to_index(agent_position)}"
            )
            if args.visualize_mesh:
                print(f"[Info] Visualizing agent's pos at {agent_position}")

            for angle in [0, 90, 180, 270]:
                agent_rotation = quat_to_coeffs(
                    quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))
                ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk
                print(f"[Info] ----> Agent rot angle (in habitat): {angle}")

                if args.visualize_mesh:
                    # Get rotation matrix
                    agent_rot_mat = scipy.spatial.transform.Rotation.from_quat(
                        agent_rotation
                    ).as_matrix()
                    agent2hat_T = np.column_stack((agent_rot_mat, agent_position))
                    agent2hat_T = np.vstack((agent2hat_T, (0, 0, 0, 1)))
                    agent2w_T = global_hat2W_T @ agent2hat_T
                    # print(f"[Info] Angle {angle}, rot mat to habitat \n {agent2hat_T}")
                    # print(f"[Info] Angle {angle}, rot mat to world \n {agent2w_T}")

                    # Transform the (0,0,0) coordinate frame to obtain that of the aagent w.r.t the o3d's coord system
                    o3d_agent_pos.transform(agent2w_T)

                    # Update the visualization
                    # o3d_visualizer.update_geometry(o3d_agent_pos)

                    # Transform back to 0,0,0. This is just for visualization purpose
                    o3d_agent_pos.transform(np.linalg.inv(agent2w_T))

                # Get an observation at the new pose (RGB-S)
                """ Note that the audio is 1 second in length and binaural.
                    TODO: adjust this duration by changing
                        audiogoal = binaural_convolved[:, :sampling_rate] in soundspaces/simulator.py
                """
                obs = simulator.get_observations_with_audiogoal_at(
                    agent_position, agent_rotation
                )
                rotation_index = simulator._rotation_angle

                # Get sensors' states (in the habitat-sim world). Here, assume RGB and depth sensors are already aligned
                sim_cur_state = simulator.get_agent_state()
                rgbd_sstate = sim_cur_state.sensor_states[
                    "rgb"
                ]  # To access rotation, position, do: rgb_state.position, rgb_state.rotation

                # Get the camera's intrinsics parameters
                obs["cv_K"] = simulator.cv_rgb_intrinsics  # 4x4
                obs["gl_K"] = simulator.rgb_intrinsics  # 4x4

                # Get the transformation from OpenCV's camera to the world (opencv cam -> opengl cam ---
                # --> habitat-sim world -> opencv (open3d, our computer vision) world)
                obs["cvCam2W_T"] = simulator.get_opencvCam_to_world_transformation(
                    rgbd_sstate.position,
                    list(rgbd_sstate.rotation.vec) + [rgbd_sstate.rotation.w],
                )

                scene_obs[(node, rotation_index)] = obs
                num_obs += 1

                # Convert the observation to an image frame for demo videos
                frame = convert_observation_to_frame(obs)

                # Store
                # TODO: continuous view
                for _ in range(args.fps):
                    scene_frames.append(frame)
                scene_audios.append(obs["audio"])

                # Debug the observation
                if args.visualize_obs:
                    frame2Show = cv2.resize(frame, (1280, 720))
                    cv2.imshow("RGBD", frame)
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break

                if args.visualize_mesh:
                    if args.test_cv_K:
                        print(f"[Info] CV RGB K: \n {simulator.cv_rgb_intrinsics}")
                        o3d_new_pcl = simulator.transform_rgbd_to_world_pcl_openCV_convention(
                            simulator.cv_rgb_intrinsics,
                            obs["rgb"],
                            obs["depth"],
                            rgbd_sstate.position,
                            list(rgbd_sstate.rotation.vec) + [rgbd_sstate.rotation.w],
                            # for some reasons, this quaternion is given in the scalar-first format
                        )
                    else:
                        o3d_new_pcl = simulator.transform_rgbd_to_world_pcl(
                            obs["rgb"],
                            obs["depth"],
                            rgbd_sstate.position,
                            list(rgbd_sstate.rotation.vec) + [rgbd_sstate.rotation.w],
                            # for some reasons, this quaternion is given in the scalar-first format
                        )
                    o3d.visualization.draw_geometries([map_mesh, o3d_new_pcl])

        print(f"-----------------------------------------------")
        print("Total number of observations: {}".format(num_obs))
        scene_obs_file = os.path.join(scene_obs_dir, scene, "{}.pkl".format(scene))
        with open(scene_obs_file, "wb") as fo:
            pickle.dump(scene_obs, fo)
        print(
            f"[Info] Saved data simulated from scene {args.scene} to\n {scene_obs_file}"
        )

        # Save images & audios to a video
        # Place to dump the demo video
        video_dir = os.path.join("data/scene_RGBS/", dataset, scene)
        video_name = "demo"
        images_to_video_with_audio(
            scene_frames,
            video_dir,
            video_name,
            scene_audios,
            sr=config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
            fps=args.fps,
        )
        print(f"[Info] Saved video {video_dir}/{video_name}.mp4")

    if not args.visualize_mesh:
        simulator.close()
        del simulator

    if args.visualize_mesh:
        o3d_visualizer.destroy_window()

    if args.visualize_obs:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Caching Replica observations ...")
    main("replica")
    # print("Caching Matterport3D observations ...")
    # main("mp3d")
