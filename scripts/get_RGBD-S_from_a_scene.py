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
from utils.rgbds_simulator import CustomSim
from utils.rgbds_simulator import global_hat2W_T
from utils.observations_conversion import convert_observation_to_frame
from utils.get_rgbds_options import get_args


def main(dataset):
    args = get_args(dataset)
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
