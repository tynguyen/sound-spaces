"""
@Brief: get RGB-sound data samples from a set of viewpoints
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
from typing import Dict, Any, List, Optional
from abc import ABC
import os
import logging
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
from utils.colmap_read_write import ColmapDataWriter
from utils.colmap_read_write import getPCLfromRGB_D


def main(dataset):
    args = get_args(dataset)
    # Place to dump the RGB-S data
    scene_obs_dir = os.path.join(args.data_saving_root, args.scene)
    data_writer = ColmapDataWriter(
        scene_obs_dir, audio_sample_rate=args.audio_sample_rate
    )

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
    config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = args.audio_sample_rate
    config.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"

    config.freeze()
    simulator = None
    # scene_obs = defaultdict(dict)
    num_obs = 0

    metadata_dir = "data/metadata/" + dataset

    # Prepare to visualize the mesh
    if args.visualize_mesh:
        o3d_visualizer = o3d.visualization.Visualizer()
        o3d_visualizer.create_window()
        pcl_list = []
        num_poses_to_show = None

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
            pcl_list.append(map_mesh)
            # o3d_visualizer.add_geometry(map_mesh)

            # # O3d Origin
            # o3d_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #     size=0.2, origin=np.array([0, 0, 0])
            # )
            # o3d_visualizer.add_geometry(o3d_origin)

            # # Instant Observation
            # o3d_obs_pcl = o3d.geometry.PointCloud()
            # o3d_visualizer.add_geometry(o3d_obs_pcl)

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
                num_obs += 1
                agent_rotation = quat_to_coeffs(
                    quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))
                ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk
                print(f"[Info] ----> Agent rot angle (in habitat): {angle}")

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
                    [rgbd_sstate.rotation.w] + list(rgbd_sstate.rotation.vec),
                )

                # scene_obs[(node, rotation_index)] = obs

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
                            [rgbd_sstate.rotation.w] + list(rgbd_sstate.rotation.vec),
                            # this quaternion is given in the scalar-first format
                        )
                    else:
                        o3d_new_pcl = (
                            simulator.transform_rgbd_to_world_pcl(
                                obs["rgb"],
                                obs["depth"],
                                rgbd_sstate.position,
                                [rgbd_sstate.rotation.w]
                                + list(rgbd_sstate.rotation.vec),
                                # for some reasons, this quaternion is given in the scalar-first format
                            ),
                        )
                    pcl_list.append(o3d_new_pcl)
                    o3d_cam_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.2
                    )
                    o3d_cam_pose.transform(obs["cvCam2W_T"])
                    pcl_list.append(o3d_cam_pose)
                    if num_poses_to_show is not None and num_obs >= num_poses_to_show:
                        break

                # Prepare data to save in the Colmap format (except no Point3D) and add near distance, far distance values to Camera instances
                # TODO: for now, assume there is only a single camera.
                camera_id = 0
                data_writer.add_camera(
                    model=simulator.colmap_pinhole_rgb_cam.model_name,
                    camera_id=camera_id,
                    width=simulator.colmap_pinhole_rgb_cam.width,
                    height=simulator.colmap_pinhole_rgb_cam.height,
                    params=simulator.colmap_pinhole_rgb_cam.params,
                )

                # Colmap Image
                image_id = f"nodeID_{node}_angleID_{angle}"
                colmap_image_instance = simulator.get_colmap_image_instance_from_observation(
                    f"{image_id}.jpg",
                    image_id,
                    camera_id,
                    obs,
                    rgbd_sstate.position,
                    [rgbd_sstate.rotation.w] + list(rgbd_sstate.rotation.vec),
                )

                data_writer.add_colmap_image(image_id, colmap_image_instance)

                # RGBD-S data
                data_writer.add_rgb_image(f"{image_id}.jpg", obs)
                data_writer.add_depth_image(f"{image_id}.png", obs)
                data_writer.add_audio_response(f"{image_id}.wav", obs)
                data_writer.add_rir_file(f"{image_id}.wav", obs)
            if args.visualize_mesh:
                if num_poses_to_show is not None and num_obs >= num_poses_to_show:
                    break
        if args.visualize_mesh:
            o3d.visualization.draw_geometries(pcl_list)
            o3d_visualizer.destroy_window()

        print(f"-----------------------------------------------")
        print("Total number of observations: {}".format(num_obs))
        # scene_obs_pkl_file = os.path.join(scene_obs_dir, "{}.pkl".format(scene))
        # Simply dumpe the data to a pickle file
        # data_writer.write_data_to_pickle_file(scene_obs, scene_obs_pkl_file)
        # print(
        #     f"[Info] Saved data simulated from scene {args.scene} to\n {scene_obs_pkl_file}"
        # )

        # Write colmap data
        data_writer.write_colmap_data_to_files(".txt")
        print(f"[Info] Saved colmap data from scene {args.scene} to\n {scene_obs_dir}")

        # Write RGB-S data
        data_writer.write_rgbds_data_to_files()
        print(f"[Info] Saved RGBD-S data from scene {args.scene} to\n {scene_obs_dir}")

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

    if args.visualize_obs:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Caching Replica observations ...")
    main("replica")
    # print("Caching Matterport3D observations ...")
    # main("mp3d")
