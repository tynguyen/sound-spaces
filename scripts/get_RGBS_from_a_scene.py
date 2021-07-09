"""
@Brief: get RGB-sound data samples from a set of viewpoints
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
from typing import Any, List, Optional
from abc import ABC
import os
import argparse
import logging
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Simulator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
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
import open3d as o3d


class Sim(SoundSpacesSim):
    def step(self, action):
        sim_obs = self._sim.get_sensor_observations()
        return sim_obs, self._rotation_angle


def updateO3dPose(visualizer, old_pose, new_pose):
    """
    @Brief: non-blocking update the pose
    """
    old_pose.vertices = old_pose.vertices
    old_pose.triangles = old_pose.triangles
    old_pose.vertex_colors = old_pose.vertex_colors
    visualizer.update_geometry(old_pose)


def refreshO3dVis(visualizer):
    """
    @Brief: refresh the open3d visualzier to reflect new appearances of objects
    """
    visualizer.poll_events()
    visualizer.update_renderer()


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
        "--scene",
        type=str,
        default="room_0",
        help="Name of a specific scene that you want to extract RGB-S. If not given, all scenes given in data/metadata will be used!",
    )
    parser.add_argument(
        "--visualize_mesh", action="store_true", help="Debug the mesh or not"
    )
    args = parser.parse_args()

    config = get_config(args.config_path, opts=args.opts)
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
    config.freeze()
    simulator = None
    scene_obs = defaultdict(dict)
    num_obs = 0
    # Place to dump the RGB-S data
    scene_obs_dir = "data/scene_observations/" + dataset
    os.makedirs(scene_obs_dir, exist_ok=True)
    metadata_dir = "data/metadata/" + dataset

    # Prepare to visualize the mesh
    if args.visualize_mesh:
        o3d_visualizer = o3d.visualization.Visualizer()
        o3d_visualizer.create_window()

    for scene in os.listdir(metadata_dir):
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

            # By default, habitat uses a different convention for cartisian cordinate system
            hat2w_T = np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1],]
            )

        for node in graph.nodes():
            # All rotation and position here are w.r.t the habitat's coordinate system which is
            # different from that of O3d. We will transform them into the o3d's later for visualization
            agent_position = graph.nodes()[node]["point"]  # (3,)
            if args.visualize_mesh:
                print(f"[Info] Visualizing agent's pos at {agent_position}")

            for angle in [0, 90, 180, 270]:
                agent_rotation = quat_to_coeffs(
                    quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))
                ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk

                if args.visualize_mesh:
                    # Get rotation matrix
                    agent_rot_mat = scipy.spatial.transform.Rotation.from_quat(
                        agent_rotation
                    ).as_matrix()
                    agent2hat_T = np.column_stack((agent_rot_mat, agent_position))
                    agent2hat_T = np.vstack((agent2hat_T, (0, 0, 0, 1)))
                    agent2w_T = hat2w_T @ agent2hat_T
                    print(f"[Info] Angle {angle}, rot mat to habitat \n {agent2hat_T}")
                    print(f"[Info] Angle {angle}, rot mat to world \n {agent2w_T}")

                    # Transform the (0,0,0) coordinate frame to obtain that of the aagent w.r.t the o3d's coord system
                    o3d_agent_pos.transform(agent2w_T)

                    # Update the visualization
                    o3d_visualizer.update_geometry(o3d_agent_pos)
                    o3d_visualizer.poll_events()
                    o3d_visualizer.update_renderer()

                    # Transform back to 0,0,0. This is just for visualization purpose
                    o3d_agent_pos.transform(np.linalg.inv(agent2w_T))

                goal_radius = 0.00001
                goal = NavigationGoal(position=agent_position, radius=goal_radius)
                episode = NavigationEpisode(
                    goals=[goal],
                    episode_id=str(0),
                    scene_id=scene_mesh_dir,
                    start_position=agent_position,
                    start_rotation=agent_rotation,
                    info={"sound": "telephone"},
                )

                if not args.visualize_mesh:
                    # Simulation configs including RGB sensor, depth sensor ...
                    episode_sim_config = merge_sim_episode_config(
                        config.TASK_CONFIG.SIMULATOR, episode
                    )
                    if simulator is None:
                        simulator = Sim(episode_sim_config)
                    simulator.reconfigure(episode_sim_config)

                    obs, rotation_index = simulator.step(None)
                    scene_obs[(node, rotation_index)] = obs
                    num_obs += 1

                # Debug the observation
                # plt.imshow(obs["rgb"])
                # plt.show()

        print(f"-----------------------------------------------")
        print("Total number of observations: {}".format(num_obs))
        scene_obs_file = os.path.join(scene_obs_dir, "{}.pkl".format(scene))
        with open(scene_obs_file, "wb") as fo:
            pickle.dump(scene_obs, fo)
        print(
            f"[Info] Saved data simulated from scene {args.scene} to\n {scene_obs_file}"
        )

    if not args.visualize_mesh:
        simulator.close()
        del simulator

    if args.visualize_mesh:
        o3d_visualizer.destroy_window()


if __name__ == "__main__":
    print("Caching Replica observations ...")
    main("replica")
    # print("Caching Matterport3D observations ...")
    # main("mp3d")
