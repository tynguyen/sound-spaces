import numpy as np
import scipy
import open3d as o3d
from typing import Dict, Any, List, Optional
from soundspaces.simulator import SoundSpacesSim
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from habitat_sim.utils.common import (
    quat_to_angle_axis,
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_coeffs,
)

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
