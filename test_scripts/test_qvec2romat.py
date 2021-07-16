import numpy as np
import scipy.spatial.transform as tf
from habitat_sim.utils.common import quat_from_angle_axis


def qvec2rotmat(qvec):
    """
    @Brief: quaternion in from q0 + i*q1 + j*q2 + k*q3 to rotation matrix
    @Args:
        qvec (List, np.array): UNIT quaternion
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def test():
    for i in range(100):
        angle = np.random.rand() * 180
        qvec = quat_from_angle_axis(
            np.deg2rad(angle), np.array([1, 0, 0])
        ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk
        q1, q2, q3 = qvec.vec
        q0 = qvec.w
        print("\n----------------\n", qvec)  # q0 -> q3
        scipy_rotmat = tf.Rotation.from_quat([q1, q2, q3, q0]).as_matrix()
        our_rotmat = qvec2rotmat([q0, q1, q2, q3])

        # Compare the two
        distance = np.linalg.norm(our_rotmat - scipy_rotmat)
        print(f"[Info] our:\n {our_rotmat}")
        print(f"[Info] Scipy:\n {scipy_rotmat}")
        print(f"[Info] L2 error: {distance}")
        if distance > 1e-4:
            print("FUKKK")
            exit()


if __name__ == "__main__":
    test()
