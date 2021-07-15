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


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def test_quat2rotmat():
    iters = 1000
    total_errors = 0
    for i in range(iters):
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
            print(f"Error {distance} is too big!!! Failed!")
            exit()
        total_errors += distance

    print(f"[Succeeded!] Total L2 error after {iters} times: {total_errors}")


def test_rotmat2quat():
    iters = 1000
    total_errors = 0
    for i in range(iters):
        angle = np.random.rand() * 180
        qvec = quat_from_angle_axis(
            np.deg2rad(angle), np.array([1, 0, 0])
        ).tolist()  # [b, c, d, a] where the unit quaternion would be a + bi + cj + dk
        q1, q2, q3 = qvec.vec
        q0 = qvec.w
        print("\n----------------\n", qvec)  # q0 -> q3
        scipy_rotmat = tf.Rotation.from_quat([q1, q2, q3, q0]).as_matrix()

        # Convert back to qvec
        est_qvec = rotmat2qvec(scipy_rotmat)

        # Compare the two
        distance = np.linalg.norm(np.array([q0, q1, q2, q3] - est_qvec))
        print(f"[Info] Orig:\n {np.array([q0, q1, q2, q3])}")
        print(f"[Info] Est:\n {est_qvec}")
        print(f"[Info] L2 error: {distance}")
        if distance > 1e-4:
            print(f"Error {distance} is too big! Failed!")
            exit()

        total_errors += distance

    print(f"[Succeeded!] Total L2 error after {iters} times: {total_errors}")


if __name__ == "__main__":
    # test_quat2rotmat()
    test_rotmat2quat()
