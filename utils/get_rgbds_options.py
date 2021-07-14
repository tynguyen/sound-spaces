import argparse


def get_args(dataset="replica"):
    """
    @Brief: parse input arguments.
    @Usage:  args = get_args(dataset)
    """
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

    return parser.parse_args()
