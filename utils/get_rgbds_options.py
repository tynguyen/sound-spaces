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
        "--data_saving_root",
        type=str,
        default=f"data/scene_colmaps/{dataset}",
        help="Root to the place that we want to write the data to",
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
        "--audio_sample_rate", default=44100, type=int, help="Audio sample rate",
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
        "--num_obs_to_generate",
        default=1e10,
        type=int,
        help="Number of observations to generate",
    )

    parser.add_argument(
        "--start_node",
        type=int,
        default=50,
        help="Index of the starting position of the agent. This is not important because we read the locations given in the graph and obtain the observation at that location",
    )

    parser.add_argument(
        "--monoaudio",
        action="store_true",
        help="If setting, save only audio for the left-ear microphone",
    )

    parser.add_argument(
        "--modalities",
        type=str,
        default="RGBDS",
        help="Modalities in the dataset to save. By default, save both RGBD and sound",
    )

    parser.add_argument(
        "--goal_node",
        type=int,
        default=0,
        help="Index of the goal's position where the agent plans to reach. This is also the sound source location",
    )

    parser.add_argument(
        "--agent_path",
        type=str,
        default="",
        help="Path to a json file that specifies the agent path e.g: node (index) and angle (degrees)",
    )

    parser.add_argument(
        "--visualize_mesh", action="store_true", help="Visualize the 3D env pcl or not"
    )
    parser.add_argument(
        "--visualize_obs", action="store_true", help="Visualize the observations or not"
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save visualization video or not"
    )
    parser.add_argument(
        "--test_cv_K",
        action="store_true",
        help="Test intrinsics parameters given in OpenCV format",
    )

    return parser.parse_args()
