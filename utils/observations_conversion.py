import numpy as np
from typing import Dict, Any, List, Optional


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
