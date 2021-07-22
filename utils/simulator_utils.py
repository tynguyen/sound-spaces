"""
@Brief: utility functions for the rgbds_simulator
@Author: Ty Nguyen
@Email: tynguyen@seas.upenn.edu
"""
import os
import numpy as np
import networkx as nx
import pickle as pkl
from copy import deepcopy


def load_points_dict(parent_folder):
    """Load points given by points.txt file in the Replica/matterport datasets

    Args:
        parent_folder (string): directory to the points.txt file (excluding the filename itself)

    Raises:
        FileExistsError: file not exist

    Returns:
        points_dict (dict): a mapping from point index to its 3-D points (in the OpenGL coord)
    """
    points_file = os.path.join(parent_folder, "points.txt")
    if not os.path.exists(points_file):
        raise FileExistsError(points_file + " does not exist!")
    # Convert from Opencv, O3d coordinate system to OpenGL's coordinate system.
    # After that, substract the height of the sensor (1.5528907 or 1.5) to make the sensor sit on the ground
    # This step is necessary because later on, we will specify the robot's height which is different from the sensor height
    # that people have used to create Replica or Matterport datasets
    if "replica" in parent_folder:
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(
            zip(points_data[:, 1], points_data[:, 3] - 1.5528907, -points_data[:, 2])
        )
    else:
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(
            zip(points_data[:, 1], points_data[:, 3] - 1.5, -points_data[:, 2])
        )
    point_idxes = [int(k) for k in points_data[:, 0]]
    points_dict = dict(zip(point_idxes, points))
    return points_dict


def create_graph_from_points_dict(points_dict, saving_file=None, graph_to_copy=None):
    """Given points_dict that maps indexes to 3-D points, create a graph

    Args:
        points_dict (dict): a mapping from indexes to 3-D points
        saving_file (str): absolute file name to save the graph (in pickle format). If None --> no saving
        graph_to_copy (networkx.Graph): if given, then we will first copy this graph and adding nodes later

    Returns:
        graph (networkx.graph): a graph that contains all points given by points_dict with
		graph.nodes[i]['point'] = 3-D point
    """
    if graph_to_copy is not None:
        graph = deepcopy(graph_to_copy)
        # graph.add_edges_from(graph_to_copy.edges)
    else:
        graph = nx.Graph()
    # Add nodes to the graph as well as dummy edges connecting these nodes
    prev_node = None
    for node, point in points_dict.items():
        if node not in graph.nodes:
            graph.add_node(node, point=point)
        if prev_node is not None:
            graph.add_edge(prev_node, node)
        prev_node = node
    if saving_file is not None:
        with open(saving_file, "wb") as fin:
            pkl.dump(graph, fin)
    return graph
