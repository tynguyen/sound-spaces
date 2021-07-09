import pickle as pkl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import open3d as o3d

pkl_file = "data/metadata/replica/room_0/graph.pkl"
mesh_file = "data/scene_datasets/replica/room_0/mesh.ply"

with open(pkl_file, "rb") as fo:
    graph = pkl.load(fo)
    # Visualize the graph
    nx.draw(graph, with_labels=True)
    plt.show()

mesh = o3d.io.read_point_cloud(mesh_file)
o3d.visualization.draw_geometries([mesh])
