import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import sample_block

def simple_visualization(pcd, labels=None):
    """
    Visualizes a point cloud with optional semantic label coloring using Open3D.

    This function displays a point cloud in an interactive Open3D window. If class
    labels are provided, it uses a categorical colormap (e.g., 'tab20') to color points
    based on their labels. Otherwise, a uniform color is applied to the entire point cloud.

    Parameters:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize.
        labels (np.ndarray, optional): An array of shape (N,) with integer class labels
                                       for each point in the point cloud. Default is None.
    Visualization Settings:
        - Uses a fixed point size (adjustable via `render_option.point_size`).
        - Color map used is matplotlib's 'tab20', which supports up to 20 distinct classes.
    Returns:
        None: The function displays the visualization and does not return a value.
    """

    if labels is not None:
        # Colorize using labels → pick a colormap
        num_classes = int(labels.max() + 1)
        default_color = [0.6, 0.6, 0.6]  # Gray
        # build a color map dict from tab20
        tab20 = plt.get_cmap("tab20")
        color_map = {i: tab20(i)[:3] for i in range(num_classes)}  
        colors = np.array([color_map.get(label, default_color) for label in labels])

        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color
        pcd.paint_uniform_color([0.2, 0.5, 0.8])

    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # Adjust point size
    render_option = vis.get_render_option()
    render_option.point_size = 1  # Adjust this value for smaller/larger points

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    file_path = "S3DIS/conferenceRoom_1.hdf5"

    data = pd.read_hdf(file_path).to_numpy()
    # Split into coordinates and labels
    points = data[:, :3]
    labels = data[:, 3].astype(int)

    # Sample points to illustrate what pointnet is trained on
    #points, labels = sample_block(points,labels,num_points=4096)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    nr_points = len(np.asarray(pcd.points))
    print("number points: ", nr_points)

    # Visualize
    simple_visualization(pcd, labels)


