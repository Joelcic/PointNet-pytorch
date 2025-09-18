import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import imageio


def create_custom_coord_frame(size=1.0, origin=[0, 0, 0]):
    """
    Creates a custom 3D coordinate frame as an Open3D LineSet.

    This function generates a simple visual representation of the 3D coordinate axes
    using colored lines:
        - X-axis in blue
        - Y-axis in green
        - Z-axis in red

    Useful for visualization and orientation in 3D scenes.

    Parameters:
        size (float): Length of each axis from the origin. Default is 1.0.
        origin (list or np.ndarray): The 3D coordinates of the frame origin. Default is [0, 0, 0].
    Returns:
        line_set (open3d.geometry.LineSet): A LineSet object representing the coordinate frame.
    """
    origin = np.array(origin)

    # Axis endpoints
    x_axis = origin + np.array([size, 0, 0])
    y_axis = origin + np.array([0, size, 0])
    z_axis = origin + np.array([0, 0, size])

    # Create 3 line segments (x, y, z axes)
    points = [origin, x_axis, origin, y_axis, origin, z_axis]
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]  # x=blue, y=green, z=red

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_fixed_size_grid_xy(min_bound, max_bound, block_size=2.0, color=[0.6, 0.6, 0.6]):
    """
    Creates a grid of 2D blocks (extended in Z) across the point cloud space.
    Each block is block_size × block_size in XY, full height in Z.
    """
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)

    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound

    x_coords = np.arange(x_min, x_max, block_size)
    y_coords = np.arange(y_min, y_max, block_size)

    line_sets = []

    for x in x_coords:
        for y in y_coords:
            cube_min = np.array([x, y, z_min])
            cube_max = np.array([x + block_size, y + block_size, z_max])
            box = o3d.geometry.AxisAlignedBoundingBox(cube_min, cube_max)
            line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
            line_set.paint_uniform_color(color)
            line_sets.append(line_set)

    return line_sets

def viz(pcd, add_coord_frame=False, add_grid=False):
    """
    Visualizes a point cloud using Open3D with optional coordinate frame and grid overlay.

    This function opens an Open3D window and renders the given point cloud. It offers
    options to include a custom coordinate frame and a fixed-size XY grid to aid in
    spatial understanding of the scene.

    Parameters:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize.
        add_coord_frame (bool): If True, adds a custom coordinate frame to the scene.
                                X=blue, Y=green, Z=red. Default is False.
        add_grid (bool): If True, adds a fixed-size XY grid beneath the point cloud
                         based on its bounding box. Default is False.
    Camera Settings:
        A predefined camera pose is used to ensure consistent viewing angles.
    Visualization Settings:
        - Point size is adjustable via `render_option.point_size`.
    Returns:
        None: The function opens a visualization window and does not return a value.
    """
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Calculate bounds of the point cloud
    if add_grid:
        bounds = pcd.get_axis_aligned_bounding_box()
        min_bound = bounds.min_bound
        max_bound = bounds.max_bound
        # Add grid lines
        grid_lines = create_fixed_size_grid_xy(min_bound, max_bound, block_size=2.0, color=[0.4, 0.4, 0.4])
        for g in grid_lines:
            vis.add_geometry(g)

    # Add custom coordinate frame with x=blue, y=green, z=red
    if add_coord_frame:
        coord_frame = create_custom_coord_frame(size=5)
        vis.add_geometry(coord_frame)

    # Adjust point size
    render_option = vis.get_render_option()
    render_option.point_size = 0.5  # Adjust this value for smaller/larger points
    camera_pose = {
        "front": [0.50169368912941492, -0.80011328446582886, 0.32881936425493824],
        "lookat": [-0.085999999999999965, 3.3594999999999997, -0.24299999999999999],
        "up": [-0.07356928702117288, 0.33927617588917025, 0.93780554299983609],
        "zoom": 1.8599999999999999
    }

    # Set camera parameters
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(camera_pose["zoom"])
    view_ctl.set_lookat(camera_pose["lookat"])
    view_ctl.set_front(camera_pose["front"])
    view_ctl.set_up(camera_pose["up"])

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def plot_train_loss(file_path):
    """
    Plots the training loss over epochs from a CSV file.

    This function reads a CSV file containing training loss values recorded during
    model training. It expects the CSV file to have a column named 'train_loss'.
    The function then plots the training loss with respect to the number of epochs.

    Parameters:
        file_path (str): The path to the CSV file containing the training loss data.
    Returns:
        None: The function displays a plot and does not return any value.
    """
    # Load CSV
    df = pd.read_csv(file_path)
    epochs = df.index
    # Plotting
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training(file_path):
    """
    Plots training and validation accuracy over epochs from a CSV file.

    This function reads a CSV file that contains training and validation accuracy
    values recorded during model training. It expects the file to have two columns:
    'train_accuracy' and 'val_accuracy'. The function plots both metrics over the
    number of epochs to help visualize model performance over time.

    Parameters:
        file_path (str): The path to the CSV file containing the accuracy data.
    Returns:
        None: The function displays the plot and does not return any value.
    """
    # Load CSV
    df = pd.read_csv(file_path)
    epochs = df.index
    # Plotting
    plt.plot(epochs, df["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def bar_plot_classes(dataset_path):
    """
    Generates a bar plot showing the distribution of point classes in the S3DIS dataset.

    This function reads labeled point cloud data from multiple areas of the S3DIS dataset,
    located at the specified dataset path. It counts the number of occurrences of each
    semantic class label (e.g., wall, floor, chair) and visualizes the class distribution
    as a bar chart.

    Parameters:
        dataset_path (str): Path to the root directory of the partitioned S3DIS dataset.
                            Expected to contain folders named Area_1 through Area_6.

    Assumptions:
        - Each area's folder may contain nested HDF5 (.hdf5) files with point cloud data.
        - Labels are located in column index 3 (i.e., the fourth column) of the data.
        - Class IDs follow a predefined mapping (`CATEGORIES` dictionary).

    Returns:
        None: The function displays a bar plot and does not return a value.
    """
    CATEGORIES = {
        'ceiling'  : 0,
        'floor'    : 1,
        'wall'     : 2,
        'beam'     : 3,
        'column'   : 4,
        'window'   : 5,
        'door'     : 6,
        'table'    : 7,
        'chair'    : 8,
        'sofa'     : 9,
        'bookcase' : 10,
        'board'    : 11,
        'stairs'   : 12,
        'clutter'  : 13
    }

    area_nums = [1,2,3,4,5,6]

    areas = list()
    for area in area_nums:
        areas.append(os.path.join(dataset_path, f'Area_{area}'))
    all_labels = []

    # Assuming `areas` is a list of folder paths like ['Area_1', 'Area_2', ...]
    for area in areas:
        hdf5_files = glob(os.path.join(area, '**', '*.hdf5'), recursive=True)

        for file_path in hdf5_files:
            data = pd.read_hdf(file_path).to_numpy()
            labels = data[:, 3]  # column 3 = class ID
            all_labels.extend(labels)

    # Count labels
    label_counts = Counter(all_labels)

    # Invert class mapping
    id_to_class = {v: k for k, v in CATEGORIES.items()}
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame({
        'Class': [id_to_class[int(k)] for k in sorted(label_counts)],
        'Count': [label_counts[k] for k in sorted(label_counts)]
    })

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df['Class'], plot_df['Count'])
    plt.xlabel('Class')
    plt.ylabel('Number of Points')
    plt.title('Point Distribution per Class in S3DIS')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def plot_confusion_matrix(conf_matrix_path):
    """
    Loads a saved confusion matrix from a CSV file and visualizes it as a heatmap.

    This function reads a CSV file containing a confusion matrix (with class names
    as both row and column headers) and generates a heatmap using seaborn for
    visual interpretation.

    Parameters:
        conf_matrix_path (str): Path to the CSV file containing the confusion matrix.
                                The CSV should have class names as both row index and column headers.
    Returns:
        None: The function displays the heatmap and does not return a value.
    """
    # Load the saved confusion matrix
    cm_df = pd.read_csv(conf_matrix_path, index_col=0)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=0.5, linecolor='lightgray')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def visualization(pcd, add_grid=False):
    """
    Visualizes a point cloud using Open3D with optional coordinate frame and grid overlay.

    This function opens an Open3D window and renders the given point cloud. It offers
    options to include a custom coordinate frame and a fixed-size XY grid to aid in
    spatial understanding of the scene.

    Parameters:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize.
        add_coord_frame (bool): If True, adds a custom coordinate frame to the scene.
                                X=blue, Y=green, Z=red. Default is False.
        add_grid (bool): If True, adds a fixed-size XY grid beneath the point cloud
                         based on its bounding box. Default is False.
    Camera Settings:
        A predefined camera pose is used to ensure consistent viewing angles.
    Visualization Settings:
        - Point size is adjustable via `render_option.point_size`.
    Returns:
        None: The function opens a visualization window and does not return a value.
    """
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Calculate bounds of the point cloud
    if add_grid:
        bounds = pcd.get_axis_aligned_bounding_box()
        min_bound = bounds.min_bound
        max_bound = bounds.max_bound
        # Add grid lines
        grid_lines = create_fixed_size_grid_xy(min_bound, max_bound, block_size=2.0, color=[0.5,0.5,0.5])
        for g in grid_lines:
            vis.add_geometry(g)

    # Adjust point size
    render_option = vis.get_render_option()
    render_option.point_size = 1.0  # Adjust this value for smaller/larger points

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def create_gif_from_3D(file_path, gif_output_path="pointcloud_rotation.gif",
                       start_angle=0, end_angle=360,
                       frame_step=4, point_size=0.5,
                       ping_pong=True):
    """
    Creates a rotating GIF of a 3D point cloud from a .pcd file using Open3D.
    Rotation goes from start_angle to end_angle, and optionally back (ping-pong).

    Parameters:
    - file_path: str or o3d.geometry.PointCloud, path to the .pcd file or a point cloud object
    - gif_output_path: str, output path for the GIF
    - start_angle: int, starting angle of rotation (degrees)
    - end_angle: int, ending angle of rotation (degrees)
    - frame_step: int, step in degrees between frames
    - point_size: float, Open3D point size for visualization
    - ping_pong: bool, if True, rotate back from end_angle to start_angle (ping-pong effect)
    """

    if isinstance(file_path, str):
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError("Point cloud has no points. Check file format or content.")
    elif isinstance(file_path, o3d.geometry.PointCloud):
        pcd = file_path
    else:
        raise ValueError("Input variable 'file_path' must be a string path or an Open3D PointCloud.")

    """# Apply optional orientation adjustment
    R_z = pcd.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 2])
    R_y = pcd.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
    R_x = pcd.get_rotation_matrix_from_axis_angle([np.deg2rad(10), 0, 0])
    pcd.rotate(R_z, center=(0, 0, 0))
    pcd.rotate(R_y, center=(0, 0, 0))
    pcd.rotate(R_x, center=(0, 0, 0))"""

    R_x = pcd.get_rotation_matrix_from_axis_angle([-np.pi / 2, 0, 0])
    #R_x2 = pcd.get_rotation_matrix_from_axis_angle([np.deg2rad(10), 0, 0])
    pcd.rotate(R_x, center=(0, 0, 0))
    #pcd.rotate(R_x2, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = point_size

    ctr = vis.get_view_control()
    # Rotate view to start_angle
    ctr.rotate(start_angle * 6, 0.0)

    images = []

    # Create forward angles list
    forward_angles = list(range(start_angle, end_angle + 1, frame_step))
    if ping_pong:
        # Create backward angles list (excluding the last frame to avoid duplication)
        backward_angles = list(range(end_angle - frame_step, start_angle - 1, -frame_step))
        angles = forward_angles + backward_angles
    else:
        angles = forward_angles

    for i in range(len(angles) - 1):
        angle_diff = angles[i + 1] - angles[i]
        ctr.rotate(angle_diff * 6, 0.0)  # rotate horizontally by difference * 6 pixels per degree
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(False)
        img = (np.asarray(img)[:, :, :3] * 255).astype(np.uint8)
        images.append(img)

    vis.destroy_window()

    imageio.mimsave(gif_output_path, images, duration=120, loop=0)
    print(f"GIF saved to {gif_output_path}")


