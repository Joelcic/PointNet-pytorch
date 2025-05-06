import numpy as np
import torch
import os
import open3d as o3d

def pointNetLoss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """
   Computes the total loss for PointNet, combining negative log-likelihood loss
   with a regularization term to encourage orthogonality of the input and feature
   transformation matrices.
   :param outputs: (torch.Tensor) The output predictions from the network of shape (B, N, C), where B is batch size, N is number of points, and C is number of classes.
   :param labels: (torch.Tensor) Ground truth class labels of shape (B, N).
   :param m3x3: (torch.Tensor) Input transformation matrix of shape (B, 3, 3) from T-Net.
   :param m64x64: (torch.Tensor) Feature transformation matrix of shape (B, 64, 64) from T-Net.
   :param alpha: (float, optional) Regularization strength for transformation matrices (default is 0.0001).
   :return:
       loss: (torch.Tensor) Combined loss value (classification + regularization).
    """
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)

    # Check if outputs are on CUDA
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()

    # Calculate matrix differences
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))

    # Compute the loss
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


def train(pointnet, optimizer, train_loader, device, val_loader=None, epochs=15, save=True, lr_scheduler=None):
    """
    Trains a PointNet model on a given dataset with optional validation and model saving.
    :param pointnet (torch.nn.Module): The PointNet model to be trained.
    :param optimizer (torch.optim.Optimizer): Optimizer used for training (e.g., Adam, SGD).
    :param train_loader (torch.utils.data.DataLoader):  DataLoader providing training data batches.
    :param device (torch.device): Device to run the model on (e.g., "cuda" or "cpu").
    :param val_loader (torch.utils.data.DataLoader):  DataLoader for validation data. If provided, validation accuracy will be computed at each epoch. (OPTIONAL)
    :param epochs (int): Number of training epochs (default is 15). (OPTIONAL)
    :param save (bool):  Whether to save the model if validation accuracy improves (default is True). (OPTIONAL)
    :param lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler to be stepped after each epoch, if provided. (OPTIONAL)
    :return:
        train_loss_list (list of float): Average training loss per epoch.
        train_acc_list (list of float): Training accuracy (%) per epoch.
        val_acc_list (list of float): Validation accuracy (%) per epoch.
    """
    best_val_acc = -1.0
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        batches = 0
        # Training phase
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()

            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0) * labels.size(1)
            correct_train += (predicted == labels).sum().item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        avg_epoch_loss = running_loss / batches
        train_acc = 100.0 * correct_train / total_train
        train_loss_list.append(avg_epoch_loss)
        train_acc_list.append(train_acc)

        # Validation phase
        pointnet.eval()
        correct = total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()
                outputs, __, __ = pointnet(inputs.transpose(1, 2))
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0) * labels.size(1)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        val_acc_list.append(val_acc)
        print("Epoch: ", epoch)
        print("correct", correct, "/", total)
        print('Valid accuracy: %d %%' % val_acc)

        # Save the model if current validation accuracy surpasses the best
        if save and val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(os.path.dirname(__file__), "pointnetmodel.pth")
            print("best_val_acc:", val_acc, "saving model at", path)
            torch.save(pointnet.state_dict(), path)
        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss_list, train_acc_list, val_acc_list

def sample_block(block_points, block_labels=None, num_points=4096):
    """
    Samples a fixed number of points from a block of point cloud data.
       If the number of available points is greater than or equal to `num_points`,
       a subset is sampled without replacement. Otherwise, sampling is done with
       replacement to reach the desired number.
    :param block_points (np.ndarray): A NumPy array of shape (N, D) representing N points with D-dimensional features (e.g., XYZ, RGB).
    :param block_labels (np.ndarray): A NumPy array of shape (N,) containing labels or classes for each point.
    :param num_points (int): The number of points to sample. Default is 4096.
    :return:
        Sampled point features of shape (num_points, D).
        Corresponding labels of shape (num_points,).
    """
    N = block_points.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    if block_labels is not None:
        return block_points[idx], block_labels[idx]
    else:
        return block_points[idx]

def pointnet_test(pointnet, dataloader):
    """
    Evaluates the accuracy of a trained PointNet model on a test dataset.

    This function sets the model to evaluation mode and iterates through the
    provided dataloader without tracking gradients. For each batch, it performs
    inference using the PointNet model, compares predictions to ground-truth labels,
    and calculates the overall classification accuracy.

    Assumes the model returns a tuple where the first element is the output tensor.

    Parameters:
        pointnet (torch.nn.Module): The trained PointNet model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing test data batches.
    Returns:
        float: The classification accuracy as a percentage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test
    pointnet.eval()
    correct = total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs, __, __ = pointnet(inputs.transpose(1, 2))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print("Accuracy: ", acc)
    print("correct", correct, "/", total)
    return acc

def predict(model, pcd, device):
    """
    Performs inference on a single point cloud using a trained model.

    This function takes a 3D point cloud (Open3D format), normalizes it by centering
    and scaling, and runs it through the given model to obtain class predictions.

    Parameters:
        model (torch.nn.Module): The trained model used for inference.
        pcd (open3d.geometry.PointCloud): The point cloud to classify.
        device (torch.device): The device (CPU or GPU) to run the model on.
    Returns:
        np.ndarray: The predicted class label(s) as a NumPy array.
    """
    points = np.asarray(pcd.points)
    points_norm = points - np.mean(points, axis=0)
    points_norm /= np.max(np.linalg.norm(points_norm, axis=1))
    points_tensor = torch.from_numpy(points_norm).float().to(device)
    points_tensor = points_tensor.T.unsqueeze(0)

    model.eval()
    # Now you can run inference
    with torch.no_grad():
        outputs, __, __ = model(points_tensor)
    _, predicted = torch.max(outputs.data, 1)

    return predicted.squeeze().cpu().numpy()

def create_fixed_size_grid_xy(min_bound, max_bound, block_size=2.0, color=[0.6, 0.6, 0.6]):
    """
    Creates a grid of 2D blocks in the XY plane, each extended along the full Z-axis.

    :param min_bound: Minimum XYZ coordinates as a list or array-like (e.g., [x_min, y_min, z_min]).
    :param max_bound: Maximum XYZ coordinates as a list or array-like (e.g., [x_max, y_max, z_max]).
    :param block_size: Size of each grid block in the XY plane (default is 2.0).
    :param color: RGB color used to paint each grid block (default is [0.6, 0.6, 0.6]).
    :return: List of Open3D LineSet objects representing the grid blocks.
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
        grid_lines = create_fixed_size_grid_xy(min_bound, max_bound, block_size=2.0, color=[0,0,0])
        for g in grid_lines:
            vis.add_geometry(g)

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