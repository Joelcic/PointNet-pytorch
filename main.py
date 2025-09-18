import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
import seaborn as sns

from models import PointNetSeg
from dataset import PointCloudData
from utils.transformations import default_transforms
from utils.partition_data import split_into_blocks
from visualization import simple_visualization
from utils.utils import *
from utils.visualization import *


def predict_sample(model, pcd, device):
    """
    Performs inference on a single sample of point cloud using a trained model.

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

def predict_pcd(model, pcd, device, color_map=None):
    """
    Performs inference on a single point cloud using a trained model.

    This function takes a 3D point cloud (Open3D format), normalizes it by centering
    and scaling, and runs it through the given model to obtain class predictions.
    """

    points = np.asarray(pcd.points)
    blocks = split_into_blocks(points=points, block_size=2.0, stride=2.0)
    print(f'Num blocks: {len(blocks)}')

    if color_map == None:
        color_map = {
            0: [1.0, 0.0, 0.0],  # Red
            1: [0.0, 1.0, 0.0],  # Green
            2: [0.0, 0.0, 1.0],  # Blue
            #13: [0.0, 0.0, 0.0],  # Black
        }
    default_color = [0.6, 0.6, 0.6]  # Gray
    pcd_all = o3d.geometry.PointCloud()
    for block in blocks:
        sampled_block = sample_block(block)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_block)
        predicted_labels = predict_sample(model=model, pcd=pcd, device=device)
        # Color the points
        colors = np.array([color_map.get(label, default_color) for label in predicted_labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_all += pcd
    
    return pcd_all



if __name__ == "__main__":
    class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table',
                   'chair', 'sofa', 'bookcase', 'board', 'stairs', 'clutter']
    color_map = {
    0:  [0.95, 0.95, 0.95],   # ceiling - light gray
    1:  [0.55, 0.85, 0.35],   # floor - greenish
    2:  [0.35, 0.35, 0.85],   # wall - blue
    3:  [0.85, 0.45, 0.15],   # beam - brownish orange
    4:  [0.65, 0.35, 0.15],   # column - darker brown
    5:  [0.25, 0.65, 0.85],   # window - cyan
    6:  [0.85, 0.35, 0.35],   # door - red
    7:  [0.85, 0.65, 0.15],   # table - yellow-orange
    8:  [0.35, 0.85, 0.35],   # chair - green
    9:  [0.55, 0.35, 0.85],   # sofa - purple
    10: [0.85, 0.55, 0.85],   # bookcase - pink
    11: [0.25, 0.25, 0.25],   # board - dark gray
    12: [0.15, 0.55, 0.55],   # stairs - teal
    13: [0.55, 0.55, 0.55],   # clutter - medium gray
}
    # build a color map dict from tab20
    tab20 = plt.get_cmap("tab20")
    color_map = {i: tab20(i)[:3] for i in range(14)}  # first 14 colors


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = len(class_names)
    pointnet = PointNetSeg(classes=num_classes)

    model_path = "pointnetmodel_weights_acc_74.pth"
    pointnet.load_state_dict(torch.load(model_path))
    pointnet.to(device)

    """ Load and plot the results"""
    #accuracy = pointnet_test(pointnet=pointnet, dataloader=test_loader)
    #plot_training("training_metrics_acc_75.csv")
    #plot_train_loss("training_metrics_acc_75.csv")
    #save_metrics_for_analys(pointnet=pointnet, data_loader=test_loader, class_names=class_names)


    file_name = "S3DIS/conferenceRoom_1.hdf5"
    pcd, labels = read_hdf5(file_name)
    simple_visualization(pcd, labels)


    pcd_predicted = predict_pcd(model=pointnet, pcd=pcd, device=device, color_map=color_map)

    visualization(pcd_predicted, add_grid=True)
    
    
    blocks = split_into_blocks(np.asarray(pcd.points), block_size=2.0, stride=2.0)
    block = sample_block(blocks[1])
    pcd_block = o3d.geometry.PointCloud()
    pcd_block.points = o3d.utility.Vector3dVector(block)

    predicted_labels = predict_sample(model=pointnet, pcd=pcd_block, device=device)
    simple_visualization(pcd_block, predicted_labels)


