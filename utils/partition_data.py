import numpy as np
import os
import pandas as pd
from glob import glob

def split_into_blocks(points, block_size=2.0, stride=1.0, dim1=0, dim2=1):
    """
      Splits a point cloud into overlapping 2D blocks in the XY-plane.
      Each block is a square of size `block_size`, and the window moves across
      the XY-plane using the given `stride`. Only blocks containing at least one point are returned.
      :param points: (np.ndarray) Input point cloud of shape (N, D), where N is the number of points and D is the dimensionality (at least 3 for XYZ).
      :param block_size: (float) Size of each square block in the XY-plane (default is 2.0).
      :param stride: (float) Stride used to slide the block window (default is 1.0).
      :param dim1 (int): Fist dimension top split on
      :param dim2 (int): Second dimension top split on
      :return:
          blocks: (list of np.ndarray) A list of arrays, each containing the points in a block.
      """

    if stride <= 0:
        raise ValueError("Stride must be greater than 0.")

    xyz = points[:, :3]
    blocks = []

    x_min, y_min = np.min(xyz[:, dim1]), np.min(xyz[:, dim2])
    x_max, y_max = np.max(xyz[:, dim1]), np.max(xyz[:, dim2])

    x_range = np.arange(x_min, x_max + 1e-3, stride)
    y_range = np.arange(y_min, y_max + 1e-3, stride)

    for x in x_range:
        for y in y_range:
            x_cond = (xyz[:, dim1] >= x) & (xyz[:, dim1] < x + block_size)
            y_cond = (xyz[:, dim2] >= y) & (xyz[:, dim2] < y + block_size)
            mask = x_cond & y_cond
            block_points = points[mask]

            if block_points.shape[0] > 0:
                blocks.append(block_points)

    return blocks

def process_room_file(file_path, output_root, block_size=2.0, stride=1.0):
    """
    Processes a point cloud file by splitting it into spatial blocks and saving each block as an HDF5 file.
    The function reads a room-level point cloud from an HDF5 file, splits it into blocks using XY-plane slicing,
    and saves each block to a separate HDF5 file in a structured directory layout: output_root/area_name/room_name/.
    :param file_path: (str) Path to the input HDF5 file containing the room point cloud data.
    :param output_root: (str) Root directory where processed block files will be saved.
    :param block_size: (float, optional) Size of each square block in the XY-plane (default is 2.0).
    :param stride: (float, optional) Stride to move the block window during splitting (default is 1.0).
    :return:
        None
    """
    data = pd.read_hdf(file_path).to_numpy()

    room_name = os.path.splitext(os.path.basename(file_path))[0]
    area_name = os.path.basename(os.path.dirname(file_path))
    output_dir = os.path.join(output_root, area_name, room_name)
    os.makedirs(output_dir, exist_ok=True)

    blocks = split_into_blocks(data, block_size=block_size, stride=stride)

    for i, block in enumerate(blocks):
        save_path = os.path.join(output_dir, f'_partition{i}_.hdf5')
        pd.DataFrame(block).to_hdf(save_path, key='space_slice')

    print(f"{len(blocks)} blocks saved to {output_dir}")

def partition_data(src, output_src, block_size=2, stride=1):
    """
    Partitions raw 3D point cloud data into overlapping blocks using a sliding window and save it as .hdf5 files.

    :param src: Root directory containing subfolders 'Area_1' to 'Area_6' with room HDF5 files.
    :param output_src: Destination directory where the partitioned blocks will be saved.
    :param block_size: Size (in meters) of each square block in the XY plane (default is 2).
    :param stride: Step size (in meters) for sliding the window when partitioning (default is 1).
    :return: None
    """
    area_nums = [1,2,3,4,5,6]
    areas = list()
    for area in area_nums:
        areas.append(os.path.join(src, f'Area_{area}'))

    # check that datapaths are valid, if not raise error
    if len(areas) == 0:
        raise FileNotFoundError("NO VALID FILEPATHS FOUND!")

    for p in areas:
        if not os.path.exists(p):
            raise FileNotFoundError(f"PATH NOT VALID: {p} \n")

    rooms = []
    for area in areas:
        rooms += glob(os.path.join(area, '**\*.hdf5'),
                                recursive=True)
    for room in rooms:
        process_room_file(room, output_src, block_size=block_size, stride=stride)

    print(f'Dataset: {src} \n partitioned and saved into: {output_src}')


if __name__ == "__main__":

    """
    A file for divide the S3DIS in parts, for training the PointNet. 
    """
    src = 'S3DIS' # Should hold all the areas in sub folders
    output_src = 'S3DIS/partitioned'

    partition_data(src, output_src, block_size=2, stride=1)

