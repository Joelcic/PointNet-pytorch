from torch.utils.data import Dataset
import os
import torch
from glob import glob
import pandas as pd
from utils import sample_block

class PointCloudData(Dataset):
    def __init__(self, dataset_path, area_nums, transforms, num_points):
        """
          INPUT
              dataset_path: path to the dataset folder
              transform   : transform function to apply to point cloud
              start       : index of the first file that belongs to dataset
              end         : index of the first file that do not belong to dataset
        """
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.num_points = num_points

        # Setup the data
        self.data_paths = []
        self._data_setup(dataset_path, area_nums)

    def __len__(self):
        return len(self.data_paths)

    def _data_setup(self,root, area_nums):
        # Gets paths to all Area folders
        areas = list()
        for area in area_nums:
            areas.append(os.path.join(root, f'Area_{area}'))

        # check that datapaths are valid, if not raise error
        if len(areas) == 0:
            raise FileNotFoundError("NO VALID FILEPATHS FOUND!")

        for p in areas:
            if not os.path.exists(p):
                raise FileNotFoundError(f"PATH NOT VALID: {p} \n")

        # get all datapaths to the .hdf5 files
        for area in areas:
            self.data_paths += glob(os.path.join(area, '**\*.hdf5'),
                                    recursive=True)

        if len(self.data_paths) == 0:
            raise FileNotFoundError("No .hdf5 files found under the provided root path!")

    def __getitem__(self, idx):
        # read data from hdf5
        data = pd.read_hdf(self.data_paths[idx]).to_numpy()
        points = data[:, :3]  # xyz points
        targets = data[:, 3]  # integer categories

        # Takes out 4096 random points from the sample
        points, targets = sample_block(points, targets, num_points=self.num_points)

        # Perform data transformation
        if self.transforms:
            points = self.transforms(points)

        # convert to torch
        torch_points = torch.from_numpy(points)
        torch_labels = torch.from_numpy(targets)
        return torch_points, torch_labels