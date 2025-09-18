from dataset import PointCloudData
from torch.utils.data import DataLoader
from models.models import PointNetSeg
from utils.transformations import *
from utils.utils import *

def test_pointnet(dataset_path, model_path, num_classes = 14, nr_point_train = 4096, test_areas = [5]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pointnet_model = PointNetSeg(classes=num_classes)
    test_transforms = default_transforms(train=False)
    test_dataset = PointCloudData(dataset_path=dataset_path, area_nums=test_areas, transforms=test_transforms,
                                  num_points=nr_point_train)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)
    # Load the models weights
    pointnet_model.load_state_dict(torch.load(model_path))
    pointnet_model.to(device)  # Make sure it's on the right device (CPU/GPU)"""

    accuracy = pointnet_test(pointnet=pointnet_model, dataloader=test_loader)

    return accuracy


if __name__ == "__main__":
    """
    A file for testing the trained PointNet on the S3DIS dataset pre partitioned according to "partition_data.py"
    """

    dataset_path = "..."

    """ To validate the PointNet on test set"""
    model_path = "..."
    accuracy = test_pointnet(dataset_path, model_path, num_classes = 14, nr_point_train = 4096, test_areas = [5])
