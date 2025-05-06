from torchvision import transforms
from transformations import *
from dataset import PointCloudData
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import PointNetSeg
from utils import *
import pandas as pd

def default_transforms(train=True):
    """
    Returns a composed set of transformations to be applied to point cloud data.
    If `train` is True, the returned transform includes data augmentation steps
    such as random rotation around the Z-axis and Gaussian noise addition.
    If False, only normalization is applied.
    :param train: (bool) Whether the transformations are for training (default is True).
    :return:
        transform: (torchvision.transforms.Compose) A composition of point cloud transforms.
    """
    if train:
        return transforms.Compose([
            Normalize(),
            RandomRotateZ(),
            GaussianNoice(),
        ])
    else:
        return transforms.Compose([
            Normalize(),
        ])

def train_pointnet(dataset_path, num_classes = 14, num_epochs=150, batch_size=16, nr_point_train=4096,
                   learning_rate=0.005, lr_step_size=20, lr_reduce = 0.5):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_areas = [1, 2, 3, 4]
    val_areas = [6]

    train_transforms = default_transforms(train=True)
    test_transforms = default_transforms(train=False)

    train_dataset = PointCloudData(dataset_path=dataset_path, area_nums=train_areas, transforms=train_transforms,
                                   num_points=nr_point_train)
    val_dataset = PointCloudData(dataset_path=dataset_path, area_nums=val_areas, transforms=test_transforms,
                                 num_points=nr_point_train)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    pointnet_model = PointNetSeg(classes=num_classes)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(pointnet_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_reduce)

    # Commence the training
    train_loss, train_acc, val_acc = train(pointnet=pointnet_model, optimizer=optimizer, train_loader=train_loader,
                                           val_loader=val_loader, device=device, save=True, epochs=num_epochs,
                                           lr_scheduler=scheduler)
    # Save training results
    results_df = pd.DataFrame({
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    results_path = "result/training_metrics_acc_75.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Training metrics saved to {results_path}")

def test_pointnet(dataset_path, model_path, num_classes = 14, nr_point_train = 4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_areas = [5]

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

    dataset_path = r'...\S3DIS_partitioned'

    """To train the PointNet
    train_pointnet(dataset_path, num_classes=14, num_epochs=150, batch_size=16, nr_point_train=4096,
                   learning_rate=0.005, lr_step_size=20, lr_reduce=0.5)
    """

    """ To validate the PointNet on test set
    model_path = "..."
    accuracy = test_pointnet(dataset_path, model_path, num_classes = 14, nr_point_train = 4096)
    """
