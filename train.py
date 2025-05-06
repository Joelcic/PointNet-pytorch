from dataset import PointCloudData
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.models import PointNetSeg
from utils.transformations import *
from utils.utils import *
import pandas as pd


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


if __name__ == "__main__":
    """
    A file for training the PointNet on the S3DIS dataset pre partitioned according to "partition_data.py"
    """

    dataset_path = r'...\S3DIS_partitioned'

    """To train the PointNet"""
    train_pointnet(dataset_path, num_classes=14, num_epochs=150, batch_size=16, nr_point_train=4096,
                   learning_rate=0.005, lr_step_size=20, lr_reduce=0.5)

