from torchvision import transforms
from utils.transformations import *
from dataset import PointCloudData
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models import PointNetSeg
from utils.utils import *
from utils.transformations import *
import pandas as pd

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

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_path = 'S3DIS_partitioned'
    num_classes = 14
    print(dataset_path)

    train_areas=[1,2,3,4]
    test_areas=[5]
    val_areas=[6]

    """ TRANING  """
    # Training parameters
    num_epochs = 150
    batch_size = 16
    nr_point_train = 4096
    learning_rate = 0.005
    lr_step_size = 20
    lr_reduce = 0.5

    # Create dataset
    train_transforms = default_transforms(train=True)
    test_transforms = default_transforms(train=False)

    train_dataset = PointCloudData(dataset_path=dataset_path, area_nums=train_areas, transforms=train_transforms, num_points=nr_point_train)
    val_dataset = PointCloudData(dataset_path=dataset_path, area_nums=val_areas, transforms=test_transforms, num_points=nr_point_train)
    test_dataset = PointCloudData(dataset_path=dataset_path, area_nums=test_areas, transforms=test_transforms, num_points=nr_point_train)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)


    pointnet = PointNetSeg(classes=num_classes)
    # Initialize the optimizer
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_reduce)

    # Start the training
    train_loss, train_acc, val_acc = train(pointnet=pointnet, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, device=device, save=True, epochs=num_epochs, lr_scheduler=scheduler)

    # Save training results
    results_df = pd.DataFrame({
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    results_path = "result/training_metrics_acc_75.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Training metrics saved to {results_path}")