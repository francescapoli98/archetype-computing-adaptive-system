import os
import tarfile

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


import tonic
import tonic.transforms as transforms



    ################################### TENTATIVO ###################################

def get_data(
    root: os.PathLike, bs_train: int, bs_test: int, dataset, valid_perc: int = 10):
    ##labels
    sensor_size = (28, 28, 2)   #tonic.datasets.NMNIST.sensor_size = (34, 34, 2)
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20) #n_time_bins=3
    ##images
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    ## ADD CENTERCROP HERE
    centercrop = tonic.transforms.CenterCrop(sensor_size=sensor_size, size=(18, 18))

    transform = transforms.Compose([denoise_transform, centercrop, frame_transform])
    
    # full_dataset = tonic.datasets.DVSGesture(save_to=extract_dir, transform=transform)

    # Split into train/validation/test (DVSGesture does not have predefined splits)
    valid_size = int(len(dataset) * (valid_perc / 100.0))
    test_size = int(len(dataset) * 0.2)  # or however you want to split
    train_size = len(dataset) - valid_size - test_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)

    return train_loader, valid_loader, test_loader

    ######################################################################


def get_nmnist_data(
    root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10):
    """Get the MNIST dataset and return the train, validation and test dataloaders.

    Args:
        root (os.PathLike): Path to the folder containing the MNIST dataset.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        valid_perc (int): Percentage of the train dataset to use for
            validation. Defaults to 10.
    """
    ##labels
    sensor_size = (28, 28, 2)   #tonic.datasets.NMNIST.sensor_size = (34, 34, 2)
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20) #n_time_bins=3
    ##images
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    ## ADD CENTERCROP HERE
    centercrop = tonic.transforms.CenterCrop(sensor_size=sensor_size, size=(18, 18))

    transform = transforms.Compose([denoise_transform, centercrop, frame_transform])
    
    train_dataset = tonic.datasets.NMNIST(save_to=root, train=False, transform=transform)

    test_dataset = tonic.datasets.NMNIST(save_to=root, train=False, transform=transform)

    valid_size = int(len(train_dataset) * (valid_perc / 100.0))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs_test, shuffle=False)

    print(f"Total samples: {len(full_dataset)}")
    print(f"Train: {train_size}, Valid: {valid_size}, Test: {test_size}")

    return train_loader, valid_loader, test_loader

def get_dvs_data(
    root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10
):
    """Get the DVS Gesture dataset and return the train, validation, and test dataloaders.

    Args:
        root (os.PathLike): Path to the folder containing the .tar.gz and where to extract the dataset.
        bs_train (int): Batch size for the train dataloader.
        bs_test (int): Batch size for the validation and test dataloaders.
        valid_perc (int): Percentage of the train dataset to use for validation. Defaults to 10.
    """

    # Check if the dataset is extracted
    extract_dir = os.path.join(root, "DVSGesture")
    archive_path = os.path.join(root, "ibmGestureTest.tar.gz")

    if not os.path.exists(extract_dir):
        print(f"Extracting {archive_path} to {extract_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root)
        print("Extraction completed.")

    sensor_size = (28, 28, 2)#tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20)
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    centercrop = tonic.transforms.CenterCrop(sensor_size=sensor_size, size=(18,18))#(100, 100))

    transform = transforms.Compose([denoise_transform, centercrop, frame_transform])

    # full_dataset = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)

    # # Split into train/validation/test (DVSGesture does not have predefined splits)
    # valid_size = int(len(full_dataset) * (valid_perc / 100.0))
    # test_size = int(len(full_dataset) * 0.2)  # or however you want to split
    # train_size = len(full_dataset) - valid_size - test_size
    

    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #     full_dataset, [train_size, valid_size, test_size]
    # )

    # train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)



    train_dataset = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)

    test_dataset = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)

    valid_size = int(len(train_dataset) * (valid_perc / 100.0))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    

    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)


    return train_loader, valid_loader, test_loader

    




# def another_get_nmnist_data(
#     root: os.PathLike, bs_train: int, bs_test: int, valid_perc: int = 10
# ):
#     # Load dataset
#     dataset = tonic.datasets.NMNIST(save_to=root, train=True)

#     # Define the transformation: Convert events to dense tensors
#     sensor_size = dataset.sensor_size  # (34, 34, 2) - (Height, Width, Polarity)
#     time_window = 10  # Number of time steps (can be adjusted)

#     transform = transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)

#     # Get an example sample
#     events, label = dataset[0]
#     frames = transform(events)  # Shape: (time_window, 2, 34, 34)

#     # Convert to float and normalize (if needed)
#     frames = torch.tensor(frames, dtype=torch.float32)

#     # Flatten spatial dimensions to match SNN input expectations
#     flattened_frames = frames.view(time_window, -1)  # Shape: (time_window, 34*34*2)
