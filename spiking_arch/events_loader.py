import os
import tarfile

import matplotlib
import matplotlib.animation as animation 


import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


import tonic
import tonic.transforms as transforms

from IPython.display import HTML



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


    sensor_size = (28, 28, 2) # tonic.datasets.DVSGesture.sensor_size  OR (128, 128, 2) 
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=20)
    denoise_transform = tonic.transforms.Denoise(filter_time=10000)
    centercrop = tonic.transforms.CenterCrop(sensor_size=sensor_size, size=(18,18))#(100, 100))

    transform = transforms.Compose([denoise_transform, centercrop, frame_transform]) 

    train_dataset = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)
    test_dataset = tonic.datasets.DVSGesture(save_to=root, train=False, transform=transform)

    valid_size = int(len(train_dataset) * (valid_perc / 100.0))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    # print([type(data[0]) for data in train_dataset])
    
    train_loader = DataLoader(train_dataset, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs_test, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs_test, shuffle=False)
    return train_loader, valid_loader, test_loader

## visualization for checking
    # train_frame, _ = train_dataset[0]
    # # valid_frame, _ = valid_dataset[0]
    # test_frame, _ = test_dataset[0]
    # # train_frame = train_frame / train_frame.max()  # normalize
    # animation = tonic.utils.plot_animation(train_frame)
    # writervideo = animation.FFMpegWriter(fps=60) 
    # ani.save('dvs_visual.mp4', writer=writervideo) 
    # display(HTML(ani.to_html5_video()))
    # animation.show()

    

