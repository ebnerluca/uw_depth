# this is a helper script that puts together datasets so they can be loaded
# with simpler getter functions for less repeated code in train/testing scripts

import csv

from depth_estimation.utils.data import (
    InputTargetDataset,
    IntPILToTensor,
    FloatPILToTensor,
    MutualRandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms


def get_example_dataset(train=False, shuffle=False, device="cpu"):

    # filenames
    index_file = "data/example_dataset/dataset.csv"
    lines = csv.reader(open(index_file).read().splitlines())
    rgb_depth_priors_tuples = [i for i in lines]

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),  # load uint8 img as tensor
                transforms.ColorJitter(brightness=0.1, hue=0.05),  # color jitter
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),  # load float img as pytorch tensor
                ReplaceInvalid(value="max"),  # replace invalid depth values (<= 0)
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])  # hflip
        target_samples_transform = transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.8, 1.2)),  # random depth scaling
            ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [IntPILToTensor(type="uint8", device=device)]  # load uint8 img as tensor
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),  # load float img as pytorch tensor
                ReplaceInvalid(value="max"),  # replace invalid depth values (<= 0)
            ]
        )
        all_transform = None
        target_samples_transform = None

    # instantiate dataset
    dataset = InputTargetDataset(
        rgb_depth_priors_tuples=rgb_depth_priors_tuples,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        target_samples_transform=target_samples_transform,
        max_priors=200,
        shuffle=shuffle,
    )

    return dataset
