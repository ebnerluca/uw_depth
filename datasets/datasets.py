# this is a helper script that puts together datasets so they can be loaded
# with simpler getter functions for less repeated code in train/testing scripts

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


def get_flsea_dataset(
    device="cpu",
    split="dataset_with_features",
    train=False,
    use_csv_samples=False,
    shuffle=False,
):

    # define csv files for input target pairs
    path_tuples_csv_files = [
        f"/home/auv/FLSea/archive/canyons/flatiron/flatiron/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/canyons/horse_canyon/horse_canyon/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/canyons/tiny_canyon/tiny_canyon/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/canyons/u_canyon/u_canyon/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/big_dice_loop/big_dice_loop/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/coral_table_loop/coral_table_loop/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/cross_pyramid_loop/cross_pyramid_loop/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/dice_path/dice_path/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/landward_path/landward_path/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/pier_path/pier_path/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/sub_pier/sub_pier/imgs/{split}.csv",
        f"/home/auv/FLSea/archive/red_sea/northeast_path/northeast_path/imgs/{split}.csv",
    ]

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])
        target_samples_transform = transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.8, 1.2)),
            ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
            ]
        )
        target_transform = transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = None
        target_samples_transform = None

    # instantiate dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=path_tuples_csv_files,
        shuffle=shuffle,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        use_csv_samples=use_csv_samples,
        target_samples_transform=target_samples_transform,
        max_samples=200,
    )

    return dataset


def get_ycb_dataset(
    device="cpu",
    split="train",
    train=False,
    use_csv_samples=False,
    shuffle=False,
):

    # define csv files for input target pairs
    base_folder = "/media/auv/Seagate_2TB/datasets/ycb/ycb_dataset/ycb_dataset/"
    imgset_file = base_folder + f"image_sets/{split}.txt"
    lines = open(imgset_file).read().splitlines()
    path_tuples = []
    for line in lines:
        scene, img = line.split("/")
        rgb = base_folder + f"data/{scene}/{img}-color.png"
        depth = base_folder + f"data/{scene}/{img}-depth.png"
        features = (
            base_folder + f"data/{scene}/matched_features/{img}-color_features.csv"
        )
        path_tuples.append((rgb, depth, features))

    # placeholder
    path_tuples_csv_files = []

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint16", custom_divider=10000.0, device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = transforms.Compose([MutualRandomHorizontalFlip()])
        target_samples_transform = transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.8, 1.2)),
            ]
        )

    # if not train
    else:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
            ]
        )
        target_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint16", custom_divider=10000.0, device=device),
                ReplaceInvalid(value="max"),
            ]
        )
        all_transform = None
        target_samples_transform = None

    # instantiate dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=path_tuples_csv_files,
        shuffle=shuffle,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        use_csv_samples=use_csv_samples,
        target_samples_transform=target_samples_transform,
        max_samples=200,
    )

    # workaround for custom tuples list as ycb provides it
    dataset.path_tuples = path_tuples
    if shuffle:
        import random

        random.shuffle(dataset.path_tuples)
    if not dataset.check_dataset():
        exit(1)

    return dataset


def get_usod10k_dataset(device="cpu", split="", train=False, use_csv_samples=False):

    # define csv files for input target pairs
    path_tuples_csv_files = [
        f"/home/auv/FLSea/archive/canyons/flatiron/flatiron/imgs/{split}.csv",
    ]

    # shuffle for training
    shuffle = True if train else False

    # transforms
    if train:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
                transforms.ColorJitter(brightness=0.1, hue=0.05),
            ]
        )
        target_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint16", device=device),  # USOD10k
                ReplaceInvalid(value="max"),
            ]
        )
        mutual_horizontal_flip = MutualRandomHorizontalFlip()
        all_transform = transforms.Compose(
            [
                mutual_horizontal_flip,
            ]
        )
        target_samples_transform = None
    # if not train
    else:
        input_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint8", device=device),
            ]
        )
        target_transform = transforms.Compose(
            [
                IntPILToTensor(type="uint16", device=device),  # USOD10k
                ReplaceInvalid(value="max"),  # , return_mask=True),
            ]
        )
        all_transform = None
        target_samples_transform = None

    # instantiate dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=path_tuples_csv_files,
        shuffle=shuffle,
        input_transform=input_transform,
        target_transform=target_transform,
        all_transform=all_transform,
        use_csv_samples=use_csv_samples,
        target_samples_transform=target_samples_transform,
        max_samples=200,
    )

    return dataset
