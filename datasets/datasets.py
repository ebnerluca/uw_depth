from depth_estimation.utils.data import (
    InputTargetDataset,
    IntPILToTensor,
    FloatPILToTensor,
    RandomFactor,
    ReplaceInvalid,
    MutualRandomHorizontalFlip,
    # MutualRandomVerticalFlip,
)
from torchvision import transforms


def get_flsea_dataset(device="cpu", split="", train=False, use_csv_samples=False):

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

    # shuffle for training
    shuffle = True  # if train else False

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
                # RandomFactor(factor_range=(0.75, 1.25)),  # randomly scale depth
                ReplaceInvalid(value="max", return_mask=True),
            ]
        )
        mutual_horizontal_flip = MutualRandomHorizontalFlip()
        both_transform = transforms.Compose(
            [
                mutual_horizontal_flip,
            ]
        )
        samples_flip_hooks = (mutual_horizontal_flip, None)

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
                ReplaceInvalid(value="max", return_mask=True),
            ]
        )
        samples_flip_hooks = (None, None)
        both_transform = None

    # instantiate dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=path_tuples_csv_files,
        shuffle=shuffle,
        input_transform=input_transform,
        target_transform=target_transform,
        both_transform=both_transform,
        use_csv_samples=use_csv_samples,
        samples_flip_hooks=samples_flip_hooks,
        max_samples=200,
    )

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
                ReplaceInvalid(value="max", return_mask=True),
            ]
        )
        mutual_horizontal_flip = MutualRandomHorizontalFlip()
        both_transform = transforms.Compose(
            [
                mutual_horizontal_flip,
            ]
        )
        samples_flip_hooks = (mutual_horizontal_flip, None)

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
                ReplaceInvalid(value="max", return_mask=True),
            ]
        )
        both_transform = None
        samples_flip_hooks = (None, None)

    # instantiate dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=path_tuples_csv_files,
        shuffle=shuffle,
        input_transform=input_transform,
        target_transform=target_transform,
        both_transform=both_transform,
        use_csv_samples=use_csv_samples,
        samples_flip_hooks=samples_flip_hooks,
        max_samples=200,
    )

    return dataset
