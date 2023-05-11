import torch

# from torch.distributions.normal import Normal
from torchvision.transforms.functional import hflip, vflip
import numpy as np

from PIL import Image
import csv

import random
from os.path import exists


class InputTargetDataset:
    """Parameters:
    - pairs_csv_files: List of filepaths to csv files which list image pairs (input and target)
    - input_transform: Transform to apply to the input RGB image, needs to return torch Tensor
    - target_transform: Transform to apply to the target depth image, needs to return torch Tensor and corresponding mask
    - both_transform: Transform to apply to both input and target image, needs to return torch Tensor"""

    def __init__(
        self,
        pairs_csv_files,
        input_transform,
        target_transform,
        shuffle=False,
        both_transform=None,
    ) -> None:

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.both_transform = both_transform

        # load pairs
        self.pairs = []
        for csv_file in pairs_csv_files:
            self.pairs += [i for i in csv.reader(open(csv_file).read().splitlines())]

        # random shuffle pairs
        if shuffle:
            random.shuffle(self.pairs)

        # checking dataset for missing files
        if not self.check_dataset():
            print("WARNING, corrupted dataset (missing files)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        # get filenames
        input_fn = self.pairs[idx][0]
        target_fn = self.pairs[idx][1]

        # read imgs
        input_img = Image.open(input_fn).resize((640, 480))
        target_img = Image.open(target_fn).resize((320, 240), resample=Image.NEAREST)

        # apply transforms
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)

        if (input_img is None) or (target_img is None) or (mask is None):
            print(f"Loading img pair failed, received None. paths:")
            print(input_fn)
            print(target_fn)
            print("Trying other image as substitution ...")
            random_idx = np.random.randint(0, len(self))
            input_img, target_img, mask = self[random_idx]

        if self.both_transform is not None:
            input_img, target_img, mask = self.both_transform(
                [input_img, target_img, mask]
            )

        return input_img, target_img, mask

    def check_dataset(self):
        """Checks dataset for missing files."""
        for pair in self.pairs:
            if (not exists(pair[0])) or (not exists(pair[1])):
                print(f"Missing files! {pair} is missing.")
                return False
        return True


class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p  # do flip or not

        if do_flip:
            for i in range(len(tensors)):
                tensors[i] = hflip(tensors[i])

        return tensors


class MutualRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p  # do flip or not

        if do_flip:
            for i in range(len(tensors)):
                tensors[i] = vflip(tensors[i])

        return tensors


class Uint8PILToTensor:
    """Converts a uint8 PIL img in range [0,255] to a torch float tensor in range [0,1]."""

    def __call__(self, img):
        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: ch x H x W
        img_np = img_np.transpose((2, 0, 1))

        # convert to tensor
        img_tensor = torch.from_numpy(img_np)

        # convert to float and divide by 255
        img_tensor = img_tensor.float().div(255)

        return img_tensor


class FloatPILToTensor:
    """Converts a float PIL img to a torch tensor, normalized if specified.

    Specify zero_add value to add to pixels which would otherwise be exactly zero,
    this can avoid issues e.g. when using the log() function since log(0) is undefined."""

    def __init__(self, normalize=False, invalid_value=0.0):
        self.normalize = normalize
        self.invalid_value = invalid_value

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # mask valid values, depth <= 0 means invalid
        mask = img_np > 0.0

        # normalize mask, set invalid to given value
        if self.normalize:
            try:
                min = img_np[mask].min()
                max = img_np[mask].max()
                img_np = (img_np - min) / (max - min)
                img_np[~mask] = self.invalid_value
            except ValueError:
                # for very odd frames the whole ground truth is invalid
                # the whole mask is False, leading to empty arrays
                # if this occurs, return None
                print(
                    "Error, img has invalid value range: "
                    + f"[{img_np.min()}, {img_np.max()}]\n"
                    + "Returning None."
                )
                return None, None

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np)
        mask = torch.from_numpy(mask)

        return img_tensor, mask


def test_dataset(device="cpu"):

    print("Testing TrainDataset class ...")

    # test specific imports
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # define dataset
    dataset = InputTargetDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/test.csv",
        ],
        shuffle=True,
        input_transform=transforms.Compose([Uint8PILToTensor()]),
        target_transform=transforms.Compose([FloatPILToTensor(normalize=True)]),
        both_transform=transforms.Compose(
            [
                MutualRandomHorizontalFlip(),
                MutualRandomVerticalFlip(),
            ]
        ),
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=4)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]

        for i in range(rgb_imgs.size(0)):

            rgb_img = rgb_imgs[i, ...]
            d_img = d_imgs[i, ...]
            plt.figure(f"rgb img {i}")
            plt.imshow(rgb_img.permute(1, 2, 0))
            plt.figure(f"d img {i}")
            plt.imshow(d_img.permute(1, 2, 0))

            # print(f"prior map {i} range: [{prior_map.min()}, {prior_map.max()}]")
            # print(f"signal map {i} range: [{signal_map.min()}, {signal_map.max()}]")
        plt.show()

        break  # only check first batch

    print("Testing DataSet class done.")


# run as "python -m depth_estimation.utils.data"
if __name__ == "__main__":
    test_dataset(device="cpu")
