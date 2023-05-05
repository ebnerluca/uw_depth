import torch

# from torch.distributions.normal import Normal
from torchvision.transforms.functional import hflip, vflip
import numpy as np

from PIL import Image
import csv

import random
from os.path import exists


class TrainDataset:
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
        target_img = Image.open(target_fn).resize((320, 240))

        # apply transforms
        input_img = self.input_transform(input_img)
        target_img = self.target_transform(target_img)
        if self.both_transform is not None:
            input_img, target_img = self.both_transform([input_img, target_img])

        return input_img, target_img

    def check_dataset(self):
        """Checks dataset for missing files."""
        for pair in self.pairs:
            if (not exists(pair[0])) or (not exists(pair[1])):
                print(f"Missing files! {pair} is missing.")
                return False
        return True


class InputTargetRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, input_target):

        do_flip = torch.rand(1) < self.p  # do flip or not

        if do_flip:
            input_target[0] = hflip(input_target[0])
            input_target[1] = hflip(input_target[1])

        return input_target


class InputTargetRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, input_target):

        do_flip = torch.rand(1) < self.p  # do flip or not

        if do_flip:
            input_target[0] = vflip(input_target[0])
            input_target[1] = vflip(input_target[1])

        return input_target


class Uint8PILToTensor:
    """Converts a uint8 PIL img in range [0,255] to a torch float tensor in range [0,1]."""

    def __call__(self, img):
        # convert to np array
        img_np = np.array(img)

        # enforce dimension order
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

    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, img):
        # convert to np array
        img_np = np.array(img)

        # normalize
        if self.normalize:
            min = img_np.min()
            max = img_np.max()
            img_np = (img_np - min) / (max - min)

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np)

        return img_tensor


def test_dataset(device="cpu"):

    print("Testing TrainDataset class ...")

    # test specific imports
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # define dataset
    dataset = TrainDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/test.csv",
        ],
        shuffle=True,
        input_transform=transforms.Compose([Uint8PILToTensor()]),
        target_transform=transforms.Compose([FloatPILToTensor(normalize=True)]),
        both_transform=transforms.Compose(
            [
                InputTargetRandomHorizontalFlip(),
                InputTargetRandomVerticalFlip(),
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
