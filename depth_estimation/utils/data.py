import torch
import numpy as np

from PIL import Image
import csv
import random
from os.path import exists

from .utils import get_distance_map


class TrainDataset:
    def __init__(
        self,
        pairs_csv_files,
        shuffle=False,
        input_transform=None,
        target_transform=None,
    ) -> None:

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform

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
        if self.input_transform is not None:
            input_img = self.input_transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        return input_img, target_img

    def check_dataset(self):
        """Checks dataset for missing files."""
        for pair in self.pairs:
            if (not exists(pair[0])) or (not exists(pair[1])):
                print(f"Missing files! {pair} is missing.")
                return False
        return True


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

    def __init__(self, normalize=False, zero_add=0.0):
        self.normalize = normalize
        self.zero_add = zero_add

    def __call__(self, img):
        # convert to np array
        img_np = np.array(img)

        # normalize
        if self.normalize:
            min = img_np.min()
            max = img_np.max()
            range = max - min
            img_np = (img_np - min) / range

        # add zero_add value to pixels which would be zero otherwise
        if self.zero_add != 0.0:
            img_np[img_np == 0.0] = self.zero_add

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np)

        return img_tensor
