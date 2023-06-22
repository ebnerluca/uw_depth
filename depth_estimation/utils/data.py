import torch

# from torch.distributions.normal import Normal
from torchvision.transforms.functional import hflip, vflip
import numpy as np

from PIL import Image
import csv
import pandas as pd

import random
from os.path import exists

from .depth_prior import get_depth_prior_from_features


class InputTargetDataset:
    """Parameters:
    - path_tuples_csv_files: List of filepaths to csv files which list image tuples (input, target (,features))
    - input_transform: Transform to apply to the input RGB image, returns torch Tensor
    - target_transform: Transform to apply to the target depth image, returns torch Tensor
    - all_transform: Transform to apply to both input, target and mask image (and depth samples as well), returns list of torch Tensors
    - target_samples_transform: Transfrom to apply to both target and depth samples, returns list of torch Tensors
    - use_csv_samples: Bool to specify if depth samples should be drawn randomly or from saved csv file
    - max_samples: max number of samples per image"""

    def __init__(
        self,
        path_tuples_csv_files,
        input_transform,
        target_transform,
        all_transform=None,
        shuffle=False,
        use_csv_samples=False,
        target_samples_transform=None,
        max_samples=200,
    ) -> None:

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.all_transform = all_transform

        # load tuples
        self.path_tuples = []
        for csv_file in path_tuples_csv_files:
            try:
                lines = csv.reader(open(csv_file).read().splitlines())
                self.path_tuples += [i for i in lines]
            except FileNotFoundError:
                print(f"{csv_file} not found, skipping...")

        # random shuffle tuples
        if shuffle:
            random.shuffle(self.path_tuples)

        # depth_samples
        self.use_csv_samples = use_csv_samples
        self.target_samples_transform = target_samples_transform
        self.max_samples = max_samples

        # checking dataset for missing files
        if not self.check_dataset():
            print("ERROR, corrupted dataset (missing files). Triggering exit(1).")
            exit(1)

        print(f"Dataset with {len(self)} pairs.")

    def __len__(self):
        return len(self.path_tuples)

    def __getitem__(self, idx):

        # get filenames
        input_fn = self.path_tuples[idx][0]
        target_fn = self.path_tuples[idx][1]

        # read imgs
        input_img = Image.open(input_fn).resize((640, 480))
        target_img = Image.open(target_fn).resize((320, 240), resample=Image.NEAREST)

        # apply input/target transforms
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)

        if not mask.any():
            print(
                f"File {target_fn} has no valid depth values, trying other image as substitution ..."
            )
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion

        # read features
        if self.use_csv_samples:

            # read depth samples file
            depth_samples_fn = self.path_tuples[idx][2]
            depth_samples = self.read_features(
                depth_samples_fn, device=target_img.device
            )

            # check if zero valid depths
            if depth_samples is None:
                print("Depth samples is None, trying other image as substitution ...")
                random_idx = np.random.randint(0, len(self))
                return self[random_idx]  # recursion

            # get parametrization imgs 2x240x320
            parametrization = get_depth_prior_from_features(
                features=depth_samples.unsqueeze(0),  # add batch dimension
                height=240,
                width=320,
            ).squeeze(
                0
            )  # remove batch dimension

            # apply transform to target and samples
            if self.target_samples_transform is not None:
                target_img, parametrization = self.target_samples_transform(
                    [target_img, parametrization]
                )

        # list of all output tensors
        tensor_list = [input_img, target_img, mask]
        if self.use_csv_samples:
            tensor_list.append(parametrization)

        # apply mutual transforms
        if self.all_transform is not None:

            tensor_list = self.all_transform(tensor_list)

        return tensor_list

    def check_dataset(self):
        """Checks dataset for missing files."""
        for tuple in self.path_tuples:
            if (not exists(tuple[0])) or (not exists(tuple[1])):
                print(f"Missing files! {tuple} is missing.")
                return False

            # check depth samples
            if self.use_csv_samples:
                try:
                    if not exists(tuple[2]):
                        print(f"Missing files! {tuple} is missing.")
                        return False
                except IndexError:
                    print(
                        f"Specified use_csv_samples, but path_tuples csv file doesnt list any! tuple: {tuple}"
                    )
                    return False

        return True

    def read_features(self, path, device="cpu"):

        # load samples (might be less than n_samples)
        depth_samples_data = pd.read_csv(path).to_numpy()  # [: self.max_samples]

        # give warning when no features
        if len(depth_samples_data) == 0:
            print(f"WARNING: Features list {path} is empty, returning None!")
            return None
        else:
            rand_idcs = np.random.permutation(len(depth_samples_data))[
                : self.max_samples
            ]
            depth_samples = depth_samples_data[rand_idcs]

        # tensor from numpy
        depth_samples = torch.from_numpy(depth_samples).to(device)

        return depth_samples


class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, tensors):

        do_flip = torch.rand(1) < self.p

        # flip
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
        do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            for i in range(len(tensors)):

                tensors[i] = vflip(tensors[i])

        return tensors


class IntPILToTensor:
    """Converts a uint8 PIL img in range [0,255] to a torch float tensor in range [0,1]."""

    def __init__(self, type="uint8", device="cpu") -> None:
        self.device = device
        if type == "uint8":
            self.divider = 255
        elif type == "uint16":
            self.divider = 65535
        else:
            self.divider = 1

    def __call__(self, img):
        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: ch x H x W
        if img_np.ndim == 3:
            img_np = img_np.transpose((2, 0, 1))
        elif img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        # convert to float and divide by 255
        img_tensor = img_tensor.float().div(self.divider)

        return img_tensor


class FloatPILToTensor:
    """Converts a float PIL img to a 1xHxW torch tensor"""

    def __init__(self, device="cpu"):

        self.device = device

    def __call__(self, img):

        # convert to np array
        img_np = np.array(img)

        # enforce dimension order: channels x height x width
        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)

        return img_tensor  # , mask


class MutualRandomFactor:
    def __init__(self, factor_range=(0.75, 1.25)) -> None:
        self.factor_range = factor_range

    def __call__(self, tensors):

        factor = (
            torch.rand(1).item() * (self.factor_range[1] - self.factor_range[0])
            + self.factor_range[0]
        )

        for i in range(len(tensors)):

            tensors[i][0, ...] *= factor

        return tensors


class ReplaceInvalid:
    def __init__(self, value=None):
        self.value = value

    def __call__(self, tensor):

        mask = get_mask(tensor)

        # if mask is empty, return None
        if not mask.any():
            print(
                "Mask is empty, meaning all depth values invalid. Returning unchanged."
            )

            return tensor, mask

        # change value of non valid pixels
        if self.value is not None:
            if self.value == "max":
                max = tensor[mask].max()
                tensor[~mask] = max
            elif self.value == "min":
                min = tensor[mask].min()
                tensor[~mask] = min
            else:
                tensor[~mask] = self.value

        return tensor, mask


def get_mask(depth):

    mask = depth.gt(0.0)

    return mask


def test_dataset(device="cpu"):

    print("Testing TrainDataset class ...")

    # test specific imports
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # define dataset
    dataset = InputTargetDataset(
        path_tuples_csv_files=[
            "/home/auv/FLSea/archive/canyons/flatiron/flatiron/imgs/dataset_with_features.csv",
        ],
        shuffle=True,
        input_transform=transforms.Compose(
            [IntPILToTensor(type="uint8", device=device)]
        ),
        target_transform=transforms.Compose(
            [
                FloatPILToTensor(device=device),
                ReplaceInvalid(value="max"),
            ]
        ),
        all_transform=transforms.Compose(
            [
                MutualRandomHorizontalFlip(),
                MutualRandomVerticalFlip(),
            ]
        ),
        use_csv_samples=True,
        target_samples_transform=transforms.Compose(
            [
                MutualRandomFactor(factor_range=(0.25, 1.75)),
            ]
        ),
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=4)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]
        masks = data[2]
        depth_samples = data[3]

        for i in range(rgb_imgs.size(0)):

            rgb_img = rgb_imgs[i, ...]
            d_img = d_imgs[i, ...]
            mask = masks[i, ...]
            d_samples = depth_samples[i, ...]
            print(f"d range: [{d_img.min()}, {d_img.max()}]")
            plt.figure(f"rgb img {i}")
            plt.imshow(rgb_img.permute(1, 2, 0))
            plt.figure(f"d img {i}")
            plt.imshow(d_img.permute(1, 2, 0))
            plt.figure(f"mask {i}")
            plt.imshow(mask.permute(1, 2, 0))
            plt.figure(f"depth with features {i}")
            plt.imshow(d_img.permute(1, 2, 0))
            plt.scatter(x=d_samples[:, 1], y=d_samples[:, 0])

            d_img_values = d_img[
                :, d_samples[:, 0].round().long(), d_samples[:, 1].round().long()
            ].squeeze()[:5]
            d_samples_values = d_samples[:, 2][:5]

            print(f"depth values at feature location: {d_img_values}")
            print(f"depth values of features: {d_samples_values}")

        plt.show()

        break  # only check first batch

    print("Testing DataSet class done.")


# run as "python -m depth_estimation.utils.data"
if __name__ == "__main__":
    test_dataset(device="cpu")
