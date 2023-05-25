import torch

# from torch.distributions.normal import Normal
from torchvision.transforms.functional import hflip, vflip
import numpy as np

from PIL import Image
import csv
import pandas as pd

import random
from os.path import exists


class InputTargetDataset:
    """Parameters:
    - path_tuples_csv_files: List of filepaths to csv files which list image tuples (input, target (,features))
    - input_transform: Transform to apply to the input RGB image, needs to return torch Tensor
    - target_transform: Transform to apply to the target depth image, needs to return torch Tensor and corresponding mask
    - both_transform: Transform to apply to both input and target image, needs to return torch Tensor
    - use_csv_samples: Bool to specify if depth samples should be drawn randomly or from saved csv file
    - samples_flip_hooks: Transform tuple hooks needed so samples can be flipped mutually with the rgb and depth img
    - max_samples: max number of samples per image"""

    def __init__(
        self,
        path_tuples_csv_files,
        input_transform,
        target_transform,
        shuffle=False,
        both_transform=None,
        use_csv_samples=False,
        samples_flip_hooks=(None, None),  # horizontal, vertical flip
        max_samples=200,
    ) -> None:

        # transforms applied to input and target
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.both_transform = both_transform

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
        self.samples_flip_hooks = samples_flip_hooks
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

        # apply transforms
        input_img = self.input_transform(input_img)
        target_img, mask = self.target_transform(target_img)

        if (input_img is None) or (target_img is None) or (mask is None):
            print(f"Loading img tuple failed, received None. paths:")
            print(input_fn)
            print(target_fn)
            print("Trying other image as substitution ...")
            random_idx = np.random.randint(0, len(self))
            return self[random_idx]  # recursion

        if self.both_transform is not None:
            input_img, target_img, mask = self.both_transform(
                [input_img, target_img, mask]
            )

        # read precomputed depth samples from csv
        if self.use_csv_samples:

            depth_samples_csv_file = self.path_tuples[idx][2]

            # need same dimensions for minibatch stacking, so we fix it
            depth_samples = np.zeros((self.max_samples, 3), dtype=np.float32)

            # load actual samples (might be less than n_samples)
            depth_samples_data = pd.read_csv(depth_samples_csv_file).to_numpy()[
                : self.max_samples
            ]

            # give warning when no features
            if len(depth_samples_data) == 0:
                print(
                    f"WARNING: Features list {self.path_tuples[idx][2]} is empty, placeholder parametrization will be used!"
                )

            # fill in data where available
            depth_samples[: len(depth_samples_data)] = depth_samples_data

            # tensor from numpy
            depth_samples = torch.from_numpy(depth_samples).to(input_img.device)

            # perform same flips as input and target for depth samples
            for hook in self.samples_flip_hooks:
                if hook is not None:
                    depth_samples = hook(
                        [depth_samples],
                        flip_same_as_last=True,
                        index_tensor_hw=(240, 320),
                    )[0]

            return input_img, target_img, mask, depth_samples

        return input_img, target_img, mask

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


class MutualRandomHorizontalFlip:
    """Randomly flips an input RGB imape and corresponding depth target horizontally with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p
        self.flipped_last = False

    def __call__(self, tensors, flip_same_as_last=False, index_tensor_hw=(None, None)):

        # do flip or not
        if flip_same_as_last:
            do_flip = self.flipped_last
        else:
            do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            # iterate through list of tensors (usually [rgb, depth] or [depth_samples])
            for i in range(len(tensors)):

                # check if tensor is image or list of depth samples
                if tensors[i].size(-1) == 3:  # tensor is list of depth samples
                    tensors[i][:, 1] = index_tensor_hw[1] - 1.0 - tensors[i][..., 1]
                else:  # tensor is img
                    tensors[i] = hflip(tensors[i])

        # save state so features can be flipped accordingly
        self.flipped_last = do_flip

        return tensors


class MutualRandomVerticalFlip:
    """Randomly flips an input RGB imape and corresponding depth target vertically with probability p.\\
    (Either both are transformed or neither of them)"""

    def __init__(self, p=0.5) -> None:
        self.p = p
        self.flipped_last = False

    def __call__(self, tensors, flip_same_as_last=False, index_tensor_hw=(None, None)):

        # do flip or not
        if flip_same_as_last:
            do_flip = self.flipped_last
        else:
            do_flip = torch.rand(1) < self.p

        # flip
        if do_flip:
            # iterate through list of tensors (usually [rgb, depth] or [depth_samples])
            for i in range(len(tensors)):

                # check if tensor is image or list of depth samples
                if tensors[i].size(-1) == 3:  # tensor is list of depth samples
                    tensors[i][:, 0] = index_tensor_hw[0] - 1.0 - tensors[i][..., 0]
                else:  # tensor is img
                    tensors[i] = vflip(tensors[i])

        # save state so features can be flipped accordingly
        self.flipped_last = do_flip

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
        # if mask.ndim == 2:
        #     mask = mask[np.newaxis, ...]

        # convert to tensor
        img_tensor = torch.from_numpy(img_np).to(self.device)
        # mask = torch.from_numpy(mask).to(self.device)

        return img_tensor  # , mask


class RandomFactor:
    def __init__(self, factor_range=(0.75, 1.25)) -> None:
        self.factor_range = factor_range

    def __call__(self, tensor):
        if tensor is None:
            return None

        factor = (
            torch.rand(1).item() * (self.factor_range[1] - self.factor_range[0])
            + self.factor_range[0]
        )

        tensor *= factor

        return tensor


class ReplaceInvalid:
    def __init__(self, value=None, return_mask=False) -> None:
        self.value = value
        self.return_mask = return_mask

    def __call__(self, tensor):

        if tensor is None:
            if self.return_mask:
                return None, None
            else:
                return None

        # mask all valid pixels (> 0.0)
        mask = tensor.gt(0.0)

        # if mask is empty, return None
        if not mask.any():
            print("Mask is empty, meaning all depth values invalid. Returning None.")
            if self.return_mask:
                return None, None
            else:
                return None

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

        if self.return_mask:
            return tensor, mask
        else:
            return tensor


def test_dataset(device="cpu"):

    print("Testing TrainDataset class ...")

    # test specific imports
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # define flip transforms (needed for hook usage later, check "sample_flip_hooks"
    mutual_horizontal_flip = MutualRandomHorizontalFlip()
    mutual_vertical_flip = MutualRandomVerticalFlip()

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
                ReplaceInvalid(return_mask=True),
            ]
        ),
        both_transform=transforms.Compose(
            [
                mutual_horizontal_flip,
                mutual_vertical_flip,
            ]
        ),
        use_csv_samples=True,
        samples_flip_hooks=[mutual_horizontal_flip, mutual_vertical_flip],
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=2)

    for batch_id, data in enumerate(dataloader):

        rgb_imgs = data[0]
        d_imgs = data[1]
        masks = data[2]
        depth_samples = data[3]

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
