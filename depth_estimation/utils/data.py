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
        if self.value == "max":
            max = tensor[mask].max()
            tensor[~mask] = max
        elif self.value == "min":
            min = tensor[mask].min()
            tensor[~mask] = min
        elif self.value is not None:
            tensor[~mask] = self.value
        else:  # mask is None:
            pass  # leave invalid values unmodified

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

    # define dataset
    dataset = InputTargetDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/test.csv",
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
                MutualRandomHorizontalFlip(),
                MutualRandomVerticalFlip(),
            ]
        ),
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=2)

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
