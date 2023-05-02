import torch
from torch.distributions.normal import Normal
import numpy as np


from PIL import Image
import csv
import random
from os.path import exists


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


def get_distance_maps(height, width, idcs_height, idcs_width, device="cpu"):
    """Returns a SxHxW tensor that captures the euclidean pixel distance to S
    sample pixels S (h,w) in the tensor."""

    dist_maps = torch.empty(0, height, width).to(device)
    for idx_height, idx_width in zip(idcs_height, idcs_width):

        # vertical and horizontal distance vectors
        height_dists = torch.arange(0, height, device=device) - idx_height
        width_dists = torch.arange(0, width, device=device) - idx_width

        # vertical and horizontal distance maps
        height_dist_map = height_dists.repeat(width, 1).transpose(0, 1)
        width_dist_map = width_dists.repeat(height, 1)

        # distance map
        dist_map = torch.sqrt(
            torch.pow(height_dist_map, 2) + torch.pow(width_dist_map, 2)
        ).unsqueeze(0)

        dist_maps = torch.cat((dist_maps, dist_map), dim=0)

    return dist_maps


def get_signal_maps(dist_map, mu=0.0, std=1.0, device="cpu"):
    """Takes a Nx1xHxW distance map as input and outputs a signal strength map.
    Points with small distance value correspond to great signal strength and vice versa."""

    # signal distribution model
    distribution = Normal(loc=mu, scale=std)
    scale = torch.exp(
        distribution.log_prob(torch.zeros(1, device=device))
    )  # used to enfore signal=1 at dist=0

    # compute signal strength for every pixel
    signal_map = torch.exp(distribution.log_prob(dist_map)) / scale

    return signal_map


def get_depth_prior_parametrization(
    targets, n_samples=200, mu=0.0, std=1.0, normalize=True, device="cpu"
):
    """Takes an Nx1xHxW ground truth depth tensor and desired number of samples,
    returns two images per batch represention a prior guess parametrization:

    - One image represents a mosaic representing the nearest neighbor guess.
    By default, sampled depth values are normalized.
    - The other image represents the distance from eatch pixel to the closest known sample.

    Inpired by: https://arxiv.org/abs/1804.02771
    """

    # output size
    height = targets.size(2)
    width = targets.size(3)

    prior_maps = torch.empty(0, 1, height, width).to(device)  # depth prior map
    distance_maps = torch.empty(0, 1, height, width).to(
        device
    )  # euclidean pixel distance map
    for batch in range(targets.size(0)):

        # get random indices
        idcs_height = torch.randint(
            low=0, high=height, size=(n_samples,), device=device
        )
        idcs_width = torch.randint(low=0, high=width, size=(n_samples,), device=device)

        # get n_samples x height x width dist maps
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=device
        )

        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)

        # sample depth priors at indices
        priors = targets[batch, 0, idcs_height, idcs_width]

        # nearest neighbor prior map
        prior_map = priors[dist_argmin]

        # linear distance model:
        # normalize and invert prior map (close points should have strong signals)
        # signal_strength_map = 1.0 - (dist_map_min / dist_map_min.max())

        # normalize the depth prior values
        if normalize:
            eps = 1e-10
            min = prior_map.min()
            max = prior_map.max()
            prior_map = (prior_map - min + eps) / (
                max - min + eps
            )  # shift and scale to range [0,1]

        # concat
        prior_maps = torch.cat((prior_maps, prior_map.unsqueeze(0)), dim=0)
        distance_maps = torch.cat((distance_maps, dist_map_min.unsqueeze(0)), dim=0)

    # probability model:
    # convert pixel distance to signal strength
    signal_strength_maps = get_signal_maps(distance_maps, mu=mu, std=std, device=device)

    parametrization = torch.cat((prior_maps, signal_strength_maps), dim=1)  # Nx2xHxW
    return parametrization


def test_get_priors():
    """Test sparse prior generation and parametrization."""

    print("Testing depth prior parametrization ...")

    # import modules only needed for testing
    import matplotlib.pyplot as plt
    import time

    # set device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # generate target ground truth maps to generate prior from
    # in this case, simple gradient images are used
    target1 = torch.linspace(0, 0.5, 320).repeat(240, 1) + torch.linspace(
        0, 0.5, 240
    ).repeat(320, 1).transpose(0, 1)
    target1 = target1[None, None, ...] * 3.0  # add batch and channel dimension
    target2 = 1.0 - target1
    targets = torch.cat((target1, target1, target2, target2), dim=0).to(device)

    # get priors and dist_map
    starttime = time.time()
    prior = get_depth_prior_parametrization(
        targets, n_samples=100, mu=0.0, std=10.0, normalize=True, device=device
    )
    prior_maps = prior[:, 0, ...].unsqueeze(1)
    signal_maps = prior[:, 1, ...].unsqueeze(1)
    elapsed_time = time.time() - starttime
    print(f"sampling time: {elapsed_time} seconds")

    # copy back to cpu for visuals
    targets = targets.cpu()
    prior_maps = prior_maps.cpu()
    signal_maps = signal_maps.cpu()

    # plot
    # for i in range(targets.size(0)):
    for i in range(1):

        target = targets[i, ...]
        prior_map = prior_maps[i, ...]
        signal_map = signal_maps[i, ...]
        plt.figure(f"target {i}")
        plt.imshow(target.permute(1, 2, 0))
        plt.figure(f"prior map {i}")
        plt.imshow(prior_map.permute(1, 2, 0))
        plt.figure(f"dist map {i}")
        plt.imshow(signal_map.permute(1, 2, 0))

        print(f"prior map {i} range: [{prior_map.min()}, {prior_map.max()}]")
        print(f"signal map {i} range: [{signal_map.min()}, {signal_map.max()}]")

    plt.show()


# run as "python -m depth_estimation.utils.data"
if __name__ == "__main__":
    test_get_priors()
