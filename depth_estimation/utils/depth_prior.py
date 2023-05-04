import torch
from torch.distributions.normal import Normal


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


def get_depth_prior_from_ground_truth(
    targets, n_samples=200, mu=0.0, std=1.0, normalize=True, device="cpu"
):
    """Takes an Nx1xHxW ground truth depth tensor and desired number of samples,
    returns two images per batch represention a prior guess parametrization:

    - One image represents a mosaic representing the nearest neighbor guess.
    By default, sampled depth values are normalized.
    - The other image represents a probability map.

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
        prior_map = priors[dist_argmin]  # 1xHxW

        # linear distance model:
        # normalize and invert prior map (close points should have strong signals)
        # signal_strength_map = 1.0 - (dist_map_min / dist_map_min.max())

        # normalize the depth prior values to [0,1]
        if normalize:
            eps = 1e-10  # avoid zero division if max == min (e.g. only one sample)
            min = prior_map.min()
            max = prior_map.max()
            prior_map = (prior_map - min) / (max - min + eps)

        # concat
        prior_maps = torch.cat((prior_maps, prior_map.unsqueeze(0)), dim=0)
        distance_maps = torch.cat((distance_maps, dist_map_min.unsqueeze(0)), dim=0)

    # probability model:
    # convert pixel distance to signal strength
    signal_strength_maps = get_signal_maps(distance_maps, mu=mu, std=std, device=device)

    parametrization = torch.cat((prior_maps, signal_strength_maps), dim=1)  # Nx2xHxW
    return parametrization


def get_depth_prior_from_features(
    idcs_height,
    idcs_widht,
    values,
    height=240,
    width=320,
    mu=0.0,
    std=1.0,
    device="cpu",
):
    """Takes lists of pixel indeces and their respective depth probes and
    returns a depth prior parametrization."""

    pass
