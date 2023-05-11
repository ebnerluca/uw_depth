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
    targets, n_samples=200, mu=0.0, std=1.0, normalize=True, masks=None, device="cpu"
):
    """Takes an Nx1xHxW ground truth depth tensor and desired number of samples,
    returns two images per batch represention a prior guess parametrization:

    - One image represents a mosaic representing the nearest neighbor guess.
    By default, sampled depth values are normalized.
    - The other image represents a probability map.

    Inpired by: https://arxiv.org/abs/1804.02771
    """

    # batch size
    batch_size = targets.size(0)

    # output size
    height = targets.size(2)
    width = targets.size(3)
    n_pixels = height * width

    # depth prior maps
    prior_maps = torch.empty(batch_size, 1, height, width).to(device)

    # euclidean distance maps
    distance_maps = torch.empty(batch_size, 1, height, width).to(device)

    # features lists with pixel indices and depth values
    features = torch.empty(batch_size, n_samples, 3).to(device)

    # utility to select pixel locations and avoid slow torch.where
    pixel_idcs = torch.arange(n_pixels).to(device)

    # for each image
    for i in range(targets.size(0)):

        # identify valid pixels
        if masks is not None:
            valid_pixel_idcs = pixel_idcs[masks[i, 0, ...].flatten()]
            n_valid_pixels = len(valid_pixel_idcs)
            if n_valid_pixels < n_samples:
                print(
                    f"WARNING: Could not find enough valid pixels in depth map. "
                    + f"Need at least {n_samples}, but found only {n_valid_pixels} samples. "
                    + f"Reducing n_samples to {n_valid_pixels} for this batch."
                )
                n_samples = n_valid_pixels
                features = features[:, :n_samples, :]
        else:
            valid_pixel_idcs = pixel_idcs

        # get random indices
        idcs_selection = torch.randperm(valid_pixel_idcs.size(0))[:n_samples]
        idcs = valid_pixel_idcs[idcs_selection]

        # convert flattened indices to height and width indices
        idcs_height = idcs.div(width, rounding_mode="floor")
        idcs_width = idcs.remainder(width)

        # get n_samples x height x width dist maps
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=device
        )

        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)

        # sample depth priors at indices
        depth_values = targets[i, 0, idcs_height, idcs_width]

        # nearest neighbor prior map
        prior_map = depth_values[dist_argmin]  # 1xHxW

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
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min
        features[i, :, 0] = idcs_height
        features[i, :, 1] = idcs_width
        features[i, :, 2] = depth_values

    # probability model:
    # convert pixel distance to signal strength
    signal_strength_maps = get_signal_maps(distance_maps, mu=mu, std=std, device=device)

    parametrization = torch.cat((prior_maps, signal_strength_maps), dim=1)  # Nx2xHxW

    return parametrization, features


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


def test_get_priors(device="cpu"):
    """Test sparse prior generation and parametrization."""

    print("Testing depth prior parametrization ...")

    # import modules only needed for testing
    import matplotlib.pyplot as plt
    import time

    # generate target ground truth maps to generate prior from
    # in this case, simple gradient images are used
    target1 = torch.linspace(0, 0.5, 320).repeat(240, 1) + torch.linspace(
        0, 0.5, 240
    ).repeat(320, 1).transpose(0, 1)
    target1 = target1[None, None, ...]  # add batch and channel dimension
    target2 = 1.0 - target1
    targets = torch.cat((target1, target1, target2, target2), dim=0).to(device)
    masks = targets > 0.5

    # get priors and dist_map
    starttime = time.time()
    prior, _ = get_depth_prior_from_ground_truth(
        targets,
        n_samples=100,
        mu=0.0,
        std=10.0,
        normalize=True,
        masks=masks,
        device=device,
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
    for i in range(targets.size(0)):

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

    print("Testing depth prior parametrization done.")


if __name__ == "__main__":
    test_get_priors()
