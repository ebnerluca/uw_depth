import torch
from torch.distributions.normal import Normal


def get_distance_maps(height, width, idcs_height, idcs_width, device="cpu"):
    """Returns a SxHxW tensor that captures the euclidean pixel distance to S
    sample pixels with coordinates (h,w)."""

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


def get_probability_maps(dist_map):
    """Takes a Nx1xHxW distance map as input and outputs a probability map.
    Pixels with small distance to closest keypoint have big probability and vice versa."""

    # normal distribution
    distribution = Normal(loc=0.0, scale=10.0)
    scale = torch.exp(
        distribution.log_prob(torch.zeros(1, device=dist_map.device))
    )  # used to enfore prob=1 at dist=0

    #  prior probability for every pixel
    prob_map = torch.exp(distribution.log_prob(dist_map)) / scale

    # exponential distribution
    # r = 0.05  # rate
    # prob_map = torch.exp(
    #     -r * dist_map
    # )  # dont multiply with r to have prior=1 at dist=0

    return prob_map


def get_depth_prior_from_ground_truth(
    targets, n_samples=200, mu=0.0, std=1.0, masks=None, device="cpu"
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
    for i in range(batch_size):

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

        # concat
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min
        features[i, :, 0] = idcs_height
        features[i, :, 1] = idcs_width
        features[i, :, 2] = depth_values

    # probability model:
    # convert pixel distance to probability
    signal_strength_maps = get_probability_maps(
        distance_maps, mu=mu, std=std, device=device
    )

    # parametrization
    parametrization = torch.cat((prior_maps, signal_strength_maps), dim=1)  # Nx2xHxW

    return parametrization, features


def get_depth_prior_from_features(
    features,
    height=240,
    width=320,
):
    """Takes lists of pixel indices and their respective depth probes and
    returns a dense depth prior parametrization.


    - One image represents the nearest neighbor guess (Inpired by: https://arxiv.org/abs/1804.02771).
    - The other image represents a probability map."""

    batch_size = features.size(0)

    # depth prior maps
    prior_maps = torch.empty(batch_size, 1, height, width).to(features.device)

    # euclidean distance maps
    distance_maps = torch.empty(batch_size, 1, height, width).to(features.device)

    # for every img, cannot vectorize because of masks with unequal length
    # (different images may have different number of features)
    for i in range(batch_size):

        # use only entries with valid depth
        mask = features[i, :, 2] > 0.0

        if not mask.any():
            max_dist = torch.sqrt(torch.pow(height, 2) + torch.pow(width, 2))
            prior_maps[i, ...] = 0.0
            distance_maps[i, ...] = max_dist
            print(
                "WARNING: Img has no valid features (depth > 0.0), using "
                + f"placeholder as parametrization (mosaic=0.0, dist={max_dist})."
            )
            continue

        # get list of indices and depth values
        idcs_height = features[i, mask, 0].round().long()
        idcs_width = features[i, mask, 1].round().long()
        depth_values = features[i, mask, 2]

        # get n_samples x height x width dist maps
        # (needs quite a bit of memory but is faster than iterating over every pixel)
        sample_dist_maps = get_distance_maps(
            height, width, idcs_height, idcs_width, device=features.device
        )
        # find min and argmin
        dist_map_min, dist_argmin = torch.min(sample_dist_maps, dim=0, keepdim=True)

        # nearest neighbor prior map
        prior_map = depth_values[dist_argmin]  # 1xHxW

        # concat
        prior_maps[i, ...] = prior_map
        distance_maps[i, ...] = dist_map_min

    # probability model:
    # convert pixel distance to probability
    prior_probability_maps = get_probability_maps(distance_maps)

    # parametrization
    parametrization = torch.cat((prior_maps, prior_probability_maps), dim=1)  # Nx2xHxW

    return parametrization


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


def test_get_priors_from_features(device="cpu"):

    # import modules only needed for testing
    import matplotlib.pyplot as plt
    import time

    print("Testing depth prior parametrization from given features ...")

    n_features = 200
    height = 240
    width = 320

    # generate target ground truth maps to generate prior from
    # in this case, simple gradient images are used
    target1 = torch.linspace(0, 0.5, width).repeat(height, 1) + torch.linspace(
        0, 0.5, height
    ).repeat(width, 1).transpose(0, 1)
    target1 = target1[None, None, ...]  # add batch and channel dimension
    target2 = 1.0 - target1
    targets = torch.cat((target1, target1, target2, target2), dim=0).to(device)

    # create some features that should be used for parametrization
    batch_size = targets.size(0)
    features = torch.empty(batch_size, 200, 3)
    for i in range(batch_size):

        # get random locations for fdepth samples
        idcs_height = torch.randperm(height)[:n_features].unsqueeze(1)
        idcs_width = torch.randperm(width)[:n_features].unsqueeze(1)

        # get the depth values at those locations
        depth_values = targets[i, 0, idcs_height, idcs_width]

        print(f"shape idcs height: {idcs_height.shape}")
        print(f"shape depth_values: {depth_values.shape}")

        # concat idcs and values to generate feature tensor
        img_features = torch.cat([idcs_height, idcs_width, depth_values], dim=1)

        # fill in
        features[i, ...] = img_features

    # get parametrization
    prior = get_depth_prior_from_features(features, height=height, width=width)
    prior_maps = prior[:, 0, ...].unsqueeze(1)
    signal_maps = prior[:, 1, ...].unsqueeze(1)

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

    print("Testing depth prior parametrization from given features done.")


if __name__ == "__main__":
    # test_get_priors()
    test_get_priors_from_features()
