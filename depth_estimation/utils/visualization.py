import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from .utils import resize_to_smallest, resize_to_biggest, resize, normalize_img


def visualize_heatmaps(imgs, img_names, resolution=None):

    for img, img_name in zip(imgs, img_names):

        print(img_name)
        print(f"Shape: {img.shape}")
        print(f"Range: [{np.amin(img)}, {np.amax(img)}]")
        print("")

    # resize
    if resolution is None:
        imgs = resize_to_smallest(imgs)
    else:
        imgs = resize(imgs, resolution)

    # generate heatmaps
    heatmaps = []
    for img_name, img in zip(img_names, imgs):

        heatmap = get_heatmap(img)
        heatmaps.append(heatmap)

        # show heatmap img
        cv2.imshow(img_name, heatmap)

    return heatmaps


def visualize_depth_histogram(imgs, img_names, n_bins=100):
    for img, img_name in zip(imgs, img_names):

        counts, bins = np.histogram(img, bins=n_bins)
        fig = plt.figure(img_name)
        plt.hist(bins[:-1], bins, weights=counts)
        fig.suptitle(img_name)

    plt.show()


def get_heatmap(img):

    out = normalize_img(img)
    colormap = plt.get_cmap("inferno")
    out = (colormap(1.0 - out) * 255).astype(np.uint8)[:, :, :3]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out


def get_tensorboard_grids(X, y, prior, pred, nrow):

    # target parametrization
    prior_map = prior[:, 0, ...].unsqueeze(1)
    dist_map = prior[:, 1, ...].unsqueeze(1)
    target_parametrization_grid = make_grid(
        torch.cat((y, prior_map, dist_map)), nrow=nrow
    )

    # rgb vs target vs pred
    rgb_resized = torch.nn.functional.interpolate(
        X, size=[pred.size(2), pred.size(3)], mode="bilinear", align_corners=True
    )
    rgb_target_pred_grid = make_grid(
        torch.cat((rgb_resized, y.repeat(1, 3, 1, 1), pred.repeat(1, 3, 1, 1))),
        nrow=nrow,
    )

    return target_parametrization_grid, rgb_target_pred_grid
