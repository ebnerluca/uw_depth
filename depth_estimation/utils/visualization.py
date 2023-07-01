# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def get_bin_centers_img(bin_edges, target, mask, device="cpu"):
    """Get visualization for usage on e.g. tensorboard which shows distribution
    of target depths and predicted bins."""

    # target shapes
    n_batch = target.size(0)
    n_edges = bin_edges.size(1)
    n_bins = n_edges - 1
    height = target.size(2)
    width = target.size(3)

    # get target bin edges by sampling the sorged target imgs at their quantiles
    target_bin_edges = torch.empty(n_batch, n_edges).to(device)
    for i in range(n_batch):

        target_sorted, _ = target[i, mask[i]].sort()
        step = target_sorted.size(0) / n_edges
        target_bin_edge_idcs = (torch.arange(n_edges) * step).long()
        target_bin_edges[i, ...] = target_sorted[target_bin_edge_idcs]

    # norm bin edges to [0,1]
    max_edges = bin_edges.amax(dim=1)
    max_target_edges = target_bin_edges.amax(dim=1)
    max = torch.stack([max_edges, max_target_edges]).amax(dim=0).unsqueeze(-1)  # Nx1
    bin_edges_normed = bin_edges / max
    target_bin_edges_normed = target_bin_edges / max

    # initialize out img and lines
    bin_edges_img = torch.zeros(n_batch, 1, height, width).to(device)
    bin_edges_img_line = torch.ones(n_batch, 1, 1, width).to(device) * 0.5
    target_bin_edges_img_line = torch.ones(n_batch, 1, 1, width).to(device) * 0.5

    n_bins = bin_edges_normed.size(1) - 1
    for i in range(n_batch):

        # draw bins
        black_white = True  # alternating black/white color for good visibility
        for j in range(n_bins):

            # get bin edges
            edge_start = (bin_edges_normed[i, j] * (width - 1)).int().item()
            edge_end = (bin_edges_normed[i, j + 1] * (width - 1)).int().item()
            target_edge_start = (
                (target_bin_edges_normed[i, j] * (width - 1)).int().item()
            )
            target_edge_end = (
                (target_bin_edges_normed[i, j + 1] * (width - 1)).int().item()
            )

            # draw lines
            bin_edges_img_line[i, 0, 0, edge_start:edge_end] = float(black_white)
            target_bin_edges_img_line[
                i, 0, 0, target_edge_start:target_edge_end
            ] = float(black_white)

            # alternate color
            black_white = not black_white

    # expand lines to multiple img rows to create full img
    bin_edges_img[:, :, : int(height / 2), :] = target_bin_edges_img_line.expand(
        n_batch, 1, int(height / 2), width
    )
    bin_edges_img[:, :, int(height / 2) :, :] = bin_edges_img_line.expand(
        n_batch, 1, int(height / 2), width
    )

    return bin_edges_img


def gray_to_heatmap(gray, colormap="inferno_r", normalize=True, device="cpu"):
    """Takes torch tensor input of shape [Nx1HxW], returns heatmap tensor of shape [Nx3xHxW].\\
    colormap 'inferno_r': [0,1] --> [bright, dark], e.g. for depths\\
    colormap 'inferno': [0,1] --> [dark, bright], e.g. for probabilities"""

    # get colormap
    colormap = plt.get_cmap(colormap)

    # gray imgs
    gray_imgs = [gray_img.cpu() for gray_img in gray]

    # normalize image wise
    if normalize:
        gray_imgs = [
            (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())
            for gray_img in gray_imgs
        ]

    # stack heatmaps batch wise (colormap does not support batches)
    heatmaps = [colormap(gray_img.squeeze())[..., :3] for gray_img in gray_imgs]
    heatmaps = np.stack(heatmaps, axis=0)

    # convert to tensor
    heatmaps = torch.from_numpy(heatmaps).permute(0, 3, 1, 2).to(device)

    return heatmaps


def get_tensorboard_grids(X, y, prior, pred, mask, bin_edges, device="cpu"):
    """Generates tensorboard grids for tensorboard summary writer.

    Inputs:
    - X: RGB input [Nx3xHxW]
    - y: ground truth depth [Nx1xHxW]
    - prior: prior parametrization [Nx2xHxW]
    - pred: prediction [Nx1xHxW]
    - mask: mask for valid depths [Nx1xHxW]
    - bin_edges: [Nxn_bins]
    """

    # error
    error = torch.abs(y - pred)
    error[~mask] = 0.0

    # target parametrization
    prior_map = prior[:, 0, ...].unsqueeze(1)
    dist_map = prior[:, 1, ...].unsqueeze(1)

    # resize rgb
    rgb_resized = torch.nn.functional.interpolate(
        X, size=[pred.size(2), pred.size(3)], mode="bilinear", align_corners=True
    )

    # get bin center visualization
    bin_centers_img = get_bin_centers_img(bin_edges, y, mask, device=device)

    # get heatmaps
    y_heatmap = gray_to_heatmap(y, device=device)
    pred_heatmap = gray_to_heatmap(pred, device=device)
    prior_heatmap = gray_to_heatmap(prior_map, device=device)
    dist_heatmap = gray_to_heatmap(dist_map, colormap="inferno", device=device)
    error_heatmap = gray_to_heatmap(error, colormap="inferno", device=device)
    mask_heatmap = gray_to_heatmap(mask.int(), colormap="inferno", device=device)
    bin_centers_heatmap = gray_to_heatmap(
        bin_centers_img, colormap="inferno", device=device
    )

    # grids
    nrow = X.size(0)
    rgb_target_pred_error_grid = make_grid(
        torch.cat(
            (rgb_resized, y_heatmap, pred_heatmap, error_heatmap, bin_centers_heatmap),
            dim=0,
        ),
        nrow=nrow,
    )
    prior_parametrization_grid = make_grid(
        torch.cat((y_heatmap, prior_heatmap, dist_heatmap, mask_heatmap), dim=0),
        nrow=nrow,
    )

    return rgb_target_pred_error_grid, prior_parametrization_grid
