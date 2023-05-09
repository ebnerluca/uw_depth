# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

# from .utils import resize_to_smallest, resize_to_biggest, resize, normalize_img


# def visualize_heatmaps(imgs, img_names, resolution=None):

#     for img, img_name in zip(imgs, img_names):

#         print(img_name)
#         print(f"Shape: {img.shape}")
#         print(f"Range: [{np.amin(img)}, {np.amax(img)}]")
#         print("")

#     # resize
#     if resolution is None:
#         imgs = resize_to_smallest(imgs)
#     else:
#         imgs = resize(imgs, resolution)

#     # generate heatmaps
#     heatmaps = []
#     for img_name, img in zip(img_names, imgs):

#         heatmap = get_heatmap(img)
#         heatmaps.append(heatmap)

#         # show heatmap img
#         cv2.imshow(img_name, heatmap)

#     return heatmaps


# def visualize_depth_histogram(imgs, img_names, n_bins=100):
#     for img, img_name in zip(imgs, img_names):

#         counts, bins = np.histogram(img, bins=n_bins)
#         fig = plt.figure(img_name)
#         plt.hist(bins[:-1], bins, weights=counts)
#         fig.suptitle(img_name)

#     plt.show()


# def get_heatmap(img):

#     out = normalize_img(img)
#     colormap = plt.get_cmap("inferno")
#     out = (colormap(1.0 - out) * 255).astype(np.uint8)[:, :, :3]
#     out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

#     return out


def gray_to_heatmap(gray, colormap="inferno_r", normalize=False, device="cpu"):
    """Takes torch tensor input of shape [Nx1HxW], returns heatmap tensor of shape [Nx3xHxW].\\
    colormap 'inferno_r': [0,1] --> [bright, dark], e.g. for depths\\
    colormap 'inferno': [0,1] --> [dark, bright], e.g. for signals"""

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


def get_tensorboard_grids(X, y, prior, pred, device="cpu"):
    """Generates tensorboard grids for tensorboard summary writer.

    Inputs:
    - X: RGB input [Nx3xHxW]
    - y: ground truth depth [Nx1xHxW]
    - prior: prior parametrization [Nx2xHxW]
    - pred: prediction [Nx1xHxW]
    - nrow: batch size

    Outputs:
    - target parametrization grid
    - rgb vs. target vs. prediction vs. error grid
    """

    # error
    error = torch.abs(y - pred)

    # target parametrization
    prior_map = prior[:, 0, ...].unsqueeze(1)
    dist_map = prior[:, 1, ...].unsqueeze(1)

    # resize rgb
    rgb_resized = torch.nn.functional.interpolate(
        X, size=[pred.size(2), pred.size(3)], mode="bilinear", align_corners=True
    )

    # get heatmaps
    y_heatmap = gray_to_heatmap(y, device=device)
    pred_heatmap = gray_to_heatmap(pred, device=device)
    prior_heatmap = gray_to_heatmap(prior_map, device=device)
    dist_heatmap = gray_to_heatmap(dist_map, colormap="inferno", device=device)
    error_heatmap = gray_to_heatmap(
        error, colormap="inferno", normalize=True, device=device
    )

    # grids
    nrow = X.size(0)
    rgb_target_pred_error_grid = make_grid(
        torch.cat((rgb_resized, y_heatmap, pred_heatmap, error_heatmap), dim=0),
        nrow=nrow,
    )
    prior_parametrization_grid = make_grid(
        torch.cat((y_heatmap, prior_heatmap, dist_heatmap), dim=0), nrow=nrow
    )

    return rgb_target_pred_error_grid, prior_parametrization_grid
