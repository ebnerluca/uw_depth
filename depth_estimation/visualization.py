import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .utils import resize_to_smallest


def visualize_heatmaps(imgs, img_names):

    for img, img_name in zip(imgs, img_names):

        print(img_name)
        print(f"Shape: {img.shape}")
        print(f"Range: [{np.amin(img)}, {np.amax(img)}]")
        print("")

    # resize
    imgs = resize_to_smallest(imgs)

    # generate heatmaps
    for img_name, img in zip(img_names, imgs):

        tmp_min = np.amin(img)
        tmp_range = np.amax(img) - tmp_min
        img = (img - tmp_min) / tmp_range
        colormap = plt.get_cmap("inferno")
        img = (colormap(1.0 - img) * 255).astype(np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # show heatmap img
        cv2.imshow(img_name, img)


def visualize_depth_histogram(imgs, img_names, n_bins=100):
    for img, img_name in zip(imgs, img_names):

        counts, bins = np.histogram(img, bins=n_bins)
        fig = plt.figure(img_name)
        plt.hist(bins[:-1], bins, weights=counts)
        fig.suptitle(img_name)

    plt.show()
