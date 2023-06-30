# visualize tiff images that might not be in range [0,1]

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename

from data.example_dataset.dataset import get_example_dataset

MAX_IMGS = 5
dataset = get_example_dataset(shuffle=True)
rgb_paths = [rgb for rgb, _, _ in dataset.path_tuples][:MAX_IMGS]
depth_paths = [depth for _, depth, _ in dataset.path_tuples][:MAX_IMGS]

img_paths = rgb_paths + depth_paths


def get_heatmap(img, colormap="inferno_r"):

    out = normalize_img(img)
    colormap = plt.get_cmap(colormap)
    out = colormap(out)[:, :, :3]

    return out


def normalize_img(img):
    "Normalize img such that all entries are in [0,1]"

    max = np.max(img)
    min = np.min(img)
    range = max - min
    out = (img - min) / range

    return out


def main():
    for img_path in img_paths:
        img = Image.open(img_path)
        img_np = np.array(img)

        if len(img_np.shape) == 2:  # if grayscale
            print(f"{basename(img_path)} depth range [{img_np.min()}, {img_np.max()}]")
            img_np = get_heatmap(img_np)

        plt.figure(basename(img_path))
        plt.imshow(img_np)

    plt.show()


if __name__ == "__main__":
    main()
