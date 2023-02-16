import cv2
import matplotlib.pyplot as plt
import numpy as np

from os.path import basename

from .utils import resize_to_smallest


def visualize_heatmaps(img_paths):

    # read
    imgs = []
    img_names = []
    for img_path in img_paths:

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_name = basename(img_path)

        print(img_name)
        print(f"Shape: {img.shape}")
        print(f"Range: [{np.amin(img)}, {np.amax(img)}]")
        print("")

        imgs.append(img)
        img_names.append(img_name)

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

    cv2.waitKey(0)
