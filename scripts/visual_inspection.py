import cv2
import numpy as np

from os.path import splitext, basename
from depth_estimation_evaluation.visualization import (
    visualize_heatmaps,
    visualize_depth_histogram,
)

paths = [
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_053306_440_LC16_render.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/MVSNet/out/depths_mvsnet/00000013_init.pfm",
    # "/home/auv/depth_estimation/UDepth/data/output/inference/RGB/PR_20221105_053800_024_LC16.tif",
    # "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_053702_410_LC16_render.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/UDepth/RGB/PR_20221105_053306_440_LC16.tif",
    # "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_054048_860_LC16_render.tif",
    # "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/UDepth/RGB/PR_20221105_053940_725_LC16.tif",
    # "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_053704_414_LC16_render.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/ManyDepth/multi/PR_20221105_053306_440_LC16_disp_multi.npy",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/ManyDepth/mono/PR_20221105_053306_440_LC16_disp_mono.npy",
]


def main():

    # read imgs
    imgs = []
    img_names = []
    for path in paths:
        if splitext(path)[1] == ".npy":
            img = np.load(path)
            img = np.squeeze(img)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
        img_names.append(basename(path))

    heatmaps = visualize_heatmaps(imgs, img_names, resolution=(500, 500))
    visualize_depth_histogram(imgs, img_names)

    cv2.waitKey(0)

    # save heatmaps
    # for name, heatmap in zip(img_names, heatmaps):
    #     cv2.imwrite(splitext(name)[0] + ".png", heatmap)


if __name__ == "__main__":
    main()
