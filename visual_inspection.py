from depth_estimation.visualization import visualize_heatmaps

heatmap_paths = [
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_054521_904_LC16_render.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/UDepth/RGB/PR_20221105_054521_904_LC16.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render/PR_20221105_053702_410_LC16_render.tif",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/UDepth/RGB/PR_20221105_053702_410_LC16.tif",
]


def main():
    visualize_heatmaps(heatmap_paths)


if __name__ == "__main__":
    main()
