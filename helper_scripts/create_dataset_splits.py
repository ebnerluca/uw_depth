# this file splits the whole dataset into training and test data and
# generates respective csv files with the names of the [imagea, ground_truth_depth] pairs

import csv
import glob
from os.path import join, basename, exists
import cv2
import numpy as np
import pandas as pd
import random

# params
images_folder = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/i20221105_053256_cv"
ground_truth_depth_folder = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/depth_render"
images_pattern = "LC16.png"
ground_truth_depth_pattern = "LC16_render.tif"
split_sizes = [
    0.8,
    0.1,
    0.1,
]  # train, validation, test percentage: Must add up to 1.
split_names = [
    "train",
    "validation",
    "test",
]

# search candidates
depth_candidate_paths = glob.glob(
    join(ground_truth_depth_folder, "*" + ground_truth_depth_pattern)
)
print(f"Found {len(depth_candidate_paths)} candidates")

# find pairs
imgs = []
depths = []
i = 0
for candidate_path in depth_candidate_paths:
    # read img
    img_candidate = cv2.imread(candidate_path, cv2.IMREAD_UNCHANGED)

    # check if depth map is incomplete, if yes then skip
    if np.any(img_candidate <= 0.0):
        continue

    # get img name
    img_name = basename(candidate_path).split(ground_truth_depth_pattern)[0]

    # append pair to imgs and depths list
    imgs.append(img_name + images_pattern)
    depths.append(img_name + ground_truth_depth_pattern)

    if i % 100 == 0:
        print(
            f"Checking depth_candidates for missing values: {i}/{len(depth_candidate_paths)}"
        )
    i += 1
print(f"Found {len(depths)} valid depth maps.")


# shuffle
random.shuffle(imgs)
random.shuffle(depths)

# get splits ranges
n_pairs = len(imgs)
n_splits = len(split_sizes)
split_ranges = []
idx = 0
for i in range(n_splits):
    split_ranges.append((idx, idx + int(split_sizes[i] * n_pairs)))
    idx += int(split_sizes[i] * n_pairs)
print(f"n_pairs: {n_pairs}, split_ranges: {split_ranges}")

# create splits, each split is a tuple with (imgs_list, depths_list)
splits = []
for split_range in split_ranges:
    splits.append(
        (
            sorted(imgs[split_range[0] : split_range[1]]),
            sorted(depths[split_range[0] : split_range[1]]),
        )
    )

# validation check
for split in splits:  # for all splits
    for img, depth in zip(split[0], split[1]):  # for every pair in the split
        img_path = join(images_folder, img)
        depth_path = join(ground_truth_depth_folder, depth)

        if (not exists(img_path)) or (not exists(depth_path)):
            print("Validation failed, file is missing!")
            print(f"Missing pair: {img_path}, {depth_path}")
            exit(1)

# Summary
print(f"Created {n_splits} splits:")
for i in range(n_splits):
    split_name = split_names[i]
    print(f"    - {split_name} split [{len(splits[i][0])} pairs]")

# write file
for i in range(n_splits):
    split_name = split_names[i]
    d = {"img": splits[i][0], "depth": splits[i][1]}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(join(images_folder, split_name + ".csv"), index=False, header=False)
