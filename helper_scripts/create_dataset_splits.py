# this file splits the whole dataset into training and test data and
# generates respective csv files with the names of the [imagea, ground_truth_depth] pairs

import csv
import glob
from os.path import join, basename, exists
import cv2
import numpy as np
import pandas as pd
import random

###### CONFIG

images_folder = "/home/auv/FLSea/archive/red_sea/pier_path/pier_path/imgs/"
ground_truth_depth_folder = "/home/auv/FLSea/archive/red_sea/pier_path/pier_path/depth/"
images_pattern = ".tiff"
ground_truth_depth_pattern = "_SeaErra_abs_depth.tif"
split_sizes = [
    1.0,
    0.0,
    0.0,
]  # train, validation, test percentage: Must add up to 1.
split_names = [
    "train",
    "validation",
    "test",
]
allow_zero = True  # allow depth imgs with pixel values zero (=invalid)
allow_zero_range = False  # allow img range [0,0]

#######

# search candidates
depth_candidate_paths = glob.glob(
    join(ground_truth_depth_folder, "*" + ground_truth_depth_pattern)
)
print(f"Found {len(depth_candidate_paths)} candidates")

# find pairs
imgs = []
depths = []
i = 0
for depth_candidate_path in depth_candidate_paths:

    # check if depth map is valid, if not then skip
    if not allow_zero or not allow_zero_range:
        depth_candidate = cv2.imread(depth_candidate_path, cv2.IMREAD_UNCHANGED)
        if not allow_zero_range and (depth_candidate.min() == depth_candidate.max()):
            continue
        if not allow_zero and (np.any(depth_candidate <= 0.0)):
            continue

    # get img name
    img_name = basename(depth_candidate_path).split(ground_truth_depth_pattern)[0]

    # append pair to imgs and depths list
    imgs.append(join(images_folder, img_name + images_pattern))
    depths.append(depth_candidate_path)

    if i % 100 == 0:
        print(
            f"Checking depth_candidates for invalid values: {i}/{len(depth_candidate_paths)}"
        )
    i += 1
print(f"Found {len(depths)} valid depth maps.")


# shuffle
pairs = list(zip(imgs, depths))
random.shuffle(pairs)


# get splits ranges
n_pairs = len(imgs)
n_splits = len(split_sizes)
split_ranges = []
idx = 0
for i in range(n_splits):
    split_ranges.append((idx, idx + int(split_sizes[i] * n_pairs)))
    idx += int(split_sizes[i] * n_pairs)
# print(f"n_pairs: {n_pairs}, split_ranges: {split_ranges}")

# create splits, each split is a tuple with (imgs_list, depths_list)
splits = []
for split_range in split_ranges:
    splits.append((sorted(pairs[split_range[0] : split_range[1]])))

# validation check
for split in splits:  # for all splits
    for img, depth in split:

        if (not exists(img)) or (not exists(depth)):
            print("Validation failed, file is missing!")
            print(f"Missing pair: {img}, {depth}")
            exit(1)

# Summary
print(f"Created {n_splits} splits:")
for i in range(n_splits):
    split_name = split_names[i]
    print(f"    - {split_name} split [{len(splits[i])} pairs]")

# write file
print(f"Writing csv files to {images_folder} ...")
for i in range(n_splits):
    split_name = split_names[i]
    split_imgs = [img for img, _ in splits[i]]
    split_depths = [depth for _, depth in splits[i]]
    d = {"img": split_imgs, "depth": split_depths}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(join(images_folder, split_name + ".csv"), index=False, header=False)

print("Done.")
