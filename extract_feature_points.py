# This file goes through all the input rgb images in a dataset
# and extracts feature points. These feature points are then used to
# sample the set of sparse priors

import cv2
import numpy as np
import pandas as pd
from os.path import splitext, join, basename, dirname, exists
from os import mkdir

from datasets.datasets import get_flsea_dataset  # , get_usod10k_dataset

##########################################
################# CONFIG #################
##########################################

# grid
n_rows, n_cols = 4, 4

# n features
nfeatures = 200

# output shapes
in_height = 480
in_width = 640
out_height = 240
out_width = 320

# get img paths
dataset = get_flsea_dataset(split="dataset")
path_tuples = dataset.path_tuples

# output
rel_folder = "features"
file_ending = "_features.csv"

debug = True

##########################################
##########################################
##########################################


# print config
print(f"input shape: {in_width}x{in_height}")
print(f"output shape: {out_width}x{out_height}")
print(f"{nfeatures} per img, devided in {n_rows}x{n_cols} cells")


def draw_keypoints(img, points, color=(0, 255, 0)):

    # iterate over points
    for point in points.astype(np.int):

        # cv2 uses x,y axis notation
        xy_tuple = (point[1], point[0])

        # draw circle
        img = cv2.circle(img, xy_tuple, color=color, radius=8, thickness=2)

    return img


def get_features(gray, row_edges, col_edges, detector, n_patch_features=10):

    # extract features for each patch
    kp_np = np.empty((0, 2))
    for i in range(n_rows):
        for j in range(n_cols):

            # create patch from img
            patch = gray[
                row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]
            ]

            # extract features for patch
            kp_patch = detector.detect(patch, None)[:n_patch_features]
            if len(kp_patch) <= 0:
                continue
            print(f"kp responses: {[p.response for p in kp_patch]}")

            # store in array, format [row, col] instead of cv2's [x,y]
            kp_np_patch = np.array([[p.pt[1], p.pt[0]] for p in kp_patch])

            # correct by patch shift
            kp_np_patch_shifted = kp_np_patch + [row_edges[i], col_edges[j]]

            # append
            kp_np = np.vstack((kp_np, kp_np_patch_shifted))

            # debug: draw mini img
            # if debug:
            #     print(f"Patch has {len(kp_patch)} features.")
            #     patch_img = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            #     patch_img = draw_keypoints(patch_img, kp_np_patch)
            #     cv2.imshow(f"patch img [{i}, {j}]", patch_img)

    # debug: sift over whole image at once
    # if debug:
    #     out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #     kp_full = sift_detector_full.detect(gray, None)
    #     kp_np_full = np.array([[p.pt[1], p.pt[0]] for p in kp_full])
    #     vis_img_full = out_img.copy()
    #     vis_img_full = draw_keypoints(vis_img_full, kp_np_full)
    #     cv2.imshow("detected points on full img", vis_img_full)
    #     ver_img = out_img.copy()
    #     ver_img = cv2.drawKeypoints(gray, kp_full, ver_img)
    #     cv2.imshow("verification_img", ver_img)

    # print(f"Found {len(kp_np)} features")

    return kp_np


nfeatures_patch = int(nfeatures / (n_rows * n_cols))
print(f"Detecting {nfeatures_patch} features per patch")

# sift detector
sift_detector = cv2.SIFT_create(
    nfeatures=nfeatures_patch,
    contrastThreshold=0.03,
    # # edgeThreshold=200,
    # sigma=1.0,
)

# edges for patch indices
row_step = in_height / n_rows
col_step = in_width / n_cols
row_edges = np.array(range(n_rows + 1), dtype=np.float)
col_edges = np.array(range(n_cols + 1), dtype=np.float)
row_edges = (row_edges * row_step).astype(np.int)
col_edges = (col_edges * col_step).astype(np.int)


# for all imgs
n_tuples = len(path_tuples)
i = 0
for path_tuple in path_tuples:

    # paths
    rgb_path = path_tuple[0]
    depth_path = path_tuple[1]
    out_dir = join(dirname(rgb_path), rel_folder)
    out_filename = splitext(basename(rgb_path))[0] + file_ending
    out_path = join(out_dir, out_filename)

    # read imgs in grayscale
    gray = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # resize imgs
    gray = cv2.resize(gray, dsize=(in_width, in_height))
    depth = cv2.resize(depth, dsize=(out_width, out_height))

    # get features
    gray_features = get_features(
        gray, row_edges, col_edges, sift_detector, nfeatures_patch
    )

    # index transform to fit output shape
    height_scale = out_height / in_height
    width_scale = out_width / in_width
    features = gray_features * [height_scale, width_scale]

    # get depth values
    depth_values = depth[features[:, 0].astype(np.int), features[:, 1].astype(np.int)]
    depth_values = depth_values[..., np.newaxis]

    # concat
    row_col_depth = np.hstack((features, depth_values))

    # filter out features where depth is invalid (= 0)
    valid_mask = row_col_depth[:, 2] > 0.0
    row_col_depth = row_col_depth[valid_mask, :]

    # write output file
    if not exists(out_dir):
        mkdir(out_dir)
    pd.DataFrame(row_col_depth).to_csv(
        out_path, header=["row", "column", "depth"], index=None
    )

    if i % 50 == 0:
        print(f"processed {i}/{n_tuples}")
    i += 1

    if debug:

        # print number of features
        print(f"Found {len(depth_values)} features ({len(row_col_depth)} valid)")

        # prepare visualization imgs
        img = cv2.imread(path_tuple[0])
        img = cv2.resize(img, dsize=(in_width, in_height))
        img_full = img.copy()
        img_full_cv2 = img.copy()

        # show img with features
        zeros_mask = depth_values[:, 0] == 0.0  # mask where depth is zero
        img = draw_keypoints(img, gray_features[~zeros_mask, :], color=(0, 255, 0))
        img = draw_keypoints(img, gray_features[zeros_mask, :], color=(0, 0, 255))
        cv2.imshow("img with features", img)

        # show depth with features
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        depth = draw_keypoints(depth, features[~zeros_mask, :], color=(0, 255, 0))
        depth = draw_keypoints(depth, features[zeros_mask, :], color=(0, 0, 255))
        cv2.imshow("depth", depth)

        # show comparison to detection on full img
        sift_detector_full = cv2.SIFT_create(nfeatures=nfeatures)
        kp_full = sift_detector_full.detect(gray, None)
        kp_np_full = np.array([[p.pt[1], p.pt[0]] for p in kp_full])
        img_full = draw_keypoints(img_full, kp_np_full)
        img_full_cv2 = cv2.drawKeypoints(gray, kp_full, img_full_cv2)
        cv2.imshow("detected points on full img (no patches)", img_full)
        cv2.imshow("verification_img", img_full_cv2)

        print("Waiting for key press ...")
        cv2.waitKey()

print("Done.")
