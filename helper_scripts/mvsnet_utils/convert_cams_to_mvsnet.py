# use this script to convert a extrinsic and intrinsic csv file to format used by MVSNet

import pandas as pd
import numpy as np
from os.path import join

intrinsic_file = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/cams/LC_intrinsics.txt"
extrinsic_file = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/cams/LC_extrinsics.csv"
output_folder = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/MVSNet_cams"

min_depth = 1000  # min depth in mm
max_depth = 3500  # max depth in mm
max_d = 192  # max depth prediction in mvsnet
interval = (max_depth - min_depth) / max_d


def np_array_to_str(array):
    np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.6f}".format})
    s = str(array)
    s = s.replace("\n ", "\n")
    s = s.replace("[", "")
    s = s.replace("]", "")
    return s


# find img names
df = pd.read_csv(extrinsic_file)
img_names = df["img_name"]

# get intrinsics
intrinsic = np.loadtxt(intrinsic_file)
intrinsic_str = np_array_to_str(intrinsic)

# get extrinsics
extrinsics = df["transform"]

# generate file for every image
for img_name, extrinsic in zip(img_names, extrinsics):
    filename = img_name + "_cam.txt"
    filepath = join(output_folder, filename)
    extrinsic_np = np.array(extrinsic.split(), dtype=np.float32)
    extrinsic_np = np.reshape(extrinsic_np, (4, 4))
    extrinsic_str = np_array_to_str(extrinsic_np)

    with open(filepath, "w") as f:
        f.write("extrinsic\n")
        f.write(str(extrinsic_str))
        f.write("\n\n")
        f.write("intrinsic\n")
        f.write(str(intrinsic_str))
        f.write("\n\n")
        f.write(f"{min_depth} {interval}")
