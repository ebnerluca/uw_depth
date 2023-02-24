import open3d as o3d
import pandas as pd
import numpy as np

transforms_path = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/export/cams/LC_extrinsics.csv"

# load transforms
df = pd.read_csv(transforms_path)
transforms = df["transform"]

# create frames
frames = []
for t in transforms:
    t_mat = np.array(t.split(), dtype=np.float32)
    t_mat = t_mat.reshape((4, 4))

    translation = t_mat[:3, 3]
    rotation = t_mat[:3, :3]

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.translate(translation, relative=False)
    frame.rotate(rotation)

    frames.append(frame)

# visualize
o3d.visualization.draw_geometries(frames)
