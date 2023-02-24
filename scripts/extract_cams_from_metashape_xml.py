from os.path import dirname, join
import numpy as np
import pandas
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

# get file path
parser = ArgumentParser()
parser.add_argument("--xml_file", required=True)
args = parser.parse_args()
folder = dirname(args.xml_file)

tree = ET.parse(args.xml_file)
chunk = tree.find("chunk")  # assume only one chunk


# get camera intrinsics
sensors = chunk.find("sensors")
# sensor_names = []
# sensor_calibrations = []
for sensor in sensors.findall("sensor"):
    sensor_name = sensor.attrib["label"]
    width = float(sensor.find("resolution").attrib["width"])
    height = float(sensor.find("resolution").attrib["height"])
    for calibration in sensor.findall("calibration"):
        if calibration.attrib["class"] == "adjusted":

            # get f, cx, cy
            f = calibration.find("f").text
            cx = calibration.find("cx").text
            cy = calibration.find("cy").text

            # intrinsics
            K = np.array(
                [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )

            # metashape uses coordinate system in the middle of the image,
            # we want origin on top left, so we correct as follows:
            K[0, 2] += width / 2.0
            K[1, 2] += height / 2.0
        else:
            continue

    # save calibration
    intrinsics_file = join(folder, sensor_name + "_intrinsics.txt")
    np.savetxt(join(folder, intrinsics_file), K, fmt="%.9f")

# get camera extrinsics
cameras = chunk.find("cameras")  # assume only one cameras folder
camera_groups = cameras.findall("group")
for group in camera_groups:

    sensor_name = group.attrib["label"]
    cams = group.findall("camera")

    img_names = []
    transforms = []
    for cam in cams:
        transform = cam.find("transform")
        if transform is None:
            continue

        # get img name and transform
        img_name = cam.attrib["label"]
        transform = transform.text

        # append to list
        img_names.append(img_name)
        transforms.append(transform)

    d = {"img_name": img_names, "transform": transforms}
    df = pandas.DataFrame(data=d)
    print(df["transform"][0])
    extrinsics_file = join(folder, sensor_name + "_extrinsics.csv")
    df.to_csv(extrinsics_file, index=False)
