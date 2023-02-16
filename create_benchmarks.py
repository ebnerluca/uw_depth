import pandas as pd
import glob
from os.path import basename, join, exists, splitext
import argparse

from depth_estimation.evaluation import Evaluation


def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_folder", type=str, required=True)
    parser.add_argument("--ground_truth_folder", type=str, required=True)
    args = parser.parse_args()

    prediction_folder = args.prediction_folder
    ground_truth_folder = args.ground_truth_folder

    # find initial prediction list
    candidate_paths = glob.glob(prediction_folder + "/*.tif")

    # construct ground truth list
    prediction_paths = []
    ground_truth_paths = []
    for path in candidate_paths:
        imgname = splitext(basename(path))[0]
        fname = join(ground_truth_folder, imgname + "_render.tif")
        if not exists(fname):
            continue
        prediction_paths.append(path)
        ground_truth_paths.append(fname)

    # initialize evaluation instance
    evaluation = Evaluation(logging=True)

    # evaluate predictions
    benchmarks, benchmarks_detailed = evaluation.evaluate(
        prediction_paths, ground_truth_paths
    )
    print(benchmarks)

    benchmarks.to_csv(join(prediction_folder, "benchmarks.csv"))
    benchmarks_detailed.to_csv(join(prediction_folder, "benchmarks_detailed.csv"))


if __name__ == "__main__":
    main()
