import pandas as pd
import glob
from os.path import basename, join, exists, splitext
import argparse

from depth_estimation.evaluation import Evaluation


def modify_prediction(pr_img):
    return pr_img


def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--prediction_folder", type=str, required=True)
    parser.add_argument("--prediction_ext_pattern", type=str, required=True)
    parser.add_argument("--ground_truth_folder", type=str, required=True)
    parser.add_argument("--use_median_scaling", action="store_true")
    args = parser.parse_args()

    method_name = args.method_name
    prediction_folder = args.prediction_folder
    ground_truth_folder = args.ground_truth_folder
    prediction_ext_pattern = args.prediction_ext_pattern

    # find initial prediction list
    candidate_paths = glob.glob(prediction_folder + "/*" + prediction_ext_pattern)

    # construct ground truth list
    prediction_paths = []
    ground_truth_paths = []
    for path in candidate_paths:
        # imgname = splitext(basename(path))[0]
        imgname = basename(path).split(prediction_ext_pattern)[0]
        fname = join(ground_truth_folder, imgname + "_render.tif")
        if not exists(fname):
            continue
        prediction_paths.append(path)
        ground_truth_paths.append(fname)

    # initialize evaluation instance
    evaluation = Evaluation(
        method_name=method_name,
        logging=True,
        use_median_scaling=args.use_median_scaling,
        modify_prediction_func=modify_prediction,
    )

    # evaluate predictions
    benchmarks, benchmarks_detailed = evaluation.evaluate(
        prediction_paths, ground_truth_paths
    )
    print(benchmarks)

    benchmarks.to_csv(join(prediction_folder, "benchmarks.csv"), index=False)
    benchmarks_detailed.to_csv(
        join(prediction_folder, "benchmarks_detailed.csv"), index=False
    )


if __name__ == "__main__":
    main()
