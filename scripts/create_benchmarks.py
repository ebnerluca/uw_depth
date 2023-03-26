import pandas as pd
import glob
from os.path import basename, join, exists, splitext
import argparse

from depth_estimation_evaluation.evaluation import Evaluation
from depth_estimation_evaluation.utils import normalize_img


def modify_prediction(pr_img):
    return normalize_img(pr_img)


def modify_ground_truth(gt_img):
    return normalize_img(gt_img)


def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--prediction_folder", type=str, required=True)
    parser.add_argument("--prediction_ext_pattern", type=str, required=True)
    parser.add_argument("--ground_truth_folder", type=str, required=True)
    args = parser.parse_args()

    method_name = args.method_name
    prediction_folder = args.prediction_folder
    ground_truth_folder = args.ground_truth_folder
    prediction_ext_pattern = args.prediction_ext_pattern

    # find initial prediction list
    candidate_paths = glob.glob(prediction_folder + "/*" + prediction_ext_pattern)
    print(f"len cands: {len(candidate_paths)}")

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

    # # contruct ground truth list (mvsnet)
    # img_names_conversion_file = "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/inference/MVSNet/out/depths_mvsnet/img_names.txt"
    # with open(img_names_conversion_file, "r") as f:
    #     img_name_conversions = f.read().split("\n")
    #     img_name_conversions.remove("")
    #     img_name_conversions = {
    #         conversion.split()[0]: conversion.split()[1]
    #         for conversion in img_name_conversions
    #     }
    # prediction_paths = []
    # ground_truth_paths = []
    # for path in candidate_paths:
    #     # imgname = splitext(basename(path))[0]
    #     prediction_name = basename(path).split(prediction_ext_pattern)[0]
    #     ground_truth_name = img_name_conversions[
    #         prediction_name + prediction_ext_pattern
    #     ]
    #     fname = join(ground_truth_folder, ground_truth_name)
    #     if not exists(fname):
    #         continue
    #     prediction_paths.append(path)
    #     ground_truth_paths.append(fname)

    # initialize evaluation instance
    evaluation = Evaluation(
        method_name=method_name,
        logging=True,
        modify_prediction_func=modify_prediction,
        modify_ground_truth_func=modify_ground_truth,
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
