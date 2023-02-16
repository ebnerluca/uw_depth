from os.path import exists, basename

import cv2
import pandas as pd

from .error_metrics import *
from .utils import resize_to_smaller


class Evaluation:
    def __init__(self, logging=False) -> None:
        self.logging = logging

    def evaluate(self, prediction_paths, ground_truth_paths):
        """Evaluates a given model on a given dataset."""

        self.check_dataset(prediction_paths, ground_truth_paths)

        n_predictions = len(prediction_paths)
        self.log(f"Evaluation for {n_predictions} img pairs ...")

        # best and worst
        best_img = None
        best_mse = np.inf
        worst_img = None
        worst_mse = 0.0

        # evaluate
        name_vec = []
        mse_vec = []
        rmse_vec = []
        for i in range(n_predictions):

            # evaluate current prediction
            mse, rmse = self.evaluate_pair(prediction_paths[i], ground_truth_paths[i])

            # best and worst
            if mse < best_mse:
                best_mse = mse
                best_img = basename(prediction_paths[i])
            if mse > worst_mse:
                worst_mse = mse
                worst_img = basename(prediction_paths[i])

            # store benchmark
            name_vec.append(basename(prediction_paths[i]))
            mse_vec.append(mse)
            rmse_vec.append(rmse)

            # print progress
            if i % 10 == 0:
                self.log(f"{i}/{n_predictions}")
            i += 1

        self.log("Creating benchmarks ...")

        # converting list to np array
        mse_vec = np.array(mse_vec)
        rmse_vec = np.array(rmse_vec)

        # mean/median of benchmarks over dataset
        mse_mean = np.mean(mse_vec)
        rmse_mean = np.mean(rmse_vec)
        mse_median = np.median(mse_vec)
        rmse_median = np.median(rmse_vec)

        # benchmarks dict
        benchmarks = {
            "mse_mean": [mse_mean],
            "rmse_mean": [rmse_mean],
            "mse_median": [mse_median],
            "rmse_median": [rmse_median],
            "best_mse": [best_mse],
            "best_img": [best_img],
            "worst_mse": [worst_mse],
            "worst_img": [worst_img],
        }

        benchmarks_detailed = {"name": name_vec, "mse": mse_vec, "rmse": rmse_vec}

        # pandas
        benchmarks = pd.DataFrame.from_dict(benchmarks)
        benchmarks_detailed = pd.DataFrame.from_dict(benchmarks_detailed)

        self.log("Evaluation done.")

        return benchmarks, benchmarks_detailed

    def evaluate_pair(self, prediction_path, ground_truth_path):
        # print(f"Comparing {basename(prediction_path)} to{basename(ground_truth_path)}")

        # read prediction and ground truth image
        pr_img = cv2.imread(prediction_path, cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(ground_truth_path, cv2.IMREAD_UNCHANGED)

        # match img dimensions
        if pr_img.shape != gt_img.shape:
            pr_img, gt_img = resize_to_smaller(pr_img, gt_img)

        # MSE
        mse = mean_squared_error(pr_img, gt_img)
        rmse = np.log(mse)  # do this rather than root_mean_squared_error bc its faster

        # self.log(f"MSE: {round(mse, 3)}, RMSE: {round(rmse, 3)}")

        return mse, rmse

    def check_dataset(self, pr_paths, gt_paths):
        # check if all files exist
        for f in pr_paths + gt_paths:
            if not exists(f):
                self.error(f"File {f} does not exist!")
                raise FileNotFoundError

        n = len(pr_paths)
        n_gt = len(gt_paths)
        if n != n_gt:
            self.error(
                f"Prection paths number ({n}) should match ground truth paths number ({n_gt})!"
            )
            raise TypeError

    def log(self, message: str):
        """Logging wrapper for evaluation class."""

        if self.logging:
            print(f"[Evaluation]: {message}")

    def error(self, message: str):
        print(f"[Evaluation][ERROR]: {message}")
