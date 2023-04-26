from os.path import exists, basename, splitext
import time

import cv2
import pandas as pd

from .error_metrics import *
from ..utils import resize_to_smaller


class Evaluation:
    def __init__(
        self,
        method_name,
        logging=False,
        # use_median_scaling=False,
        modify_prediction_func=None,
        modify_ground_truth_func=None,
    ) -> None:
        self.method_name = method_name
        self.logging = logging
        self.modify_prediction = modify_prediction_func
        self.modify_ground_truth = modify_ground_truth_func
        if self.modify_prediction is not None:
            self.log("Prediction modifier function set!")
        if self.modify_ground_truth is not None:
            self.log("Ground truth modifier function set!")

    def evaluate(self, prediction_paths, ground_truth_paths):
        """Evaluates a given model on a given dataset and creates benchmarks such as RMSE etc."""

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
        mare_vec = []
        rse_vec = []
        accval_vec = []
        n_invalid = 0
        for i in range(n_predictions):

            # check foramat of prediction
            if splitext(prediction_paths[i])[1] == ".npy":
                pr_img = np.load(prediction_paths[i])
            else:
                pr_img = cv2.imread(prediction_paths[i], cv2.IMREAD_UNCHANGED)

            # ground truth is .tif
            gt_img = cv2.imread(ground_truth_paths[i], cv2.IMREAD_UNCHANGED)

            # get rid of unnecessary dimensions
            pr_img = pr_img.squeeze()
            gt_img = gt_img.squeeze()

            # skip if gt is incomplete
            if np.any(gt_img == 0.0):
                n_invalid += 1
                continue

            # modify prediction if some modification function is set
            if self.modify_prediction is not None:
                pr_img = self.modify_prediction(pr_img)

            if self.modify_ground_truth is not None:
                gt_img = self.modify_ground_truth(gt_img)

            mse, rmse, mare, rse, accval = self.evaluate_pair(pr_img, gt_img)

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
            mare_vec.append(mare)
            rse_vec.append(rse)
            accval_vec.append(accval)

            # print progress
            if i % 50 == 0:
                self.log(f"{i}/{n_predictions}")
                self.log(
                    f"Evaluating prediction {basename(prediction_paths[i])} with ground truth {basename(ground_truth_paths[i])}"
                )
            i += 1

        self.log("Creating benchmarks ...")

        # converting list to np array
        mse_vec = np.array(mse_vec)
        rmse_vec = np.array(rmse_vec)
        mare_vec = np.array(mare_vec)
        rse_vec = np.array(rse_vec)
        accval_vec = np.array(accval_vec)

        # mean/median of benchmarks over dataset
        mse_mean = np.mean(mse_vec)
        rmse_mean = np.mean(rmse_vec)
        mare_mean = np.mean(mare_vec)
        rse_mean = np.mean(rse_vec)
        accval_mean = np.mean(accval_vec)

        # benchmarks dict
        benchmarks = {
            "method": [self.method_name],
            "mse_mean": [mse_mean],
            "rmse_mean": [rmse_mean],
            "mare_mean": [mare_mean],
            "rse_mean": [rse_mean],
            "accval_mean": [accval_mean],
            "best_mse": [best_mse],
            "best_img": [best_img],
            "worst_mse": [worst_mse],
            "worst_img": [worst_img],
        }

        benchmarks_detailed = {
            "name": name_vec,
            "mse": mse_vec,
            "rmse": rmse_vec,
            "mare": mare_vec,
            "rse": rse_vec,
            "accval": accval_vec,
        }

        # pandas
        benchmarks = pd.DataFrame.from_dict(benchmarks)
        benchmarks_detailed = pd.DataFrame.from_dict(benchmarks_detailed)

        self.log("Evaluation done.")
        self.log(
            f"Warning: {n_invalid} image pairs were skipped because the ground truth was incomplete."
        )
        return benchmarks, benchmarks_detailed

    # def evaluate_pair(self, prediction_path, ground_truth_path):
    def evaluate_pair(self, pr_img, gt_img):

        # match img dimensions
        if pr_img.shape != gt_img.shape:
            pr_img, gt_img = resize_to_smaller(pr_img, gt_img)

        # benchmarks
        mse = mean_squared_error(pr_img, gt_img)
        rmse = np.sqrt(mse)  # do this rather than root_mean_squared_error bc its faster
        mare = mean_absolute_relative_error(pr_img, gt_img)
        rse = relative_squared_error(pr_img, gt_img)
        accval = accuracy_value(pr_img, gt_img)

        return mse, rmse, mare, rse, accval

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
