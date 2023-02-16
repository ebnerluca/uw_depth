import numpy as np


def mean_squared_error(pr_img, gt_img):
    """Mean Squared Error (MSE)"""

    se = np.sum((pr_img.astype(np.double) - gt_img.astype(np.double)) ** 2)
    mse = se / (pr_img.shape[0] * pr_img.shape[1])

    return mse


def root_mean_squared_error(pr_img, gt_img):
    """Root Mean Squared Error (RMSE)"""

    se = np.sum((pr_img.astype(np.double) - gt_img.astype(np.double)) ** 2)
    mse = se / (pr_img.shape[0] * pr_img.shape[1])
    rmse = np.sqrt(mse)

    return rmse


def root_mean_squared_log_error(pr_img, gt_img):
    """Root Mean Squared Log Error (RMSLE)"""

    sle = np.sum(
        (
            np.log(pr_img.astype(np.double) + 1.0)
            - np.log(gt_img.astype(np.double) + 1.0)
        )
        ** 2
    )
    msle = sle / (pr_img.shape[0] * pr_img.shape[1])
    rmsle = np.sqrt(msle)

    return rmsle


def mean_absolute_relative_error(pr_img, gt_img):
    """Mean Absolute Relative Error (MARE)"""

    are = np.sum(np.abs(pr_img - gt_img) / gt_img)  # element wise division
    mare = are / (pr_img.shape[0] * pr_img.shape[1])

    return mare


def mean_relative_absolute_error(pr_img, gt_img):
    """Mean Relative Absolute Error (MRAE)"""

    gt_mean = np.sum(gt_img) / (gt_img.shape[0] * gt_img.shape[1])  # simple predictor
    are = np.sum(np.abs(pr_img - gt_img)) / np.sum(np.abs(gt_img - gt_mean))
    mare = are / (pr_img.shape[0] * pr_img.shape[1])

    return mare


def mean_relative_squared_error(pr_img, gt_img):
    """Mean Relative Squared Error (MRSE)"""

    gt_mean = np.sum(gt_img) / (gt_img.shape[0] * gt_img.shape[1])  # simple predictor
    sre = np.sum((pr_img - gt_img) ** 2) / np.sum((gt_img - gt_mean) ** 2)
    msre = sre / (pr_img.shape[0] * pr_img.shape[1])

    return msre


def accuracy_value(pr_img, gt_img, d=1.25):
    sum = np.sum(np.maximum(pr_img / gt_img, gt_img / pr_img) < d)
    acc = sum / (pr_img.shape[0] * pr_img.shape[1])

    return acc
