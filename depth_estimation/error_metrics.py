import numpy as np


def mean_squared_error(pr_img, gt_img):
    """Mean Squared Error (MSE): Mean of squard pixel errors."""

    se = np.sum((pr_img.astype(np.double) - gt_img.astype(np.double)) ** 2)
    mse = se / (pr_img.shape[0] * pr_img.shape[1])

    return mse


def root_mean_squared_error(pr_img, gt_img):
    """Root Mean Squared Error (RMSE): Root of MSE"""

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
    """Mean Absolute Relative Error (MARE): Mean of pixel wise relative errors (error normalized by ground truth)"""

    are = np.sum(np.abs((pr_img - gt_img) / gt_img))  # element wise division
    mare = are / (pr_img.shape[0] * pr_img.shape[1])

    return mare


def relative_absolute_error(pr_img, gt_img):
    """Relative Absolute Error (RAE): Total absolute error normalized by total absolute error of simple predictor"""

    gt_mean = np.mean(gt_img)  # simple predictor
    rae = np.sum(np.abs(pr_img - gt_img)) / np.sum(np.abs(gt_img - gt_mean))

    return rae


def relative_squared_error(pr_img, gt_img):
    """Relative Squared Error (RSE): Total squared error normalized by total squared error of simple predictior."""

    gt_mean = np.mean(gt_img)  # simple predictor
    rse = np.sum((pr_img - gt_img) ** 2) / np.sum((gt_img - gt_mean) ** 2)

    return rse


def accuracy_value(pr_img, gt_img, d=1.25):
    sum = np.sum(np.maximum(pr_img / gt_img, gt_img / pr_img) < d)
    acc = sum / (pr_img.shape[0] * pr_img.shape[1])

    return acc
