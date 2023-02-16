import numpy as np


def mean_squared_error(pr_img, gt_img):
    """Mean Squared Error (MSE) between two images with identical dimensions."""

    mse = np.sum((pr_img.astype(np.double) - gt_img.astype(np.double)) ** 2)
    mse /= pr_img.shape[0] * pr_img.shape[1]

    return mse


def root_mean_squared_error(pr_img, gt_img):
    """Root Mean Squared Error (RMSE) between two images with identical dimensions."""

    mse = np.sum((pr_img.astype(np.double) - gt_img.astype(np.double)) ** 2)
    mse /= pr_img.shape[0] * pr_img.shape[1]
    rmse = np.sqrt(mse)

    return rmse


def root_mean_squared_error(mean_squared_error):
    """Root Mean Squared Error (RMSE) between two images with identical dimensions."""

    rmse = np.sqrt(mean_squared_error)

    return rmse


def root_mean_squared_error_log(pr_img, gt_img):
    pass


def absolute_relative_error(pr_img, gt_img):
    pass


def squared_relative_error(pr_img, gt_img):
    pass


def accuracy_value(pr_img, gt_img, d=1.25):
    pass
