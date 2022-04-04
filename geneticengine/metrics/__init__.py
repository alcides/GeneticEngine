from __future__ import annotations

import numpy as np


def mae(y_pred, y_gt):
    """
    Calculate mean absolute error

    y_pred: the predicted output
    y_gt: ground truth
    :return: The mean absolute error.
    """

    return np.mean(np.abs(y_pred - y_gt))


def precision(y_pred, y_gt, binary=False):
    """
    Calculate precision. A higher precision corresponds to a better prediction

    y_pred: the predicted output
    y_gt: ground truth
    binary: Set to true if y_pred is already binarised
    :return: The precision.
    """
    y_pred = np.array(y_pred)
    y_gt = np.array(y_gt)
    if not binary:
        y_pred[y_pred < 0] = -1
        y_pred[y_pred >= 0] = 1

    n_positives = len(y_pred[y_pred == 1])

    fp = sum((y_pred[y_pred == 1] - y_gt[y_pred == 1]) / 2)
    precision = (n_positives - fp) / n_positives

    return precision


def recall(y_pred, y_gt, binary=False):
    """
    Calculate recall. A higher recall corresponds to a better prediction

    y_pred: the predicted output
    y_gt: ground truth
    binary: Set to true if y_pred is already binarised
    :return: The recall.
    """
    y_pred = np.array(y_pred)
    y_gt = np.array(y_gt)
    if not binary:
        y_pred[y_pred < 0] = -1
        y_pred[y_pred >= 0] = 1

    n_predicitons = len(y_pred)
    n_positives = len(y_pred[y_pred == 1])

    tp = n_positives - sum((y_pred[y_pred == 1] - y_gt[y_pred == 1]) / 2)
    fn = abs(sum((y_pred[y_pred == -1] - y_gt[y_pred == -1]) / 2))

    recall = tp / (tp + fn)
    return recall


def f1_score(y_pred, y_gt, binary=False):
    """
    Calculate f1_score. A higher f1_score corresponds to a better prediction

    y_pred: the predicted output
    y_gt: ground truth
    binary: Set to true if y_pred is already binarised
    :return: The f1_score.
    """
    y_pred = np.array(y_pred)
    y_gt = np.array(y_gt)
    if not binary:
        y_pred[y_pred < 0] = -1
        y_pred[y_pred >= 0] = 1

    n_positives = len(y_pred[y_pred == 1])

    fp = sum((y_pred[y_pred == 1] - y_gt[y_pred == 1]) / 2)
    tp = n_positives - fp
    fn = abs(sum((y_pred[y_pred == -1] - y_gt[y_pred == -1]) / 2))

    denom = tp + ((fn + fp) / 2)
    if denom == 0:
        return 0

    f1_score = tp / denom
    return f1_score


def mse(y_pred, y_gt):
    """
    Calculate mean squared error. A higher mean squared error corresponds to a better prediction

    y_pred: the predicted output
    y_gt: ground truth
    binary: Set to true if y_pred is already binarised
    :return: The mean squared error.
    """

    return np.mean(np.square(y_pred - y_gt))


def rmse(y_pred, y_gt):
    """
    Calculate root mean squared error. A higher root mean squared error corresponds to a better prediction

    y_pred: the predicted output
    y_gt: ground truth
    binary: Set to true if y_pred is already binarised
    :return: The root mean squared error.
    """

    return np.sqrt(np.mean(np.square(y_pred - y_gt)))
