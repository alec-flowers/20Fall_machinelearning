# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def rmse(y, y_hat):
    return np.sqrt(2*np.sum((y - y_hat)**2) / (2 * len(y_hat)))

def compute_rmse(y, tx, w):
    y_hat = tx.dot(w)
    return rmse(y, y_hat)