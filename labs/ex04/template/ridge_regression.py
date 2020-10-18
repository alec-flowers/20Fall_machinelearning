# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse, compute_rmse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(tx)
    D = tx.shape[1]
    
    w = np.linalg.solve(tx.T.dot(tx)+2*N*lambda_*np.identity(D),tx.T.dot(y))
    e = compute_rmse(y, tx, w)
    
    return w, e
