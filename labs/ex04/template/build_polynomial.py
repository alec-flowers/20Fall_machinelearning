# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    degree = int(degree)
    basis = np.power(x,0).reshape(x.shape[0],1)
    for i in range(1,degree+1):
        basis = np.insert(basis, 1, np.power(x,i), axis = 1)
    return basis
