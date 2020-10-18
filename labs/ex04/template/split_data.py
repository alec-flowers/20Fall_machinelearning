# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    rand_x = np.random.permutation(x)
    np.random.seed(seed)
    rand_y = np.random.permutation(y)
    
    t = int(ratio*len(x))
    
    train_x = rand_x[:t]
    test_x = rand_x[t:]
    train_y = rand_y[:t]
    test_y = rand_y[t:]
    
    return train_x, train_y, test_x, test_y
