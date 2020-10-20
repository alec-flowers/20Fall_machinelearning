# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_error(y, tx, w):
	"""
	Computes error e.
	"""
	return y - tx.dot(w)

def mse(e):
	'Calculates and returns MSE between two vectors of same size'
	return np.sum(e ** 2) / (2 * len(e))


def mae(e):
	'Calculates and returns MAE between two vectors of same size'
	return np.sum(np.abs(e)) / len(e)


def rmse(e):
	'Calculates and returns RMSE between two vectors of same size'
	return np.sqrt(2 * mse(e))


def compute_loss(y, tx, w, error_fn='MSE'):
	"""
	Calculate the loss between dependent variable and prediction.

	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	w : ndarray of shape (n_weights,)
		Weight vector
	
	error_fn : string selecting ['MSE', 'MAE', 'RMSE']

	Returns
	----------
	error : np.float64
		error between dependent variable and prediction
	
	"""

	e = compute_error(y, tx, w)
	if error_fn == 'MSE':
		error = mse(e)
	elif error_fn == 'MAE':
		error = mae(e)
	elif error_fn == 'RMSE':
		error = rmse(e)
	else:
		raise NotImplementedError('Did not match a loss function')
	return error

def least_squares(y, tx):
	"""Least Squares
	
	Parameters
	----------
	y : ndarray of shape (n_samples,)
		Array of labels

	tx : ndarray of shape (n_samples, n_features)
		Training data

	Returns
	----------
	w : np.array of shape(1, D)
		Optimal weights calculated using normal equations.

	loss : np.float64
		RMSE loss for corresponding weight value
		
	"""
	w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
	loss = compute_loss(y, tx, w, 'MSE')
	return w, loss
