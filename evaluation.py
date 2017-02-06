"""
	This file contains evaluation methods that take in a set of predicted labels 
		and a set of ground truth labels and calculate precision, recall, accuracy, f1, and ranking loss
"""
from collections import defaultdict
import numpy as np

def all_metrics(yhat, y):
	return accuracy(yhat, y), precision(yhat, y), recall(yhat, y), f1(yhat, y)

def accuracy(yhat, y):
	return sum(intersect_size(yhat, y) / union_size(yhat, y)) / y.shape[0]

def precision(yhat, y):
	num = intersect_size(yhat, y) / yhat.sum(axis=1)
	num = [0 if np.isnan(n) else n for n in num]
	return sum(num) / y.shape[0]

def recall(yhat, y):
	return sum(intersect_size(yhat, y) / y.sum(axis=1)) / y.shape[0]

def f1(yhat, y):
	return sum(2*intersect_size(yhat, y) / (y.sum(axis=1) + yhat.sum(axis=1))) / y.shape[0]

def union_size(yhat, y):
	return np.logical_or(yhat, y).sum(axis=1)

def intersect_size(yhat, y):
	return np.logical_and(yhat, y).sum(axis=1)