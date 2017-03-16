"""
	This file contains evaluation methods that take in a set of predicted labels 
		and a set of ground truth labels and calculate precision, recall, accuracy, f1, and ranking loss
"""
from collections import defaultdict
import numpy as np

from scipy import interp
from sklearn.metrics import roc_curve, auc

def all_metrics(yhat, y):
        #returns dict, dict, dict
        names = ["acc", "prec", "rec", "f1"]
        macro = all_macro(yhat, y)
        micro = all_micro(yhat, y)
        metrics = {names[i]: macro[i] for i in range(len(macro))}
        metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})

        auc_vals = auc_metrics(yhat, y)
        if auc_vals is not None:
            fpr = auc_vals[0]
            tpr = auc_vals[1]
            roc_auc = auc_vals[2]
            metrics.update(roc_auc)
            return metrics, fpr, tpr
        else:
            return metrics, None, None
	

def all_macro(yhat, y):
	return accuracy(yhat, y), precision(yhat, y), recall(yhat, y), f1(yhat, y)

def all_micro(yhat, y):
	return micro_accuracy(yhat, y), micro_precision(yhat, y), micro_recall(yhat, y), micro_f1(yhat, y)

def accuracy(yhat, y):
	return sum(intersect_size(yhat, y, 1) / union_size(yhat, y, 1)) / y.shape[0]

def precision(yhat, y):
	num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
	#correct for divide-by-zeros
	num[np.isnan(num)] = 0.
	return sum(num) / y.shape[0]

def recall(yhat, y):
	return sum(intersect_size(yhat, y, 1) / y.sum(axis=1)) / y.shape[0]

def f1(yhat, y):
	return sum(2*intersect_size(yhat, y, 1) / (y.sum(axis=1) + yhat.sum(axis=1))) / y.shape[0]

def micro_accuracy(yhat, y):
        num = intersect_size(yhat, y, 0) / union_size(yhat, y, 0)
        num[np.isnan(num)] = 0.
        return np.mean(num)

def micro_precision(yhat, y):
	num = intersect_size(yhat, y, 0) / yhat.sum(axis=0)
	num[np.isnan(num)] = 0.
	return np.mean(num)

def micro_recall(yhat, y):
        num = intersect_size(yhat, y, 0) / y.sum(axis=0)
        num[np.isnan(num)] = 0.
        return np.mean(num)

def micro_f1(yhat, y):
        num = 2*intersect_size(yhat,y,0) / (y.sum(axis=0) + yhat.sum(axis=0))
        num[np.isnan(num)] = 0.
        return np.mean(num)

def auc_metrics(yhat, y):
        if yhat.shape[0] <= 1:
            return
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(y.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat[:,i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                roc_auc["auc_%d" % i] = auc(fpr[i], tpr[i])

        #macro-AUC: kind of like an average of the ROC curves for all classes
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y.shape[1])]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(y.shape[1]):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= y.shape[1]
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["auc"] = auc(fpr["macro"], tpr["macro"])

        #micro-AUC: just looks at each individual prediction
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), yhat.ravel())
        roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

        return fpr, tpr, roc_auc

def union_size(yhat, y, axis):
	#axis=0 for label-level union (micro). axis=1 for instance-level (macro)
	return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
	#axis=0 for label-level union (micro). axis=1 for instance-level (macro)
	return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics):
    print
    print("[MACRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"])
    print("[MICRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])
    print
