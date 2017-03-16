"""
	This script takes in a set of notes and creates a sparse matrix for them
		The notes should already be in SUBJECT_ID/CHARTTIME sorted order, though order isn't important for this baseline
	It also takes reads the corresponding codes for each patient and creates a sklearn-ready matrix of labels

	Mostly been using construct_X_Y within jupyter
"""
import csv
import numpy as np
import os
import sys

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import evaluation

#Constants
C = 0.1
MAX_ITER = 100

def main(Y, dataset):
	
	old_version = get_version()
	
	print("creating sparse matrix")
	notefile = '../mimicdata/notes_%d_train_%s.csv' % (Y, dataset)
	print("notefile: " + str(notefile))
	X, yy = construct_X_Y(notefile, Y, False)

	print(X.getrow(0).toarray())
	print(yy[:2])

	devfile = notefile.replace("train", "dev")
	X_dv, yy_dv = construct_X_Y(devfile, Y, False)

	print("X.shape: " + str(X.shape))
	print("yy.shape: " + str(yy.shape))
	print("X_dv.shape: " + str(X_dv.shape))
	print("yy_dv.shape: " + str(yy_dv.shape))

	# build the classifier
	clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=MAX_ITER))
	# train
	clf.fit(X, yy)
	#evaluate
	yhat = clf.predict(X_dv)
	metrics, fpr, tpr = evaluation.all_metrics(yhat, yy_dv)
	print("[MACRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"])
	print("[MICRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])

	print("sanity check on train")
	yhat_tr = clf.predict(X)
	metrics_t, fpr_t, tpr_t = evaluation.all_metrics(yhat_tr, yy)
	print("[MACRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics_t["acc"], metrics_t["prec"], metrics_t["rec"], metrics_t["f1"], metrics_t["auc"])
	print("[MICRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics_t["acc_micro"], metrics_t["prec_micro"], metrics_t["rec_micro"], metrics_t["f1_micro"], metrics_t["auc_micro"])

	roc_auc = {"macro": metrics["auc"], "micro": metrics["auc_micro"]}
	roc_auc.update({i: metrics["auc_%d" % i] for i in range(Y)})
	if old_version:
		write_auc(fpr, tpr, roc_auc, Y)
	else:
		plot_auc(fpr, tpr, roc_auc)

def construct_X_Y(notefile, Y, notebook_print=True):
	"""
		Each new subject_id, label_list pair is a new row

		Returns: csr_matrix where each row is a BOW for the subject's entire set of notes
		Dimension: (# subjects in dataset) x (vocab size)
	"""
	yy = []
	with open(notefile, 'r') as notesfile:
		reader = csv.reader(notesfile)
		next(reader)
		i = 0
		cur_id = 0
		cur_labels = set([])

		subj_inds = []
		indices = []
		data = []

		if notebook_print:
			print("Processing")
		for row in reader:
			if i % 10000 == 0:
				if notebook_print:
					print(".")
				else:
					print(str(i) + " done")
			subject_id = int(row[0])
			label_set = set([int(l) for l in row[2:]])
			if subject_id != cur_id or label_set != cur_labels:
				subj_inds.append(len(indices))
				cur_id = subject_id
				cur_labels = label_set
				yy.append([1 if i in cur_labels else 0 for i in range(Y)])
			text = row[1]
			for word in text.split():
				index = int(word)
				indices.append(index)
				data.append(1)
			i += 1
		subj_inds.append(len(indices))

	return csr_matrix((data, indices, subj_inds)), np.array(yy)

def write_auc(fpr, tpr, roc_auc, Y):
	#write the AUC values for later visualization
	with open("auc_" + str(Y) + ".csv", 'w') as outfile:
		outfile.write(','.join(['label', 'measure', 'values']) + "\n")
		for label in fpr.keys():
			fpr_line = [str(label), 'fpr']
			fpr_line.extend([str(v) for v in fpr[label]])
			outfile.write(','.join(fpr_line) + "\n")
	
			tpr_line = [str(label), 'tpr']
			tpr_line.extend([str(v) for v in tpr[label]])
			outfile.write(','.join(tpr_line) + "\n")

			auc_line = [str(label), 'auc', str(roc_auc[label])]
			outfile.write(','.join(auc_line) + "\n")

def get_version():
	#use the older version of keras when not running on my pc
	import socket
	return ("james" not in socket.gethostname())

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|] dataset"))
		sys.exit(0)
	main(int(sys.argv[1]), sys.argv[2])
