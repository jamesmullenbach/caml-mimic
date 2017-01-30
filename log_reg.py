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

def main(Y):

	print("creating sparse matrix")
	notefile = '../mimicdata/notes_' + str(Y) + '_train_final.csv'
	X, yy = construct_X_Y(notefile, Y)

	print(X.getrow(0).toarray())
	print(yy[:2])

	print("X.shape: " + str(X.shape))
	print("yy.shape: " + str(yy.shape))

	# build the classifiers
	classifiers = [LogisticRegression() for _ in range(Y)]
	for i,y in enumerate(yy):
		print("Fitting classifier " + str(i))
		classifiers[i].fit(X, y)

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
			print("Processing", end="")
		for row in reader:
			if i % 10000 == 0:
				if notebook_print:
					print(".", end="")
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

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(int(sys.argv[1]))