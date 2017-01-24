"""
	This script takes in a set of notes and creates a sparse matrix for them
		The notes should already be in SUBJECT_ID/CHARTTIME sorted order, though order isn't important for this baseline
	It also takes in the corresponding codes for each patient
	Does logistic regression to predict codes from the text
		One LR classifier for each code
"""
import csv
import os
import sys

from collections import Counter
from scipy.sparse import csr_matrix

def main(Y):

	print("creating sparse matrix")
	notefile = '../mimicdata/notes_' + str(Y) + '_train.csv'
	X = construct_csr_matrix(notefile)

	print(X.getrow(0).toarray())

"""
	Returns: csr_matrix where each row is a BOW for the subject's entire set of notes
	Dimension: (# subjects in dataset) x (vocab size)
"""
def construct_csr_matrix(notefile):
	with open(notefile, 'r') as notesfile:
		reader = csv.reader(notesfile)
		next(reader)
		i = 0
		cur_id = 2

		subj_inds = [0]
		indices = []
		data = []
		vocab = {}

		for row in reader:
			if i % 10000 == 0:
				print(str(i) + " rows processed")
			subject_id = int(row[0])
			if subject_id != cur_id:
				subj_inds.append(len(indices))
				cur_id = subject_id
			text = row[2]
			for word in text.split():
				index = vocab.setdefault(word, len(vocab))
				indices.append(index)
				data.append(1)
			i += 1
	return csr_matrix((data, indices, subj_inds))


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])