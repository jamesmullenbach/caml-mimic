"""
	This script reads the output of get_notes.py and builds a vocabulary of terms of size 40k
	Size of vocab-10 dataset: 1522558 rows
	Size of vocab-100 dataset: 2047967 rows
	Output: txt file with the 40k words
"""
import csv
import numpy as np
import sys

from collections import defaultdict
from scipy.sparse import csr_matrix

MIN_OCCURRENCES = 3
VOCAB_SIZE = 40000
total_rows = {'10': '1522558', '100': '2047967'}

def main(dataset):
	"""
			1. Create sparse document occurrence matrix (rows are terms, columns are notes)
			2. Drop any row with sum < MIN_OCCURRENCES
			3. Calculate tf-idf scores, keep list of top VOCAB_SIZE terms
			4. Drop all rows corresponding to non-top-VOCABS_SIZE terms
	"""
	with open('../mimicdata/notes_' + dataset + '.csv', 'r') as csvfile:
		reader = csv.reader(csvfile)

		#0. read in data
		print("reading in data...")
		#holds number of terms in each document
		note_numwords = []
		#indices where notes start
		note_inds = [0]
		#indices of discovered words
		indices = []
		#holds a bunch of ones
		data = []
		#keep track of discovered words
		vocab = {}
		#preallocate array to hold number of notes each term appears in
		note_occur = np.zeros(400000, dtype=int)
		i = 0
		for row in reader:
			if i % 10000 == 0:
				print(str(i) + " of " + total_rows[dataset] + " read")
			text = row[1]
			for term in text.split():
				#put term in vocab if it's not there. else, get the index
				index = vocab.setdefault(term, len(vocab))
				indices.append(index)
				data.append(1)
			#record where the next note starts
			note_inds.append(len(indices))
			indset = set(indices[note_inds[-2]:note_inds[-1]])
			for ind in indset:
				note_occur[ind] += 1
			note_numwords.append(numwords)
			i += 1
		#clip trailing zeros
		note_occur = note_occur[note_occur>0]
		print(len(note_occur))
		#build lookup table for terms
		num2term = {n:w for w,n in vocab.items()}

		#1. create sparse document matrix
		print("building matrix")
		C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
		#also need the numwords array to be a sparse matrix
		note_numwords = csr_matrix(1 / np.array(note_numwords))
		
		#2. remove rows with less than 3 total occurrences
		print("removing rare terms")
		#inds holds indices of rows corresponding to terms that occur in < 3 documents
		inds = np.nonzero(note_occur >= MIN_OCCURRENCES)[0]
		print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
		#drop those rows
		C = C[inds,:]
		note_occur = note_occur[inds]

		#3. calculate tf-idf scores
		#each term gets one score: sum of its row over sum of
		print("calculating tf-idf scores")
		#note_numwords is number of words in each note. tf is a csr matrix b/c we made note_numwords into one
		tf = C.multiply(note_numwords)
		N = note_numwords.shape[1]
		idf = np.log(N / note_occur)
		print(idf.shape)
		tf = np.asarray(tf.sum(1)).flatten()
		print(tf.shape)
		tfidf = np.multiply(tf, idf)
		tfidf = np.squeeze(np.asarray(tfidf))
		print(tfidf.shape)

		print("sorting to get top " + str(VOCAB_SIZE))
		args = tfidf.argsort()
		if len(args) > VOCAB_SIZE:
			inds = args[-VOCAB_SIZE:]
		else:
			inds = args

		kept_terms = []
		scores = []
		print("writing output")
		vocab_file = open('../mimicdata/vocab_' + str(MIN_OCCURRENCES) + '.txt', 'w')
		for ind in inds:
			kept_terms.append(num2term[ind])
			scores.append(tfidf[ind])
			vocab_file.write(num2term[ind] + "\n")
		vocab_file.close()
		print("sampling of the kept terms")
		print(kept_terms[-10:])
		print(scores[-10:])
		#slice 'em
		print("reducing matrix to top 40k")
		C = C[inds]




if __name__ == "__main__":
	if sys.argv[1] not in ["10", "100", "full"]:
		print("usage: python build_vocab.py [10|100|full]")
		sys.exit(0)
	main(sys.argv[1])