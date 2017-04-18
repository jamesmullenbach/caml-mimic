"""
    This script reads a sorted training dataset and builds a vocabulary of terms of given size
    Output: txt file with the vocab_size words
"""
import csv
import numpy as np
import operator
import sys

from collections import defaultdict
from scipy.sparse import csr_matrix

from constants import DATA_DIR

def main(vocab_size, Y, vocab_min):
    """
            1. Create sparse document occurrence matrix (rows are terms, columns are notes)
            2. Drop any row with sum < vocab_min
            3. Calculate tf-idf scores, keep list of top vocab_size terms
            4. Drop all rows corresponding to non-top-vocab_size terms
    """
    if vocab_size != "full":
        vocab_size = int(vocab_size)
    with open('%s/notes_%s_train.csv' % (DATA_DIR, Y), 'r') as csvfile:
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
        #build lookup table for terms
        num2term = {}
        #preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            if i % 10000 == 0:
                print(str(i) + " read")
            text = row[2]
            numwords = 0
            for term in text.split():
                #put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            #record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            #go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        #clip trailing zeros
        note_occur = note_occur[note_occur>0]

        #turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word,ind in sorted(vocab.iteritems(), key=operator.itemgetter(1))])

        #1. create sparse document matrix
        print("building matrix")
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        print("C.shape: " + str(C.shape))
        #also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))
        
        #2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        #inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        #drop those rows
        C = C[inds,:]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        #3. calculate tf-idf scores
        #each term gets one score: sum of its row over sum of
        print("calculating tf-idf scores")
        #note_numwords is number of words in each note. tf is a csr matrix b/c we made note_numwords into one
        tf = C.multiply(note_numwords)
        N = note_numwords.shape[1]
        idf = np.log(N / note_occur)
        tf = np.asarray(tf.sum(1)).flatten()
        tfidf = np.multiply(tf, idf)
        tfidf = np.squeeze(np.asarray(tfidf))

        
        print("sorting to get top " + str(vocab_size))
        args = tfidf.argsort()
        if vocab_size != "full" and len(args) > vocab_size:
            inds = args[-vocab_size:]
        else:
            inds = args

        kept_terms = []
        scores = []
        print("writing output")
        vocab_file = open('%s/vocab_lookup_%s_%s_%d.txt' % (DATA_DIR, str(vocab_size), Y, vocab_min), 'w')
        print("inds: " + str(inds))
        for ind in inds:
            kept_terms.append(vocab_list[ind])
            scores.append(tfidf[ind])
            vocab_file.write(vocab_list[ind] + "\n")
        vocab_file.close()
        print("sampling of the kept terms")
        print(kept_terms[-10:])
        print(scores[-10:])


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python build_vocab.py vocab_size [|Y|] vocab_min")
        sys.exit(0)
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
