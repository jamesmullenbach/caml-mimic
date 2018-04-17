"""
    This script reads a sorted training dataset and builds a vocabulary of terms of given size
    Output: txt file with vocab words
    Drops any token not appearing in at least vocab_min notes

    This script could probably be replaced by using sklearn's CountVectorizer to build a vocab
"""
import csv
import numpy as np
import operator

from collections import defaultdict
from scipy.sparse import csr_matrix

from constants import DATA_DIR, MIMIC_3_DIR

def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        #header
        next(reader)

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
        vocab_list = np.array([word for word,ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        #1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
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

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")

