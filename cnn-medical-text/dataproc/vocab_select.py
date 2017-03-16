"""
    This script takes in a vocabulary file and a label set size
    It reads the training, dev, and test sets for the label set size and rewrites
    the text as sequences of vocab indices, ignoring unknown words
    It also rewrites the labels as indices
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

from constants import DATA_DIR

def main(Y, vocab_min):
    #load in vocab
    print("loading vocab")
    v_list = []
    with open('%s/vocab_%s_%d.txt' % (DATA_DIR, Y, vocab_min), 'r') as vocabfile:
        for line in vocabfile:
            v_list.append(line.rstrip())

    codes = pd.read_csv('%s/labels_%s_filtered.csv' % (DATA_DIR, Y), usecols=['ICD9_CODE'], squeeze=True).unique()
    c_dict = {str(code): str(np.where(codes==code)[0][0]) for code in codes}

    vocab = set(v_list)
    v_dict = {w: v_list.index(w) for w in v_list}
    vocab_filter(Y, vocab, v_dict, c_dict, 'train')
    vocab_filter(Y, vocab, v_dict, c_dict, 'dev')
    vocab_filter(Y, vocab, v_dict, c_dict, 'test')

    #write vocab/label lookups to files to make debugging downstream easier
    print("writing lookup tables")
    with open("%s/vocab_lookup_%s_%s.csv" % (DATA_DIR, Y, vocab_min), 'w') as vocabfile:
        vocabfile.write(','.join(['ID', 'WORD']) + "\n")
        for word, ind in iter(v_dict.items()):
            vocabfile.write(','.join([str(ind), word]) + "\n")

    with open("%s/label_lookup_%s.csv" % (DATA_DIR, Y), 'w') as labelfile:
        labelfile.write(','.join(['ID', 'LABEL']) + "\n")
        for label, ind in iter(c_dict.items()):
            labelfile.write(','.join([ind, label]) + "\n")


def vocab_filter(Y, v_set, v_dict, c_dict, dataset):
    """
        Take in the vocab set for existence checking, and the dict for indexing
    """
    print("filtering " + dataset + " dataset")
    with open('%s/notes_%s_%s_labeled.csv' % (DATA_DIR, Y, dataset), 'r') as infile:
        with open('%s/notes_%s_%s_full.csv' % (DATA_DIR, Y, dataset), 'w') as outfile:
            reader = csv.reader(infile)
            next(reader)
            #don't need chart time anymore
            outfile.write(','.join(['SUBJECT_ID', 'TEXT', 'LABELS']) + "\n")
            i = 0
            for row in reader:
                if i % 10000 == 0:
                    print(str(i))

                #indexify the text
                text = row[1]
                filtered = ' '.join([str(v_dict[w]) for w in text.split() if w in v_set])

                #indexify the labels
                labels = row[2:]
                filtered_labels = [c_dict[label] for label in labels]
                label_str = ';'.join(filtered_labels)

                #write output
                outline = [row[0], filtered, label_str]
                outfile.write(','.join(outline) + "\n")
                i += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])    
