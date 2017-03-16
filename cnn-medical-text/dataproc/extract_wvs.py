"""
    Use the vocabulary to load a matrix of pre-trained word vectors
"""
import os
import sys

import gensim.models

from constants import *
from datasets import load_lookups

import numpy as np

def main(vocab_size, Y, data, vocab_min):
    wv_file = os.path.join(DATA_DIR, "raw.w2v") if data == "raw" else os.path.join(DATA_DIR, "processed_%d.w2v" % (Y))
    model = gensim.models.Word2Vec.load(wv_file)
    wv = model.wv
    #free up memory
    del model

    v_dict, _ = load_lookups(vocab_size, Y, vocab_min)

    #go through vocab in order
    #find vocab word in wv.index2word, then call wv.word_vec(wv.index2word[i])
    #put results into one big matrix
    W, words = build_matrix(v_dict, wv)

    #smash that save button
    outfile = os.path.join(DATA_DIR, "raw.embed") if data == "raw" else os.path.join(DATA_DIR, "processed_%d.embed" % (Y))
    save_embeddings(W, words, outfile)

def build_matrix(v_dict, wv):
    W = np.zeros((len(v_dict), len(wv.word_vec(wv.index2word[0])) ))
    words = []
    for j,(idx, word) in enumerate(v_dict.iteritems()):
        found = False
        for i in range(len(wv.index2word)):
            if word == wv.index2word[i]:
                W[idx][:] = wv.word_vec(wv.index2word[i])
                words.append(word)
                break
        if j % 100 == 0:
            print(j)
    return W, words

def save_embeddings(W, words, outfile):
    with open(outfile, 'w') as o:
        for i in range(len(W)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_wvs.py vocab_size Y [raw|processed] vocab_min")
        sys.exit(0)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
