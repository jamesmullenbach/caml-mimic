"""
    Use the vocabulary to load a matrix of pre-trained word vectors
"""
import csv
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
    #v_dict starts at 1 (saving 0 for the pad character), but gensim word vectors starts at 0
    W = np.zeros((len(v_dict)+1, len(wv.word_vec(wv.index2word[0])) ))
    words = [PAD_CHAR]
    W[0][:] = np.zeros(len(wv.word_vec(wv.index2word[0])))
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
        #write pad token
        #pad_line = PAD_TOKEN + " " + " ".join(["0" for i in range(EMBEDDING_SIZE)])
        #o.write(pad_line + "\n")
        for i in range(len(W)):
            line = [words[i]]
            line.extend([str(d) for d in W[i]])
            o.write(" ".join(line) + "\n")

def load_embeddings(Y, data):
    embed_file = os.path.join(DATA_DIR, "raw.embed") if data == "raw" else os.path.join(DATA_DIR, "processed_%s.embed" % (str(Y)))
    W = []
    with open(embed_file) as ef:
        r = csv.reader(ef, delimiter=" ")
        for line in r:
            W.append(np.array(line[1:]).astype(np.float))
    W = np.array(W)
    return W

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_wvs.py vocab_size Y [raw|processed] vocab_min")
        sys.exit(0)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
