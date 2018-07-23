"""
    Pre-computes the vocab-indexed version of each code description
"""
import datasets
from constants import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import csv

from tqdm import tqdm

def vocab_index_descriptions(vocab_file, vectors_file):
    #load lookups
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.strip()
            if line != '':
                vocab.add(line)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    desc_dict = datasets.load_code_descriptions()
        
    tokenizer = RegexpTokenizer(r'\w+')

    with open(vectors_file, 'w') as of:
        w = csv.writer(of, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in tqdm(desc_dict.items()):
            #same preprocessing steps as in get_discharge_summaries
            tokens = [t.lower() for t in tokenizer.tokenize(desc) if not t.isnumeric()]
            inds = [w2ind[t] if t in w2ind.keys() else len(w2ind)+1 for t in tokens]
            w.writerow([code] + [str(i) for i in inds])
