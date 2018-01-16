import datasets
from constants import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import csv

from tqdm import tqdm

def vocab_index_descriptions(vocab_file, vectors_file):
    #load lookups
    ind2w, w2ind = datasets.load_vocab_dict(vocab_file)
    desc_dict = datasets.load_code_descriptions()
        
    tokenizer = RegexpTokenizer(r'\w+')

    with open(vectors_file, 'w') as of:
        w = csv.writer(of, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in tqdm(desc_dict.iteritems()):
            #same preprocessing steps as in get_discharge_summaries
            tokens = [t.lower() for t in tokenizer.tokenize(desc.decode('latin-1')) if not t.isnumeric()]
            inds = [w2ind[t] if t in w2ind.keys() else len(w2ind)+1 for t in tokens]
            w.writerow([code] + [str(i) for i in inds])
