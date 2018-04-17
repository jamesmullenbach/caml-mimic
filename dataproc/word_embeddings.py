"""
    Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import gensim.models.word2vec as w2v
import csv

from constants import *

class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())

def word_embeddings(Y, notes_file, embedding_size, min_count, n_iter):
    modelname = "processed_%s.w2v" % (Y)
    sentences = ProcessedIter(Y, notes_file)

    model = w2v.Word2Vec(size=embedding_size, min_count=min_count, workers=4, iter=n_iter)
    print("building word2vec vocab on %s..." % (notes_file))
    
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

