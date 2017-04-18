"""
    Pre-train embeddings using w2v and varembed models
"""
import gensim.models.word2vec as w2v
import csv
import os
import sys

from constants import *

class RawIterAndProcess(object):

    def __init__(self, Y, filename):
        self.filename = filename
        self.subjects = set([])
        print("getting the list of relevant subjects")
        with open('%s/patients_%s.csv' % (DATA_DIR, Y), 'r') as csvfile:
            patientreader = csv.reader(csvfile)
            next(patientreader)
            for line in patientreader:
                self.subjects.add(int(line[0]))

    def __iter__(self):

        with open(self.filename) as f:
            with open('%s/notes_%s.csv' % (DATA_DIR, Y), 'w') as outfile:
                outfile.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + '\n')
                notereader = csv.reader(csvfile)
                next(notereader)
                i = 0
                for line in notereader:
                    if i % 10000 == 0:
                        print(i)
                    subj = int(line[1])
                    note = unicode(line[10])
                    yield note.split()
                    if subj in self.subjects:
                        tokens = [t.lower() for t in tokenizer.tokenize(note) if not t.isnumeric() and t.lower() not in stop_words]
                        text = '"' + ' '.join(tokens) + '"'
                        if i % 10000 == 0:
                            print(text[:80])
                        outfile.write(','.join([line[1], line[4], text]) + '\n')
                    i += 1

class RawIterNoProcess(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield unicode(row[10]).split()

class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield unicode(row[3]).split()

def main(Y, data, min_count, n_iter):
    if data == "raw":
        filename = "NOTEEVENTS.csv"
        modelname = "raw.w2v"
        should_preproc = raw_input("Want to preprocess as well? (y/n) > ")
        if "y" in should_preproc:
            sentences = RawIterAndProcess(Y, os.path.join(DATA_DIR, filename))
        else:
            sentences = RawIterNoProcess(Y, os.path.join(DATA_DIR, filename))
    else:
        filename = "notes_%s.csv" % (Y)
        modelname = "processed_%s.w2v" % (Y)
        sentences = ProcessedIter(Y, os.path.join(DATA_DIR, filename))
#    for i, sent in enumerate(sentences):
#        if i > 1:
#            break
#        print(sent)
    model = w2v.Word2Vec(size=EMBEDDING_SIZE, min_count=min_count, workers=4, iter=n_iter)
    print("building word2vec vocab...")
    model.build_vocab(sentences)
    print("training...")
    model.train(sentences)
#    for i in range(n_iter):
#        model.train(sentences)
    model.save(modelname)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python word_embeddings.py Y [raw|processed] min_count n_iter")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
