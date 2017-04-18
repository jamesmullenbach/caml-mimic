"""
    Just does token/type ratios on train/dev/test (output of concat_and_split)
"""
import csv
import os
import sys

from constants import *

def main(Y):
    tr = token_type_ratio("train", Y)
    dv = token_type_ratio("dev", Y)
    te = token_type_ratio("test", Y)
    print("TRAIN tokens, types, ratio: " + str(tr))
    print("DEV tokens, types, ratio: " + str(dv))
    print("TEST tokens, types, ratio: " + str(te))

def token_type_ratio(split, Y):
    with open("%s/notes_%s_%s_split.csv" % (DATA_DIR, Y, split)) as infile:
        r = csv.reader(infile)
        #header
        next(r)
        i = 0
        tokens = 0
        vocab = set()
        for row in r:
            if i % 10000 == 0:
                print(str(i) + " done")
            text = row[2].split()
            tokens += len(text)
            for w in text:
                vocab.add(w)
            i += 1
    return tokens, len(vocab), tokens / float(len(vocab))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
        sys.exit(0)
    main(int(sys.argv[1]))
