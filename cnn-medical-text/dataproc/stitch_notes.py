"""
    Stitch together notes for a HADM_ID using n 0's
"""
import csv
import os
import sys

from constants import *

def main(vocab_size, Y, vocab_min, num_zeros):
    stitch(vocab_size, Y, vocab_min, num_zeros, "train")
    stitch(vocab_size, Y, vocab_min, num_zeros, "dev")
    stitch(vocab_size, Y, vocab_min, num_zeros, "test")

def stitch(vocab_size, Y, vocab_min, num_zeros, dataset):
    print("processing %s" % (dataset))
    filename = "%s/notes_%s_%s_full_%s.csv" % (DATA_DIR, Y, dataset, vocab_size)
    with open(filename, 'r') as notesfile:
        with open(filename.replace(".csv", "_stitched.csv"), 'w') as outfile:
            r = csv.reader(notesfile)
            header = next(r)
            outfile.write(",".join(header) + "\n")
            
            first = next(r)
            stitched = first[2]
            cur_hadm = int(first[1])
            cur_subj = first[0]
            cur_labels = first[3]
            i = 0
            for line in r:
                if i % 10000 == 0:
                    print("%d done" % (i))
                hadm = int(line[1])
                if hadm == cur_hadm:
                    #pad and add the new note
                    stitched += " " + " ".join(['0' for _ in range(int(num_zeros))])
                    stitched += " " + line[2]
                else:
                    if cur_hadm != 0:
                        outfile.write(",".join([cur_subj, str(cur_hadm), stitched, cur_labels]) + "\n")
                    stitched = line[2]
                    cur_hadm = hadm
                    cur_subj = line[0]
                    cur_labels = line[3]
                i += 1

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("usage: python " + str(os.path.basename(__file__) + " [vocab_size] [|Y|] [vocab_min] [num_zeros]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
