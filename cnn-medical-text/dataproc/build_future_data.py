"""
    Takes in data sorted by subject id, time, and concatenated together so that there's one doc for each admission
        (output of stitch_notes.py)
    Filters down to the subjects with more than one admission
    Creates two datasets: one with just (labels_t, labels_t+1), and one with (text, labels_t, labels_t+1)
"""
import os
import sys

from constants import *

import pandas as pd

def main(Y, dataset):
    filename = "%s/notes_%s_%s.csv" % (DATA_DIR, Y, dataset)
    
    df = pd.read_csv(filename)
    #get counts
    df["NUM_HADM"] = df.groupby("SUBJECT_ID")["HADM_ID"].transform("count")
    #filter out single-admission rows
    df = df[df["NUM_HADM"] > 1]
    df = df.sort(["SUBJECT_ID"])

    M = df.as_matrix()
    build_datasets(M, Y, dataset)

def build_datasets(M, Y, dataset):
    with open("%s/future_codes_%s_%s.csv" % (DATA_DIR, Y, dataset), 'w') as codes_file:
        with open("%s/future_codes_text_%s_%s.csv" % (DATA_DIR, Y, dataset), 'w') as text_file:
            codes_file.write(",".join(["CUR_CODES", "NEXT_CODES"]) + "\n")
            text_file.write(",".join(["CUR_TEXT", "CUR_CODES", "NEXT_CODES"]) + "\n")
            cur_subj = M[0][0]
            cur_text = M[0][2]
            cur_codes = M[0][3]
            for i in range(1, M.shape[0]):
                next_subj = M[i][0]
                if next_subj != cur_subj:
                    #don't write anything
                    cur_subj = next_subj
                    cur_text = M[i][2]
                    cur_codes = M[i][3]
                else:
                    #write
                    next_codes = M[i][3]
                    codes_file.write(",".join([cur_codes, next_codes]) + "\n")
                    text_file.write(",".join([cur_text, cur_codes, next_codes]) + "\n")
                    cur_subj = next_subj
                    cur_text = M[i][2]
                    cur_codes = M[i][3]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [train/dev/test]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
