"""
    Takes in data sorted by subject id and time and transformed from words to indices
        (output of vocab_select.py)
    Sorts admissions by number of notes for doc-lstm batching in attn model
"""
import os
import sys

from constants import *

import pandas as pd

def main(Y, dataset):
    filename = "%s/notes_%s_%s_full_indices.csv" % (DATA_DIR, Y, dataset)
    
    df = pd.read_csv(filename)
    #get counts
    df["NUM_NOTES"] = df.groupby("HADM_ID")["HADM_ID"].transform("count")
    df = df.sort(["NUM_NOTES", "HADM_ID"])

    df.to_csv("%s/notes_%s_%s_attn.csv" % (DATA_DIR, Y, dataset), index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [train/dev/test]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
