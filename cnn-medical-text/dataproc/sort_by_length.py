"""
    Reads in a dataset (train/dev/test, full or single) w/ labels and sorts by text word length
    Also removes instances where the text is empty
"""

import csv
import os
import sys

import pandas as pd

from constants import DATA_DIR

def main(infilename, outfilename, have_length):
    have_length = True if have_length == "1" else False
    sort(infilename, outfilename, have_length)

def sort(infilename, outfilename, have_length):
    print("reading data")    
    names = ["SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"]
    if have_length:
        names.append("length")
    df = pd.read_csv(infilename, names=names)

    #drop rows with null text
    print(len(df))
    df = df.dropna(subset=["TEXT"])
    #drop header
    df = df[df["TEXT"] != "TEXT"]
    print(len(df))

    print("adding seq length column")
    df['length'] = df.apply(lambda row: len(str(row["TEXT"]).split()), axis=1)
    print("sorting by seq length")
    df = df.sort(['length'])

    #if we had length, replace old file with sorted one. else make new one
    df.to_csv(outfilename, index=False)
    
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [infilename] [outfilename] [have_length (0 or 1)]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
