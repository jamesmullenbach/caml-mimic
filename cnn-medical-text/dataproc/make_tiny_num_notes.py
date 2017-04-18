"""
    Select a random set of num_notes's from attn-formatted data to make a sample dataset from
"""
import csv
import os
import random
import sys

import pandas as pd

from constants import *

def main(infilename, outfilename, N):
    sample(infilename, outfilename, N)

def sample(infilename, outfilename, N):
    df = pd.read_csv(infilename, dtype={"NUM_NOTES": int})
    sample = random.sample(df["NUM_NOTES"].unique(), N)
    df = df[df["NUM_NOTES"].isin(sample)]
    df.to_csv(outfilename, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python " + str(os.path.basename(__file__) + " [infilename] [outfilename] [N]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
