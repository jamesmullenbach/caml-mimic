"""
    Simply takes in a raw (NOTEEVENTS) or processed (output of get_notes) file and outputs
    only the text into a .txt
"""
import csv
import os
import sys

from constants import *

def main(Y, data):
    filename = os.path.join(DATA_DIR,"NOTEEVENTS.csv") if data == "raw" else os.path.join(DATA_DIR,"notes_%d.csv") % (Y)
    outname = os.path.join(DATA_DIR, "raw_words.txt") if data == "raw" else os.path.join(DATA_DIR, "notes_%d_words.txt") % (Y)
    with open(filename, 'r') as f:
        with open(outname, 'w') as o:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                o.write(row[2] + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python to_plain_text.py Y [raw|processed]")
        sys.exit(0)
    main(int(sys.argv[1]), sys.argv[2])
