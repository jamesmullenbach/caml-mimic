"""
	Reads in a dataset w/ labels and sorts by text word length
	Also removes instances where the text is empty
"""

import csv
import os
import sys

import pandas as pd

from constants import DATA_DIR

def main(filename):
	sort(filename)

def sort(filename):
	print("reading data")
	names = ["SUBJECT_ID", "TEXT", "LABELS"]
	df = pd.read_csv(filename, names=names)

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

	df.to_csv(filename.replace('.csv', '_sorted.csv'), index=False)
	
	return df
	

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [filename]"))
		sys.exit(0)
	main(sys.argv[1])
