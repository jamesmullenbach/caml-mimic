"""
	This script takes in the file containing (SUBJECT_ID, CHARTTIME, TEXT) triples
	and sorts them by subject id and charttime using pandas. Because SQL was really determined to run out of memory
"""
import os
import pandas as pd
import sys

from constants import DATA_DIR

def main(Y):
	print("reading data")
	df = pd.read_csv('%s/notes_%s.csv' % (DATA_DIR, Y),
					 parse_dates=['CHARTTIME'],
					 infer_datetime_format=True)

	print("dropping if no CHARTTIME")
	df = df.dropna(subset=['CHARTTIME'])

	print("sorting")
	df = df.sort(['SUBJECT_ID', 'CHARTTIME'])

	print("writing output to %s/notes_%s_sorted.csv" % (DATA_DIR, Y))
	df.to_csv('%s/notes_%s_sorted.csv' % (DATA_DIR, Y), index=False)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])
