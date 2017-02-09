"""
	This script takes in the file containing (SUBJECT_ID, CHARTTIME, TEXT) triples
	and sorts them by subject id and charttime using pandas. Because SQL was really determined to run out of memory
"""
import os
import pandas as pd
import sys

def main(Y):
	print("reading data")
	df = pd.read_csv('../mimicdata/notes_' + str(Y) + '.csv',
					 parse_dates=['CHARTTIME'],
					 infer_datetime_format=True)

	print("dropping if no CHARTTIME")
	df = df.dropna(subset=['CHARTTIME'])

	print("sorting")
	df = df.sort(['SUBJECT_ID', 'CHARTTIME'])

	print("writing output")
	df.to_csv('../mimicdata/notes_' + str(Y) + '_sorted.csv', index=False)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])
