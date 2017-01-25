"""
	When group_and_sort is run, note entries with null CHARTTIME values are dropped.
	This means that a small number of subjects are completely dropped
	This script updates the patients list and labels list to drop the same subjects that were dropped
"""
import csv
import os
import sys

import pandas as pd

def main(Y):

	note_subjects = pd.read_csv('../mimicdata/notes_' + str(Y) + '_sorted.csv', usecols=['SUBJECT_ID'], squeeze=True).unique()

	print("Filtering patients list")
	with open('../mimicdata/patients_' + str(Y) + '.csv', 'r') as patientfile:
		with open('../mimicdata/patients_' + str(Y) + '_filtered.csv', 'w') as filteredfile:
			reader = csv.reader(patientfile)
			next(reader)
			filteredfile.write('SUBJECT_ID\n')
			for row in reader:
				subj_id = int(row[0])
				if subj_id in note_subjects:
					filteredfile.write(str(subj_id) + '\n')

	print("Filtering labels list")
	with open('../mimicdata/labels_' + str(Y) + '.csv', 'r') as labelfile:
		with open('../mimicdata/labels_' + str(Y) + '_filtered.csv', 'w') as filteredfile:
			reader = csv.reader(labelfile)
			next(reader)
			filteredfile.write('SUBJECT_ID,ICD9_CODE\n')
			for row in reader:
				subj_id = int(row[0])
				if subj_id in note_subjects:
					filteredfile.write(','.join([str(subj_id), row[1]]) + '\n')

		

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])	