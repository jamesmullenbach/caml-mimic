"""
	When group_and_sort is run, note entries with null CHARTTIME values are dropped.
	This means that a small number of subjects are completely dropped
	This script updates the patients list and labels list to drop the same subjects that were dropped
"""
import csv
import os
import sys

import pandas as pd

from constants import DATA_DIR

def main(Y):

	note_subjects = pd.read_csv('%s/notes_%s_sorted.csv' % (DATA_DIR, Y), usecols=['SUBJECT_ID'], squeeze=True).unique()
	print(len(note_subjects))

	print("Filtering patients list")
	with open('%s/patients_%s.csv' % (DATA_DIR, Y), 'r') as patientfile:
		with open('%s/patients_%s_filtered.csv' % (DATA_DIR, Y), 'w') as filteredfile:
			reader = csv.reader(patientfile)
			next(reader)
			filteredfile.write('SUBJECT_ID\n')
			for row in reader:
				subj_id = int(row[0])
				if subj_id in note_subjects:
					filteredfile.write(str(subj_id) + '\n')

	print("Filtering labels list")
	with open('%s/labels_%s_w_times.csv' % (DATA_DIR, Y), 'r') as labelfile:
		with open('%s/labels_%s_filtered.csv' % (DATA_DIR, Y), 'w') as filteredfile:
			reader = csv.reader(labelfile)
			next(reader)
			filteredfile.write('SUBJECT_ID,HADM_ID,ICD9_CODE,ADMITTIME,DISCHTIME\n')
			for row in reader:
				subj_id = int(row[0])
				if subj_id in note_subjects:
					filteredfile.write(','.join(row) + '\n')

		

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])	
