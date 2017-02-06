"""
	This script takes in a set of labels sorted by subject ID
	It splits the labels into train/dev/test sets, replacing the code by an index for the code
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

TRAIN_PCT = 0.5
DEV_PCT = 0.25
TEST_PCT = 0.25

def main(Y):

	#select which indices go to which set
	print("finding unique subject ids")
	num_subjects = len(pd.read_csv('../mimicdata/labels_' + str(Y) + '_filtered.csv', usecols=['SUBJECT_ID'], squeeze=True).unique())
	print("finding unique codes and building lookup table")
	codes = pd.read_csv('../mimicdata/labels_' + str(Y) + '_filtered.csv', usecols=['ICD9_CODE'], squeeze=True).unique()
	c_dict = {code: np.where(codes==code)[0][0] for code in codes}

	#create and write headers for train, dev, test
	train_file = open('../mimicdata/labels_' + str(Y) + '_train.csv', 'w')
	dev_file = open('../mimicdata/labels_' + str(Y) + '_dev.csv', 'w')
	test_file = open('../mimicdata/labels_' + str(Y) + '_test.csv', 'w')
	train_file.write(','.join(['SUBJECT_ID', 'ICD9_CODE']) + "\n")
	dev_file.write(','.join(['SUBJECT_ID', 'ICD9_CODE']) + "\n")
	test_file.write(','.join(['SUBJECT_ID', 'ICD9_CODE']) + "\n")

	with open('../mimicdata/labels_' + str(Y) + '_filtered.csv', 'r') as notesfile:
		reader = csv.reader(notesfile)
		next(reader)
		i = 0
		subj_seen = 0
		cur_subj = 0
		for row in reader:
			#write to file according to train/dev/test split
			if i % 10000 == 0:
				print(str(i) + " read")

			subj_id = int(row[0])
			if subj_id != cur_subj:
				subj_seen += 1
				cur_subj = subj_id

			code_ind = c_dict[int(row[1])]

			if subj_seen < num_subjects * TRAIN_PCT:
				train_file.write(','.join([row[0], str(code_ind)]) + "\n")
			elif subj_seen >= num_subjects * TRAIN_PCT and subj_seen < num_subjects * (TRAIN_PCT + DEV_PCT):
				dev_file.write(','.join([row[0], str(code_ind)]) + "\n")
			else:
				test_file.write(','.join([row[0], str(code_ind)]) + "\n")
			i += 1

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(sys.argv[1])	