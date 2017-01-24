"""
	This script takes in a set of vocab words and a set of notes.
	It filters the text to leave only the vocab words and divides the data into train/dev/test sets
"""
import csv
import os
import sys

import pandas as pd

TRAIN_PCT = 0.5
DEV_PCT = 0.25
TEST_PCT = 0.25

def main(Y, vocab_min):
	#load in vocab
	print("loading vocab")
	vocab = set([])
	with open('../mimicdata/vocab_' + str(vocab_min) + '.txt', 'r') as vocabfile:
		for line in vocabfile:
			vocab.add(line.rstrip())

	#select which indices go to which set
	print("finding unique subject ids")
	num_subjects = len(pd.read_csv('../mimicdata/notes_' + str(Y) + '_sorted.csv', usecols=['SUBJECT_ID'], squeeze=True).unique())

	#create and write headers for train, dev, test
	train_file = open('../mimicdata/notes_' + str(Y) + '_train.csv', 'w')
	dev_file = open('../mimicdata/notes_' + str(Y) + '_dev.csv', 'w')
	test_file = open('../mimicdata/notes_' + str(Y) + '_test.csv', 'w')
	train_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")
	dev_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")
	test_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")

	with open('../mimicdata/notes_' + str(Y) + '_sorted.csv', 'r') as notesfile:
		reader = csv.reader(notesfile)
		next(reader)
		i = 0
		subj_seen = 0
		cur_subj = 0
		for row in reader:
			#filter text, write to file according to train/dev/test split
			if i % 10000 == 0:
				print(str(i) + " read")
			text = row[3]
			filtered = ' '.join([w for w in text.split() if w in vocab])

			subj_id = int(row[1])
			if subj_id != cur_subj:
				subj_seen += 1
				cur_subj = subj_id

			if subj_id < num_subjects * TRAIN_PCT:
				train_file.write(','.join([row[1], row[2], filtered]) + "\n")
			elif subj_id >= num_subjects * TRAIN_PCT and subj_id < num_subjects * (TRAIN_PCT + DEV_PCT):
				dev_file.write(','.join([row[1], row[2], filtered]) + "\n")
			else:
				test_file.write(','.join([row[1], row[2], filtered]) + "\n")
			i += 1

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min]"))
		sys.exit(0)
	main(sys.argv[1], sys.argv[2])	