"""
	Reads in train, dev, test data
	For each distinct (X,Y) pair, select the longest note for a patient-admission
	Output to a new file
"""

import csv
import os
import sys

def main(Y):
	longest_note(Y, 'train')
	longest_note(Y, 'dev')
	longest_note(Y, 'test')

def longest_note(Y, dataset):
	print("selecting longest note for " + dataset + " dataset")
	with open('../mimicdata/notes_' + str(Y) + '_' + dataset + "_final.csv", 'r') as f:
		with open('../mimicdata/notes_' + str(Y) + '_' + dataset + '_single.csv', 'w') as outfile:
			r = csv.reader(f)
			next(r)

			outfile.write(','.join(['SUBJECT_ID', 'TEXT', 'LABELS']) + "\n")

			cur_subj = 0
			cur_labels = set([])
			max_length = 0
			longest_note = ""

			i = 0
			for row in r:
				if i % 10000 == 0:
					print(str(i))
				#do stuff
				subj = int(row[0])
				length = len(row[1].split())
				if length > max_length:
					max_length = length
					longest_note = row[1]
				label_set = set([int(l) for l in row[2].split(';')])

				#if we get to another distinct (X,Y) pair, write
				if subj != cur_subj or cur_labels != label_set:
					# reset variables, write the longest note out
					if cur_subj != 0:
						outline = [str(cur_subj), longest_note]
						outline.extend([str(l) for l in cur_labels])
						outfile.write(','.join(outline) + "\n")
					cur_subj = subj
					cur_labels = label_set
					max_length = 0
					longest_note = ""
				i += 1

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(int(sys.argv[1]))
