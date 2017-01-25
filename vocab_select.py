"""
	This script takes in a vocabulary file and a label set size
	It reads the training, dev, and test sets for the label set size and rewrites
	the text as sequences of vocab indices, ignoring unknown words
"""
import csv
import os
import sys

def main(Y, vocab_min):
	#load in vocab
	print("loading vocab")
	v_list = []
	with open('../mimicdata/vocab_' + str(vocab_min) + '.txt', 'r') as vocabfile:
		for line in vocabfile:
			v_list.append(line.rstrip())

	vocab = set(v_list)
	v_dict = {w: v_list.index(w) for w in v_list}
	vocab_filter(vocab, v_dict, 'train')
	vocab_filter(vocab, v_dict, 'dev')
	vocab_filter(vocab, v_dict, 'test')

def vocab_filter(v_set, v_dict, dataset):
	"""
		Take in the vocab set for existence checking, and the dict for indexing
	"""
	print("filtering " + dataset + " dataset", end="")
	with open('../mimicdata/notes_10_' + dataset + '.csv', 'r') as infile:
		with open('../mimicdata/notes_10_' + dataset + '_ind.csv', 'w') as outfile:
			reader = csv.reader(infile)
			#don't need chart time anymore
			outfile.write(','.join(['SUBJECT_ID', 'TEXT']))
			i = 0
			for row in reader:
				if i % 10000 == 0:
					print(str(i))
				text = row[2]
				filtered = ' '.join([str(v_dict[w]) for w in text.split() if w in v_set])
				outfile.write(','.join([row[0], filtered]) + "\n")
				i += 1

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min]"))
		sys.exit(0)
	main(sys.argv[1], sys.argv[2])	