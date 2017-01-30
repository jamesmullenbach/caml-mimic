"""
	Runs a ConvNet over the data to predict ICD-9 diagnosis codes
"""
import csv
import os
import numpy as np
import sys

from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape
from keras.layers.convolutional import Convolutional1D
from log_reg import construct_label_lists

def main(Y):
	"""
		main function which sequentially loads the data, builds the model, trains, and evaluates
	"""
	(X_tr, Y_tr), (X_dv, Y_dv) = load_data(Y)

def build_model():
	model = None
	return model

def train(model, X_tr, Y_tr):
	pass

def evaluate(model, X_dv, Y_dv):
	return acc,prec,rec,f1

def load_data(Y):
	"""
		For the convnet, each note will be a separate instance, rather than each subject
		Adapt the methods from log_reg for this
		Read both notes and label files simultaneously to ensure correspondence
		return:
			X_tr: list of lists of words. basically as formatted in the csv's already
			Y_tr: list of lists of labels. like the log_reg labels, but with "duplicates"
			X_dv: same as X_tr but dev data
			Y_dv: same as Y_tr but dev data
	"""
	X_tr, Y_tr, X_dv, Y_dv = [], [], [], []

	notes_filename = '../mimicdata/notes_' + str(Y) + '_train_ind.csv'
	labels_filename = '../mimicdata/labels_' + str(Y) + '_train.csv'
	with open(notes_filename, 'r') as notesfile:
		with open(labels_filename, 'r') as labelsfile:
			#go thru the notes file
			#when you see a new subject id, build its label vector from labels file
			#when you see the same subject id as before, add a copy of the label vector to Y
			cur_subj = 0
			cur_y_vec = [0] * Y
			note_reader = csv.reader(notesfile)
			label_reader = csv.reader(labelsfile)

			next(note_reader)
			next(label_reader)

			for row in note_reader:
				subj_id = int(row[0])
				text = row[1]
				if subj_id != cur_subj:
					#make new label vector from the labelsfile
					cur_y_vec = [0] * Y
					found_next = False
					while not found_next:
						l_row = next(label_reader)
						subj = int(l_row[0])
						if subj == cur_subj:
							code = int(l_row[1])
							cur_y_vec[code] = 1
						else:
							if sum(cur_y_vec) > 0:
								found_next = True
							else:
								print("you fucked up")

				x_vec = [int(w) for w in text.split()]
				X_tr.append(x_vec)
				Y_tr.append(cur_y_vec)

	return (X_tr, Y_tr), (X_dv, Y_dv)



if __name__ == "__main__":
	#just take in the label set size
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
		sys.exit(0)
	main(int(sys.argv[1]))