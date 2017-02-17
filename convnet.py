"""
	Runs a ConvNet over the data to predict ICD-9 diagnosis codes
	options for single window or multi-window
	
	Framework: Keras
"""
from collections import defaultdict
import csv
import os
import numpy as np
import sys

import evaluation

from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence

#Embedding constants
VOCAB_SIZE = 40000
EMBEDDING_SIZE = 64
DROPOUT_EMBED = 0.5
DROPOUT_DENSE = 0.5

#Convolution constants
FILTER_SIZE = 5
STRIDE = 1
CONV_DIM_FACTOR = 10

#training constants
BATCH_SIZE = 32
NUM_EPOCHS = 5

#others
MAX_LENGTH = 400

def main(Y, vocab_min, model_path, dataset):
	"""
		main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
	"""
	old_version = get_version()

	model = build_model(Y, old_version=old_version)
	#model = build_model_multiwindow(Y, 3, 5, 1, old_version=old_version)

	(X_tr, Y_tr), (X_dv, Y_dv) = load_data(Y, dataset, notebook=False)

	print("padding sequences")
	X_tr = sequence.pad_sequences(X_tr, maxlen=MAX_LENGTH, padding='post', truncating='post')
	X_dv = sequence.pad_sequences(X_dv, maxlen=MAX_LENGTH, padding='post', truncating='post')

	print("getting lookups")
	v_dict, c_dict = load_lookups(Y, vocab_min)

	#shuffle before just in case
	np.random.seed(1337)
	np.random.shuffle(X_tr)
	np.random.seed(1337)
	np.random.shuffle(Y_tr)
	print("##### OTHER PARAMS ######")
	print("# Sequence length: %d, conv window: %d, dropout: %f" % (MAX_LENGTH, CONV_DIM_FACTOR*Y, DROPOUT_DENSE))
	print("training model")
	hist = train(model, X_tr, Y_tr, X_dv, Y_dv)
	print("evaluating on dev")
	preds,metrics,fpr, tpr = evaluate(model, X_dv, Y_dv)
	print("[MACRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"])
	print("[MICRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])
	print

	print("sanity check on train")
	preds_t,metrics_t, fpr_t, tpr_t = evaluate(model, X_tr, Y_tr)
	print("[MACRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics_t["acc"], metrics_t["prec"], metrics_t["rec"], metrics_t["f1"], metrics_t["auc"])
	print("[MICRO] accuracy, precision, recall, f-measure, AUC")
	print(metrics_t["acc_micro"], metrics_t["prec_micro"], metrics_t["rec_micro"], metrics_t["f1_micro"], metrics_t["auc_micro"])

	print("ROC AUC measures")
	preds = np.array(preds)
	Y_dv = np.array(Y_dv)
	roc_auc = {"macro": metrics["auc"], "micro": metrics["auc_micro"]}
	roc_auc.update({i: metrics["auc_%d" % i] for i in range(Y)})
	if old_version:
		write_auc(fpr, tpr, roc_auc, Y)
	else:
		plot_auc(fpr, tpr, roc_auc)

	print("writing predictions and labels")
	preds = [[i for i in range(len(p)) if p[i] == 1] for p in preds]
	preds_t = [[i for i in range(len(p)) if p[i] == 1] for p in preds_t]
	golds = [[i for i in range(len(g)) if g[i] == 1] for g in Y_dv]
	golds_t = [[i for i in range(len(g)) if g[i] == 1] for g in Y_tr]
	write_preds(preds, 'dev.preds')
	write_preds(preds_t, 'train.preds')
	write_preds(golds, 'dev.golds')
	write_preds(golds_t, 'train.golds')

	if not old_version:
		model.save(model_path)

def build_model(Y, old_version=False):
	"""
		Builds the single window CNN model
		params:
			Y: size of the label space
		returns:
			model: the CNN model
	"""
	model = Sequential()
	#no input length bc it's not constant
	model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, input_length=MAX_LENGTH))
	model.add(Convolution1D(Y*CONV_DIM_FACTOR, FILTER_SIZE, activation='tanh'))
	if old_version:
		#http://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
		from keras.layers import MaxPooling1D, Reshape
		model.add(MaxPooling1D(pool_length=MAX_LENGTH-FILTER_SIZE+1))
		model.add(Reshape((Y*CONV_DIM_FACTOR,)))
	else:
		from keras.layers.pooling import GlobalMaxPooling1D
		model.add(GlobalMaxPooling1D())
	#model.add(Dense(Y*CONV_DIM_FACTOR/2))
	#model.add(Dropout(DROPOUT_DENSE))
	model.add(Dense(Y))
	model.add(Dropout(DROPOUT_DENSE))
	model.add(Activation('sigmoid'))
	model.compile(optimizer='rmsprop', loss='binary_crossentropy')
	print(model.summary())
	return model

def build_model_multiwindow(Y, s, l, step, old_version=False):
	"""
		Builds the multi-window CNN model
		params:
			Y: size of the label space
			s: the smallest filter size
			l: the largest filter size
			step: size difference between consecutive filters
		returns:
			cnn_multi: the model
	"""
	from keras.layers import Input, merge
	from keras.models import Model
	model_input = Input(shape=(MAX_LENGTH,))
	embed = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, input_length=MAX_LENGTH)(model_input)

	convs = []
	pools = []
	if old_version:
		reshapes = []
	for i,sz in enumerate(range(s, l+1, step)):
		convs.append(Convolution1D(Y*CONV_DIM_FACTOR, sz, activation='tanh')(embed))
		if old_version:
			#http://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
			from keras.layers import MaxPooling1D, Reshape
			pools.add(MaxPooling1D(pool_length=MAX_LENGTH-FILTER_SIZE+1))(convs[i])
			reshapes.add(Reshape((Y*CONV_DIM_FACTOR,)))(pools[i])
		else:
			from keras.layers.pooling import GlobalMaxPooling1D
			pools.add(GlobalMaxPooling1D())(convs[i])

	if old_version:
		merged = merge(reshapes, mode='concat', concat_axis=1)
	else:
		merged = merge(pools, mode='concat', concat_axis=1) 

	dense = Dense(CONV_DIM_FACTOR*Y)(merged)
	dropout = Dropout(DROPOUT_DENSE)(dense)
	activation = Activation('sigmoid')(dropout)
	cnn_multi = Model(input=model_input, output=activation)
	
	cnn_multi.compile(optimizer='rmsprop', loss='binary_crossentropy')
	print(model.summary())
	return cnn_multi

def train(model, X_tr, Y_tr, X_dv, Y_dv):
	#fit the model, validating with dev data
	hist = model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_data=(X_dv, Y_dv))
	return hist

def evaluate(model, X_dv, Y_dv):
	#predict, threshold, and get (macro) metrics
	preds = model.predict(X_dv)
	preds[preds >= 0.5] = 1
	preds[preds < 0.5] = 0

	metrics, fpr, tpr = evaluation.all_metrics(preds, Y_dv)
	return preds, metrics, fpr, tpr

def plot_auc(fpr, tpr, roc_auc):
	#plot the AUC values for current visualization
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(fpr["micro"], tpr["micro"], label='micro ROC (area={0:0.3f})'.format(roc_auc["micro"]))
	plt.plot(fpr["macro"], tpr["macro"], label='macro ROC (area={0:0.3f})'.format(roc_auc["macro"]))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")
	plt.legend(loc="lower right")
	plt.show()

def write_auc(fpr, tpr, roc_auc, Y):
	#write the AUC values for later visualization
	with open("auc_" + str(Y) + ".csv", 'w') as outfile:
		outfile.write(','.join(['label', 'measure', 'values']) + "\n")
		for label in fpr.keys():
			fpr_line = [str(label), 'fpr']
			fpr_line.extend([str(v) for v in fpr[label]])
			outfile.write(','.join(fpr_line) + "\n")
	
			tpr_line = [str(label), 'tpr']
			tpr_line.extend([str(v) for v in tpr[label]])
			outfile.write(','.join(tpr_line) + "\n")

			auc_line = [str(label), 'auc', str(roc_auc[label])]
			outfile.write(','.join(auc_line) + "\n")

def get_version():
	#use the older version of keras when not running on my pc
	import socket
	return ("james" not in socket.gethostname())

def write_preds(preds, filename):
	#write predictions to a csv
	with open(filename, 'w') as outfile:
		for p in preds:
			outfile.write(','.join([str(p_i) for p_i in p]) + "\n")	

def load_data(Y, dataset="single", notebook=True):
	"""
		params:
			Y: size of label space
			dataset: "single" for the one-note-per-subject data, or "full" for all notes
			notebook: True if running in jupyter notebook (for printing)
		return:
			X_tr: list of lists of words. basically as formatted in the csv's already
			Y_tr: list of lists of labels. like the log_reg labels, but with "duplicates"
			X_dv: same as X_tr but dev data
			Y_dv: same as Y_tr but dev data
	"""
	X_tr, Y_tr, X_dv, Y_dv = [], [], [], []

	notes_filename = '../mimicdata/notes_' + str(Y) + '_train_' + dataset + '.csv'
	print("Processing train")
	i = 0
	with open(notes_filename, 'r') as notesfile:
		#go thru the notes file
		#an instance is literally just the array of words, and array of labels (turned into indicator array)
		if i % 10000 == 0:
			if notebook:
				print( ".",)
			else:
				print( str(i) + " done")
		note_reader = csv.reader(notesfile)

		next(note_reader)

		for row in note_reader:
			text = row[1]
			labels = row[2:]
			x_vec = [int(w) for w in text.split()]
			label_ints = [int(l) for l in labels]
			y_vec = [1 if i in label_ints else 0 for i in range(Y)]
			X_tr.append(x_vec)
			Y_tr.append(y_vec)
			i += 1
	print
	
	notes_filename = notes_filename.replace('train', 'dev')
	print("Processing dev")
	i = 0
	with open(notes_filename, 'r') as notesfile:
		if i % 10000 == 0:
			if notebook:
				print(".",)
			else:
				print(str(i) + " done")
		note_reader = csv.reader(notesfile)
		next(note_reader)
		for row in note_reader:
			text = row[1]
			labels = row[2:]
			x_vec = [int(w) for w in text.split()]
			label_ints = [int(l) for l in labels]
			y_vec = [1 if i in label_ints else 0 for i in range(Y)]
			X_dv.append(x_vec)
			Y_dv.append(y_vec)
			i += 1
	print
	return (np.array(X_tr), np.array(Y_tr)), (np.array(X_dv), np.array(Y_dv))

def load_lookups(Y, vocab_min):
	v_dict = defaultdict(str)
	c_dict = defaultdict(str)
	with open('../mimicdata/vocab_lookup_' + str(vocab_min) + '.csv', 'r') as vocabfile:
	    vr = csv.reader(vocabfile)
	    next(vr)
	    for row in vr:
	        v_dict[int(row[0])] = row[1]

	with open('../mimicdata/label_lookup_' + str(Y) + '.csv', 'r') as labelfile:
	    lr = csv.reader(labelfile)
	    next(lr)
	    for row in lr:
	        c_dict[int(row[0])] = row[1]
	return (v_dict, c_dict)	

if __name__ == "__main__":
	#just take in the label set size
	if len(sys.argv) < 4:
		print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min] [model_path] [dataset (single or full)]"))
		sys.exit(0)
	main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])

