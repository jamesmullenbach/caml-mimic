"""
    Use these methods to load data
"""
from collections import defaultdict
import csv
import numpy as np

from constants import *

#yield some tensors. file should hold data sorted by sequence length, for batching
def data_generator(filename, batch_size, Y):
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_insts = []
        cur_labels = []
        cur_length = 0
        for row in r:
            #find the next batch_size instances with the same length
            text = row[1]
            length = int(row[3])
            if length > cur_length:
                if len(cur_insts) > 0:
                    #create the tensors
                    yield np.array(cur_insts), np.array(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[2].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])
                #reset length
                cur_length = length
            else:
                if len(cur_insts) == batch_size:
                    #create the tensors
                    yield np.array(cur_insts), np.array(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[2].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])

def load_lookups(vocab_size, Y, vocab_min):
    v_dict = defaultdict(str)
    c_dict = defaultdict(str)
    with open("%s/vocab_lookup_%s_%s_%s.txt" % (DATA_DIR, str(vocab_size), str(Y), str(vocab_min)), 'r') as vocabfile:
        vr = csv.reader(vocabfile)
        next(vr)
        for i,row in enumerate(vr):
            if len(row) > 1:
                v_dict[int(row[0])] = row[1]
            else:
                v_dict[i] = row[0]

    with open(DATA_DIR + "/label_lookup_" + str(Y) + '.csv', 'r') as labelfile:
        lr = csv.reader(labelfile)
        next(lr)
        for row in lr:
            c_dict[int(row[0])] = row[1]
    return (v_dict, c_dict)  

def load_all_data(Y, dataset="single", notebook=True):
    """
        Loads data into 4 tensors (for the fixed-length model case)
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
