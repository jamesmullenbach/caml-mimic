"""
    Use these methods to load data
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

from constants import *

def attn_generator(filename, batch_size, Y, max_to_generate=sys.maxint):
    #generator for hattn network
    #apparently, there can be some length-0 notes in the attn datasets? so just add a single pad character to those
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_note_set = []
        cur_note_sets = []
        cur_labels = []
        cur_num_notes = 0
        num_generated = 0
        while True:
            if num_generated > max_to_generate:
                print("reached the max that was passed: %d" % max_to_generate)
                #yield np.array(cur_note_sets), np.array(cur_labels)
                break

            cur_note_set = []
            try:
                row = next(r)
            except StopIteration:
                print("done")
                yield np.array(cur_note_sets), np.array(cur_labels)
                break

            cur_hadm = int(row[1])
            text = [int(w) for w in row[2].split()]
            if len(text) == 0:
                print("empty text found")
                text.append(0)
            labels = [int(c) for c in row[3].split(';')]
            #only add up to MAX_NOTES notes
            actual_notes = int(row[4])
            num_notes = min(actual_notes, MAX_NOTES)

            if num_notes != cur_num_notes:
#                if num_notes > 200:
#                    #that's just too many
#                    print("reached instance with > 200 notes. stopping")
#                    yield np.array(cur_note_sets), np.array(cur_labels)
#                    break
                cur_num_notes = num_notes
                #yield what you have so far, then reset
                if len(cur_note_sets) > 0:
                    num_generated += 1
                    yield np.array(cur_note_sets), np.array(cur_labels)
                cur_note_sets = []
                cur_labels = []

            cur_note_set.append(text)
            cur_labels.append([1 if i in labels else 0 for i in range(Y)])

            #add this text and the next num_notes - 1 texts to cur_note_set
            for i in range(actual_notes-1):
                row = next(r)
                #if we have more than MAX_NOTES notes, ignore the rest of them
                if i >= MAX_NOTES - 1:
                    continue
                hadm = int(row[1])
                if hadm != cur_hadm:
                    print("got diff hadm. this shouldn't happen")
                text = [int(w) for w in row[2].split()]
                if len(text) == 0:
                    print("empty text found")
                    text.append(0)
                cur_note_set.append(text)

            #add the note set to the note sets
            cur_note_sets.append(cur_note_set)
            if len(cur_note_sets) == batch_size:
                num_generated += 1
                yield np.array(cur_note_sets), np.array(cur_labels)
                cur_note_sets = []
                cur_labels = [] 

def next_codes_text_generator(filename, batch_size, Y):
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_codes = []
        cur_notes = []
        cur_labels = []
        for row in r:
            text = [int(w) for w in row[0].split()]
            codes = [int(c) for c in row[1].split(';')]
            next_codes = [int(c) for c in row[2].split(';')]
            cur_codes.append(codes)
            cur_notes.append(text)
            cur_labels.append([1 if i in next_codes else 0 for i in range(Y)])
            if len(cur_notes) >= 1:
                yield [cur_codes, cur_notes], cur_labels
                cur_codes = []
                cur_notes = []
                cur_labels = []
        yield [cur_codes, cur_notes], cur_labels

def codes_only_generator(filename, batch_size, Y):
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_insts = []
        cur_labels = []
        for row in r:
            cur_codes = [int(c) for c in row[1].split(';')]
            next_codes = [int(c) for c in row[2].split(';')]
            cur_insts.append(cur_codes)
            cur_labels.append([1 if i in next_codes else 0 for i in range(Y)])
            if len(cur_insts) >= 1:
                yield cur_insts, cur_labels
                cur_insts = []
                cur_labels = []
        #yield cur_insts, cur_labels

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
            text = row[2]
            length = int(row[4])
            if length > cur_length:
                if len(cur_insts) > 0:
                    #create the tensors
                    yield np.array(cur_insts), np.array(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[3].split(';')]
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
                labels = [int(l) for l in row[3].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])

def split_docs_generator(filename, batch_size, batch_length, filter_size, Y):
    """
        (Mostly) fixed-size batching for better performance. For use with modified convnet.
        For docs longer than batch_size, split them into ceil(len/batch_size) separate instances i'll call 'split-docs'
        Make the split-docs overlap such that the conv filters won't lose information (use filter_size)
        Pad the last batch if needed
        Will have to ignore docs longer than (batch_length-filter_size+1)*batch_size words, which shouldnt be many
            (this value is 65280 for batch size 64, batch length 1024)
        Data and target tensors will not be the same size
        Return: data/target tensors, array indicating which doc starts where, which will be needed for the model to properly
            concatenate conv outputs across the whole batch
    """
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        start_inds = []
        cur_insts = []
        cur_labels = []
        cur_length = 0
        batch_length_eff = batch_length - filter_size + 1
        combine_with_long = True
        for row in r:
            #get next document
            text = [int(w) for w in row[2].split()]
            labels = [int(l) for l in row[3].split(';')]
            if len(text) < batch_length:
                combine_with_long = False
                #still group by length, so we don't have to pad these shorter docs
                length = int(row[4])
                if length > cur_length:
                    if len(cur_insts) > 0:
                        #create the tensors
                        yield np.array(cur_insts), np.array(cur_labels), np.array(start_inds)
                        #clear
                        cur_insts = []
                        cur_labels = []
                        start_inds = []
                    start_inds.append(len(cur_insts))
                    cur_insts.append(text)
                    labels = [int(l) for l in row[3].split(';')]
                    cur_labels.append([1 if i in labels else 0 for i in range(Y)])
                    #reset length
                    cur_length = length
                else:
                    if len(cur_insts) == batch_size:
                        #create the tensors
                        yield np.array(cur_insts), np.array(cur_labels), np.array(start_inds)
                        #clear
                        cur_insts = []
                        cur_labels = []
                        start_inds = []
                    start_inds.append(len(cur_insts))
                    cur_insts.append(text)
                    labels = [int(l) for l in row[3].split(';')]
                    cur_labels.append([1 if i in labels else 0 for i in range(Y)])
            else:
                #first, check if there's still room in this batch
                num_insts = int(math.ceil((len(text) - filter_size + 1)/float(batch_length_eff)))
                if (len(cur_insts) + num_insts > batch_size or not combine_with_long) and len(cur_insts) > 0:
                    #yield the current batch if no room or if previous doc was smaller length
                    yield np.array(cur_insts), np.array(cur_labels), np.array(start_inds)
                    #clear
                    cur_insts = []
                    cur_labels = []
                    start_inds = []
                    combine_with_long = True
                start_inds.append(len(cur_insts))
                #pad as needed
                desired_length = (num_insts - 1) * batch_length_eff + batch_length
                text.extend([0 for _ in range(int(desired_length - len(text)))])
                #split doc and add all to batch
                insts = [text[i:i+batch_length] for i in range(0, len(text) - filter_size + 1, batch_length_eff)]
                cur_insts.extend(insts)
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])
        yield np.array(cur_insts), np.array(cur_labels), np.array(start_inds)

def load_lookups(vocab_size, Y, vocab_min):
    v_dict = defaultdict(str)
    c_dict = defaultdict(str)
    with open("%s/vocab_lookup_%s_%s_%s.txt" % (DATA_DIR, str(vocab_size), str(Y), str(vocab_min)), 'r') as vocabfile:
        vr = csv.reader(vocabfile)
        #next(vr)
        for i,row in enumerate(vr):
            if len(row) > 1:
                v_dict[int(row[0])+1] = row[1]
            else:
                v_dict[i+1] = row[0]

    with open(DATA_DIR + "/label_lookup_" + str(Y) + '.csv', 'r') as labelfile:
        lr = csv.reader(labelfile)
        next(lr)
        for row in lr:
            c_dict[int(row[0])] = row[1]
    return (v_dict, c_dict)  

def load_code_descriptions():
    desc_dict = defaultdict(str)
    with open("%s/label_desc_lookup.txt" % (DATA_DIR), 'r') as descfile:
        r = csv.reader(descfile, delimiter=" ")
        for row in r:
            code = row[0]
            desc = " ".join(row[1:]).lstrip()
            desc_dict[code] = desc
    return desc_dict

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
