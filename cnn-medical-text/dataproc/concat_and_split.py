"""
    Concatenate the labels with the notes data and split by subject id, randomly into 50/25/25 train/dev/test
"""
import csv
from datetime import datetime
import os
import random
import sys

from constants import DATA_DIR

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

TRAIN_PCT = 0.5
DEV_PCT = 0.25
TEST_PCT = 0.25

def main(Y):
    with open('%s/labels_%s_filtered.csv' % (DATA_DIR, Y), 'r') as labelsfile:
        concat_data(labelsfile, Y)
    #read what you just wrote
    with open("%s/notes_%s_labeled.csv" % (DATA_DIR, Y), 'r') as labeledfile:
        split_data(labeledfile, Y)
    #remove intermediate file
    os.remove("%s/notes_%s_labeled.csv" % (DATA_DIR, Y))

def concat_data(labelsfile, Y):
    print("CONCATENATING")
    with open('%s/notes_%s_sorted.csv' % (DATA_DIR, Y), 'r') as notesfile:
        with open('%s/notes_%s_labeled.csv' % (DATA_DIR, Y), 'w') as outfile:
            note_reader = csv.reader(notesfile)
            next(note_reader)

            labels_gen = next_labels(labelsfile)
            cur_subj, cur_labels, cur_admit_time, cur_disch_time = next(labels_gen)

            outfile.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

            i = 0
            for row in note_reader:
                if i % 10000 == 0:
                    print(str(i) + " done")
                subj_id = int(row[0])
                hadm_id = row[1]
                note_time = datetime.strptime(row[2], DATETIME_FORMAT)
                text = row[3]

                #keep getting the next label set until we either:
                #    find a match
                #    get to the next subject
                #    get a label set time with admit time > note time
                #    if the latter two happen, just skip this note. it has no valid codes associated with it
                keep_searching = True
                while keep_searching:
                    if cur_subj == subj_id:
                        if note_time > cur_admit_time and note_time < cur_disch_time:
                            # MATCH! write the output
                            outline = [str(subj_id), hadm_id, text]
                            outline.extend(cur_labels)
                            outfile.write(','.join(outline) + "\n")
                            keep_searching = False
                        elif note_time < cur_admit_time:
                            keep_searching = False
                    elif cur_subj > subj_id:
                        keep_searching = False
                    if keep_searching:
                        cur_subj, cur_labels, cur_admit_time, cur_disch_time = next(labels_gen)
                i += 1

def split_data(labeledfile, Y):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_file = open('%s/notes_%s_train_split.csv' % (DATA_DIR, Y), 'w')
    dev_file = open('%s/notes_%s_dev_split.csv' % (DATA_DIR, Y), 'w')
    test_file = open('%s/notes_%s_test_split.csv' % (DATA_DIR, Y), 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT', 'LABELS']) + "\n")

    #every time you hit a new hadm_id, flip a coin
    cur_split = flip_coin()
    reader = csv.reader(labeledfile)
    next(reader)
    i = 0
    cur_hadm = 0
    for row in reader:
        #filter text, write to file according to train/dev/test split
        if i % 10000 == 0:
            print(str(i) + " read")
        text = row[3]

        subj_id = int(row[0])
        hadm_id = int(float(row[1]))
        if hadm_id != cur_hadm:
            cur_split = flip_coin()
            cur_hadm = hadm_id

        if cur_split == "train":
            train_file.write(','.join(row) + "\n")
        elif cur_split == "dev":
            dev_file.write(','.join(row) + "\n")
        else:
            test_file.write(','.join(row) + "\n")
        i += 1
    train_file.close()
    dev_file.close()
    test_file.close()
    
def flip_coin():
    val = random.random()
    if val < TRAIN_PCT:
        return "train"
    elif val >= TRAIN_PCT and val < TRAIN_PCT + DEV_PCT:
        return "dev"
    else:
        return "test"

def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_admit_time = datetime.strptime(first_label_line[3], DATETIME_FORMAT)
    cur_disch_time = datetime.strptime(first_label_line[4], DATETIME_FORMAT)
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        code = row[2]
        admit_time = datetime.strptime(row[3], DATETIME_FORMAT)
        disch_time = datetime.strptime(row[4], DATETIME_FORMAT)
        if not (admit_time == cur_admit_time and disch_time == cur_disch_time) \
            or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_admit_time, cur_disch_time
            cur_labels = [code]
            cur_subj = subj_id
            cur_admit_time = admit_time
            cur_disch_time = disch_time
        else:
            #add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_admit_time, cur_disch_time
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
        sys.exit(0)
    main(int(sys.argv[1]))
