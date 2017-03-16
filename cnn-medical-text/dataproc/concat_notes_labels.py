"""
    This method takes in a labels file that contains time information, and the train, dev, and test note sets
    It finds the correspondence between notes and label sets, then writes that output to a new file
"""
import csv
from datetime import datetime
import os
import sys

from constants import DATA_DIR

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def main(Y):

    with open('%s/labels_%s_filtered.csv' % (DATA_DIR, Y), 'r') as labelsfile:
        concat_data(labelsfile, Y, 'train')
        concat_data(labelsfile, Y, 'dev')
        concat_data(labelsfile, Y, 'test')

def concat_data(labelsfile, Y, dataset):
    print("processing " + dataset + " data")
    with open('%s/notes_%s_%s.csv' % (DATA_DIR, Y, dataset), 'r') as notesfile:
        with open('%s/notes_%s_%s_labeled.csv' % (DATA_DIR, Y, dataset), 'w') as outfile:
            note_reader = csv.reader(notesfile)
            next(note_reader)

            labels_gen = next_labels(labelsfile)
            cur_subj, cur_labels, cur_admit_time, cur_disch_time = next(labels_gen)

            outfile.write(','.join(['SUBJECT_ID', 'TEXT', 'LABELS']) + "\n")

            i = 0
            for row in note_reader:
                if i % 10000 == 0:
                    print(str(i) + " done")
                subj_id = int(row[0])
                note_time = datetime.strptime(row[1], DATETIME_FORMAT)
                text = row[2]

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
                            outline = [str(subj_id), text]
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
