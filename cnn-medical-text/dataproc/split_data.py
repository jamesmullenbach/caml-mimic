"""
    This script takes in a set of notes sorted by subject ID and chart time
    It divides the data into train/dev/test sets
"""
import csv
import os
import sys

import pandas as pd

from constants import DATA_DIR

TRAIN_PCT = 0.5
DEV_PCT = 0.25
TEST_PCT = 0.25

def main(Y):

    #select which indices go to which set
    print("finding unique subject ids")
    num_subjects = len(pd.read_csv('%s/notes_%s_sorted.csv' % (DATA_DIR, Y), usecols=['SUBJECT_ID'], squeeze=True).unique())
    print("num subjects: " + str(num_subjects))

    #create and write headers for train, dev, test
    train_file = open('%s/notes_%s_train.csv' % (DATA_DIR, Y), 'w')
    dev_file = open('%s/notes_%s_dev.csv' % (DATA_DIR, Y), 'w')
    test_file = open('%s/notes_%s_test.csv' % (DATA_DIR, Y), 'w')
    train_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'CHARTTIME', 'TEXT']) + "\n")

    with open('%s/notes_%s_sorted.csv' % (DATA_DIR, Y), 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)
        i = 0
        subj_seen = 0
        cur_subj = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")
            text = row[2]

            subj_id = int(row[0])
            if subj_id != cur_subj:
                subj_seen += 1
                cur_subj = subj_id

            if subj_seen < num_subjects * TRAIN_PCT:
                train_file.write(','.join(row) + "\n")
            elif subj_seen >= num_subjects * TRAIN_PCT and subj_seen < num_subjects * (TRAIN_PCT + DEV_PCT):
                dev_file.write(','.join(row) + "\n")
            else:
                test_file.write(','.join(row) + "\n")
            i += 1
    train_file.close()
    dev_file.close()
    test_file.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|]"))
        sys.exit(0)
    main(sys.argv[1])    
