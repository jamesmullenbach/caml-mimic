import csv
import os
import sys

from constants import DATA_DIR

def main(Y, word):
    subjects = set([])
    print("getting the list of relevant subjects")
    with open('%s/patients_%s.csv' % (DATA_DIR, Y), 'r') as csvfile:
        patientreader = csv.reader(csvfile)
        next(patientreader)
        for line in patientreader:
            subjects.add(int(line[0]))

    print("processing notes file in search of word %s" % (word))
    with open('%s/notes_%s_train_first10000.csv' % (DATA_DIR, Y), 'r') as csvfile:
        notereader = csv.reader(csvfile)
        next(notereader)
        i = 0
        for line in notereader:
            if i % 10000 == 0:
                print(i)
#            subj = int(line[1])
#            if subj in subjects:
#            words = set(line[10].split())
            words = set(line[2].split())
            if word in words:
                print("found word %s: %s" % (word, words))
            i += 1

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " Y word"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
