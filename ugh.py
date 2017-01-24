import pandas as pd

note_subjects = pd.read_csv('../mimicdata/patients_10.csv', usecols=['SUBJECT_ID'], squeeze=True).unique()

label_subjects = pd.read_csv('../mimicdata/labels_10.csv', usecols=['SUBJECT_ID'], squeeze=True).unique()

print(len(note_subjects))
print(len(label_subjects))