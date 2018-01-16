# Data processing
This directory holds scripts to slice 'n dice the data. 

There's various pipelines to structure the data into various formats that are generated as training/testing instances for different models by the generators in `datasets.py`

There are probably better ways of doing all this preprocessing that don't involve so many intermediate files, and I may refactor (combine) some of these scripts to make the process involve less manual labor and disk space.

Check the comments in the scripts for more details on what they do.

Throughout everything below, let `Y` be the size of the label space you want to consider (i.e. `Y = 10` means you want to consider the top 10 most frequent codes)

## Required steps for all formats
To start, create the SQL databases as described [here](https://mimic.physionet.org/gettingstarted/dbsetup/), and use the queries in queries.sql to create:

* a list of the top `Y` most common codes called `labels_Y.csv`
* a list of all patients that have at least one code in the top `Y` most common codes called `patients_Y.csv`
* You will need at least the `ADMISSIONS`, `DIAGNOSES_ICD`, and `NOTEEVENTS` tables (`NOTEEVENTS.csv` is a required file)

### Format-specific
To build instances containing all notes for an admission concatenated together in sequence, do the following in order:

1. `python get_notes.py Y` -> `notes_Y.csv`
2. `python group_and_sort.py Y` -> `notes_Y_sorted.csv` (temp file)
3. `python filter_patients_and_labels.py` -> `(patients_Y_filtered.csv, labels_Y_filtered.csv)`
4. `python concat_and_split.py Y` -> `(notes_Y_train_split.csv, notes_Y_dev_split.csv, notes_Y_test_split.csv)` (temp files)
5. `python build_vocab.py V Y min_occur` -> `vocab_lookup_V_Y_min_occur.txt`
6. (optional) `python remove_redundancy.py Y threshold` -> `(notes_Y_train_no_redundancies.csv, notes_Y_dev_no_redundancies.csv, notes_Y_test_no_redundancies.csv)` (temp files)
    * This will remove spans of words longer than `threshold` shared with earlier notes in the same admission
    * Unknown whether this helps much downstream
7. `python vocab_select.py V Y min_occur` -> `(notes_Y_train_V_indices.csv, notes_Y_dev_V_indices.csv, notes_Y_test_V_indices.csv)` (temp files)
8. `python stitch_notes.py V Y min_occur num_zeros` -> `(notes_Y_train_V_stitched.csv, note_Y_dev_V_stitched.csv, notes_Y_test_V_stitched.csv)` (temp files)
9. `python sort_by_length.py filename notes_Y_train.csv 0` -> `notes_Y_train.csv` (repeat this for dev, test)
*  this data will include ALL notes for the patient list for `Y`, and will work with the `datasets.data_generator` and `datasets.split_docs_generator` used in `learn/training.py`

To build instances grouped by admission and sorted by number of notes in the admission, do the following in order:

* Steps 1-7 above
* `python build_attn_data.py Y train` -> `notes_10_train_attn.csv` (repeat for dev, test)
* this data will include ALL notes for the patient list for `Y`, and will work with the `datasets.attn_generator` used in `learn/hattn_training.py`

To build instances containing just the discharge summary note for each admission, do the following in order:

* `python get_discharge_summaries.py Y` -> `notes_Y_disch.csv`
* Steps 2,3 above
* `python concat_and_split_disch.py Y` -> `(notes_Y_train_split.csv, notes_Y_dev_split.csv, notes_Y_test_split.csv)` (temp files)
* Steps 5,7,9 above (may require slight alterations of the code)
* this data will include only discharge summary notes for each patient in the list for `Y`, and will work with the `datasets.data_generator` used in `learn/training.py`

To build instances containing all notes for an admission concatenated together in sequence, labeled by the codes for the same patient's next admission, do the following in order:

* Steps 1-8 above
* `python build_future_data.py Y train` -> `(future_codes_Y_train.csv, future_codes_text_Y_train.csv)` (repeat for dev, test)
* this creates two datasets, which will work with `datasets.codes_only_generator` and `datasets.next_codes_text_generator` used in `learn/training.py`

At the end of this, make sure you edit the `constants.py` file to point to the directory holding the data

You can also delete any files marked (temp files) in the instructions above - technically, all you need is the final `notes_Y_train.csv` files, though I recommend keeping the others around (you may need `notes_Y.csv` to train word embeddings, for example).

### Training word embeddings
To train word2vec embeddings for initialization in neural network models, do the following in order:

* `python word_embeddings.py Y dataset min_count n_iter` -> `something.w2v` (see script for details)
* `python extract_wvs.py V Y dataset min_occur` -> `something.embed` (for use in `learn/models.py`)
* Note that the `min_occur` you use here should be less than or equal to the `min_occur` used in data processing
