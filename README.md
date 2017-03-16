# cnn-medical-text

[Data description](https://mimic.physionet.org/about/mimic/)

##Code for the models:
* keras_stuff/training.py - trains cnn or rnn models in Keras.
* keras_stuff/models.py - model building methods in Keras
* keras_stuff/convnet_pad.py - fixed-length convolutional network in Keras.
* torch_stuff/training.py - trains cnn or rnn models in PyTorch
* torch_stuff/models.py - model building methods in PyTorch
* constants.py - constants related to NN models. should (ideally) be changed rarely
* datasets.py - data loading methods
* log_reg.py - BOW logistic regression in scikit-learn.
* persistence.py - saving params, metrics, etc.

##Code for evaluation:
* evaluation.py - [micro-/macro-]accuracy, precision, recall, F1, AUC

##Code for preprocessing:
* dataproc/build_vocab.py - selects top N words for the vocabulary from training set
* dataproc/concat_notes_labels.py - joins notes and labels according to times
* dataproc/filter_patients_and_labels.py - drops patients whose notes had no times associated with them
* dataproc/get_notes.py - selects notes corresponding to patients with at least one label in the label space. Also tokenizes/does text preproc
* dataproc/group_and_sort.py - drop rows from notes with no times, sorts by patient ID and time
* dataproc/longest_note.py - selects the single longest note, per patient-visit
* dataproc/sort_by_length.py - sort note data by word length for batching by length
* dataproc/split_data.py - splits notes into train/dev/test based on # subjects (50/25/25 split)
* dataproc/split_labels.py - splits labels '' '' '' ''
* dataproc/vocab_select.py - map words in the notes to indices
