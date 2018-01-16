# cnn-medical-text

Code for assigning ICD codes to EHR notes using a CNN with attention.

## Dependencies
* Python 2.7.13
* pytorch 0.2.0
* tqdm
* scikit-learn 0.18.1
* numpy 1.12.1, scipy 0.19.0, pandas 0.20.1
* jupyter-notebook 5.0.0
* gensim 2.3.0
* nltk 3.2.3

## Data processing

First, edit `constants.py` to point to the directories holding MIMIC-II and MIMIC-III datasets. Then, organize your data with the following structure:

```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions
└───mimic2/
|   |   MIMIC_RAW_DSUMS
|   |   MIMIC_ICD9_mapping
|   |   training_indices.data
|   |   testing_indices.data
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (already in repo)
```
The MIMIC-II files can be obtained from [this repository](https://physionet.org/works/ICD9CodingofDischargeSummaries/).

Now, make sure your python path includes the base directory of this repository. Then, in Jupyter Notebook, run all cells in `notebooks/dataproc_mimic_II.ipynb` and `notebooks/dataproc_mimic_III.ipynb`.

## Model predictions

The predictions that provide the results in the paper are provided in `predictions/`. Each directory contains: 

* `preds_test.psv`, a pipe-separated value file containing the HADM_ID's and model predictions of all testing examples
* `code_scores_test.json`, which holds the 100 top code predictions and their scores, which can be used to verify precision@k results
* `train_model.sh`, which trains a model with the hyperparameters provided in the paper.

To directly reproduce our results from the predictions, for example the CNN reslults on MIMIC-II, run `python get_metrics_for_saved_predictions.py predictions/CNN_MIMIC_2`.

## Training a new model

To train a new model from scratch, please use the script `learn/training.py`. Execute `python training.py -h` for a full list of input arguments and flags. The `train_model.sh` scripts in the `predictions/` subdirectories can serve as examples.
