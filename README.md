**Status:** Archived. Code is provided as-is with no updates expected. Unfortunately I don't have the personal time to dedicate to maintaining this repo/responding to issues, nor access to the MIMIC dataset anymore, though I hope the model code and data splits can still be of use to the community.

# caml-mimic

Code for the paper [Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695).

## Dependencies
* Python 3.6, though 2.7 should hopefully work as well
* pytorch 0.3.0
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.20.3
* jupyter-notebook 5.0.0
* gensim 3.2.0
* nltk 3.2.4

Other versions may also work, but the ones listed are the ones I've used


## Data processing

To get started, first edit `constants.py` to point to the directories holding your copies of the MIMIC-II and MIMIC-III datasets. Then, organize your data with the following structure:
```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions (already in repo)
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

Now, make sure your python path includes the base directory of this repository. Then, in Jupyter Notebook, run all cells (in the menu, click Cell -> Run All) in `notebooks/dataproc_mimic_II.ipynb` and `notebooks/dataproc_mimic_III.ipynb`. These will take some time, so go for a walk or bake some cookies while you wait. You can speed it up by skipping the "Pre-train word embeddings" sections. 

## Saved models

To directly reproduce the results of the paper, first run the data processing steps above. We provide our pre-trained models for CAML and DR-CAML for the MIMIC-III full-label dataset. They are saved as `model.pth` in their respective directories. We also provide an `evaluate_model.sh` script to reproduce our results from the models.

## Training a new model

To train a new model from scratch, please use the script `learn/training.py`. Execute `python training.py -h` for a full list of input arguments and flags. The `train_new_model.sh` scripts in the `predictions/` subdirectories can serve as examples (or you can run those directly to use the same hyperparameters).

## Model predictions

The predictions that provide the results in the paper are provided in `predictions/`. Each directory contains: 

* `preds_test.psv`, a pipe-separated value file containing the HADM_ID's and model predictions of all testing examples
* `train_new_model.sh`, which trains a new model with the hyperparameters provided in the paper.

To reproduce our F-measure results from the predictions, for example the CNN results on MIMIC-II, run `python get_metrics_for_saved_predictions.py predictions/CNN_mimic2_full`.

