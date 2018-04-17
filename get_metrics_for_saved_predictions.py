"""
    Get all metrics from a directory with saved predictions
"""
from collections import defaultdict
import csv
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from constants import *
import datasets
import evaluation
from learn import tools


if len(sys.argv) < 2:
    print("usage: python get_metrics_for_saved_predictions.py [path_to_saved_predictions_dir]")
    sys.exit(0)

model_dir = sys.argv[1]
dset = model_dir[model_dir.index('mimic'):]
parts = dset.split('_')

Y = 'full' if parts[1].startswith('full') else 50
version = 'mimic2' if parts[0].endswith('2') else 'mimic3'
data_dir = MIMIC_2_DIR if version == 'mimic2' else MIMIC_3_DIR
train_file = '%s/train.csv' % data_dir if version == 'mimic2' else '%s/train_%s.csv' % (data_dir, str(Y))
test_file = '%s/test.csv' % data_dir if version == 'mimic2' else '%s/test_%s.csv' % (data_dir, str(Y))

ind2c, _ = datasets.load_full_codes(train_file, version=version)
c2ind = {c:i for i,c in ind2c.items()}
num_labels = len(ind2c)

preds = defaultdict(lambda: [])
            
print("loading predictions")
with open('%s/preds_test.psv' % model_dir, 'r') as f:
    r = csv.reader(f, delimiter='|')
    for row in r:
        if len(row) > 1:
            try:
                preds[row[0]] = set([c2ind[c] for c in row[1:] if c != ''])
            except:
                import pdb; pdb.set_trace()
        else:
            preds[row[0]] = set([])
            
print("loading ground truth")
golds = defaultdict(lambda: [])
with open(test_file, 'r') as f:
    r = csv.reader(f)
    #header
    next(r)
    for row in r:
        codes = set([c2ind[c] for c in row[3].split(';')])
        golds[row[1]] = codes

have_scores = os.path.exists('%s/pred_scores_test.json' % model_dir)
if have_scores:
    with open('%s/pred_scores_test.json' % model_dir, 'r') as f:
        scors = json.load(f)
        
hadm_ids = sorted(set(golds.keys()).intersection(set(preds.keys())))
yhat = np.zeros((len(hadm_ids), num_labels))
if have_scores:
    yhat_raw = np.zeros((len(hadm_ids), num_labels))
else:
    yhat_raw = None
y = np.zeros((len(hadm_ids), num_labels))
   
print("reformatting predictions")
for i,hadm_id in tqdm(enumerate(hadm_ids)):
    yhat_inds = [1 if j in preds[hadm_id] else 0 for j in range(num_labels)]
    gold_inds = [1 if j in golds[hadm_id] else 0 for j in range(num_labels)]
    yhat[i] = yhat_inds
    y[i] = gold_inds
    if have_scores:
        yhat_raw_inds = [scors[hadm_id][ind2c[j]] if ind2c[j] in scors[hadm_id] else 0 for j in range(num_labels)]
        yhat_raw[i] = yhat_raw_inds
   
if version == "mimic3" and Y == "full":
    print("evaluating code-type metrics")
    diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts = evaluation.results_by_type(Y, model_dir, version)
    f1_diag = evaluation.diag_f1(diag_preds, diag_golds, type_dicts[0], hadm_ids)
    f1_proc = evaluation.proc_f1(proc_preds, proc_golds, type_dicts[1], hadm_ids)
    print("[BY CODE TYPE] f1-diag f1-proc")
    print("%.4f %.4f" % (f1_diag, f1_proc))

k = [5] if Y == 50 else [8,15]
print("evaluating all other metrics")
metrics = evaluation.all_metrics(yhat, y, k=k, yhat_raw=yhat_raw)

evaluation.print_metrics(metrics)

