from collections import defaultdict
import csv
import json
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
dset = model_dir[model_dir.index('MIMIC'):]
parts = dset.split('_')

Y = 'full' if len(parts) == 2 or parts[2].startswith('FULL') else 50
version = 'mimic2' if parts[1].startswith('2') else 'mimic3'
data_dir = MIMIC_2_DIR if version == 'mimic2' else MIMIC_3_DIR
train_file = '%s/train.csv' % data_dir if version == 'mimic2' else '%s/train_%s.csv' % (data_dir, str(Y))
test_file = '%s/test.csv' % data_dir if version == 'mimic2' else '%s/test_%s.csv' % (data_dir, str(Y))

num_labels = tools.get_num_labels(Y, version=version)
ind2c, _ = datasets.load_full_codes(train_file, version=version)
c2ind = {c:i for i,c in ind2c.iteritems()}

preds = defaultdict(lambda: [])
            
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
            
golds = defaultdict(lambda: [])
with open(test_file, 'r') as f:
    r = csv.reader(f)
    #header
    next(r)
    for row in r:
        codes = set([c2ind[c] for c in row[3].split(';')])
        golds[row[1]] = codes

#import pdb; pdb.set_trace()

with open('%s/code_scores_test.json' % model_dir, 'r') as f:
    scors = json.load(f)
        
hadm_ids = sorted(set(golds.keys()).intersection(set(preds.keys())))
yhat = np.zeros((len(hadm_ids), num_labels))
yhat_raw = np.zeros((len(hadm_ids), num_labels))
y = np.zeros((len(hadm_ids), num_labels))
   
for i,hadm_id in tqdm(enumerate(hadm_ids)):
    yhat_inds = [1 if j in preds[hadm_id] else 0 for j in range(num_labels)]
    yhat_raw_inds = [scors[hadm_id][ind2c[j]] if ind2c[j] in scors[hadm_id] else 0 for j in range(num_labels)]
    gold_inds = [1 if j in golds[hadm_id] else 0 for j in range(num_labels)]
    yhat[i] = yhat_inds
    yhat_raw[i] = yhat_raw_inds
    y[i] = gold_inds
   
metrics = evaluation.all_metrics(yhat, y)
k = 5 if Y == 50 else 8
prec_at_k = evaluation.precision_at_k(yhat_raw, y, k)
metrics['prec_at_%d' % k] = prec_at_k
rec_at_k = evaluation.recall_at_k(yhat_raw, y, k)
metrics['rec_at_%d' % k] = rec_at_k

evaluation.print_metrics(metrics)
