"""
    Saving relevant things.
"""
import csv
import json

import numpy as np
import torch

from constants import *
from learn import models

def save_metrics(metrics_hist_all, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        #concatenate dev, train metrics into one dict
        data = metrics_hist_all[0].copy()
        data.update({"%s_te" % (name):val for (name,val) in metrics_hist_all[1].items()})
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_all[2].items()})
        json.dump(data, metrics_file, indent=1)

def save_params_dict(params):
    with open(params["model_dir"] + "/params.json", 'w') as params_file:
        json.dump(params, params_file, indent=1)

def write_preds(yhat, model_dir, hids, fold, ind2c, yhat_raw=None):
    """
        INPUTS:
            yhat: binary predictions matrix 
            model_dir: which directory to save in
            hids: list of hadm_id's to save along with predictions
            fold: train, dev, or test
            ind2c: code lookup
            yhat_raw: predicted scores matrix (floats)
    """
    preds_file = "%s/preds_%s.psv" % (model_dir, fold)
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for yhat_, hid in zip(yhat, hids):
            codes = [ind2c[ind] for ind in np.nonzero(yhat_)[0]]
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
    if fold != 'train' and yhat_raw is not None:
        #write top 100 scores so we can re-do @k metrics later
        #top 100 only - saving the full set of scores is very large (~1G for mimic-3 full test set)
        scores_file = '%s/pred_100_scores_%s.json' % (model_dir, fold)
        scores = {}
        sortd = np.argsort(yhat_raw)[:,::-1]
        for i,(top_idxs, hid) in enumerate(zip(sortd, hids)):
            scores[int(hid)] = {ind2c[idx]: float(yhat_raw[i][idx]) for idx in top_idxs[:100]}
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=1)
    return preds_file

def save_everything(args, metrics_hist_all, model, model_dir, params, criterion, evaluate=False):
    """
        Save metrics, model, params all in model_dir
    """
    save_metrics(metrics_hist_all, model_dir)
    params['model_dir'] = model_dir
    save_params_dict(params)

    if not evaluate:
        #save the model with the best criterion metric
        if not np.all(np.isnan(metrics_hist_all[0][criterion])):
            if criterion == 'loss_dev': 
                eval_val = np.nanargmin(metrics_hist_all[0][criterion])
            else:
                eval_val = np.nanargmax(metrics_hist_all[0][criterion])

            if eval_val == len(metrics_hist_all[0][criterion]) - 1:                

		#save state dict
                sd = model.cpu().state_dict()
                torch.save(sd, model_dir + "/model_best_%s.pth" % criterion)
                if args.gpu:
                    model.cuda()
    print("saved metrics, params, model to directory %s\n" % (model_dir))
