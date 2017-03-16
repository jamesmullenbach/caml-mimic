"""
    Use these methods to save stuff from convnet runs
"""
import json
import sys

from constants import *


def save_metrics(metrics_hist, metrics_hist_tr, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        #concatenate dev, train metrics into one dict
        data = metrics_hist.copy()
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_tr.items()})
        json.dump(data, metrics_file, indent=1)

def rewrite_metrics(prev_metrics, metrics_hist, metrics_hist_tr, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        #concatenate dev, train metrics into one dict
        data = metrics_hist.copy()
        data.update({"%s_tr" % (name):val for (name, val) in metrics_hist_tr.items()})
        #add new data to old
        for metric in prev_metrics.keys():
            prev_metrics[metric].extend(data[metric])
        json.dump(prev_metrics, metrics_file, indent=1)

def save_params(model_dir, Y, vocab_min, dataset, model, n_epochs, framework, filter_size="n/a", min_filter="n/a", max_filter="n/a",
                conv_dim_factor="n/a", padding="var"):
    with open(model_dir + "/params.json", 'w') as params_file:
        param_names = ["Y", "dataset", "Num epochs", "Vocab size", "Embedding size", "Embed dropout", "Conv activation", "Conv window size",
                "Conv window type", "Conv output size", "Dense dropout", "Optimizer", "Loss", "Embed init", "Learning rate",
                "Padding", "Vocab min occurrences", "Framework"]
        filter_sz = filter_size if model != "cnn_multi" else ';'.join([str(i) for i in range(min_filter, max_filter + 1)])
        param_vals = [Y, dataset, n_epochs, VOCAB_SIZE, EMBEDDING_SIZE, DROPOUT_EMBED, ACTIVATION_CONV, filter_sz, WINDOW_TYPE, 
                conv_dim_factor*Y, DROPOUT_DENSE, OPTIMIZER, LOSS, EMBED_INIT, LEARNING_RATE, padding, vocab_min, framework]
        data = {name: str(val) for (name,val) in zip(param_names, param_vals)}
        json.dump(data, params_file, indent=1)

def rewrite_params(model_dir, dataset, n_epochs):
    with open(model_dir + "/params.json", 'r') as params_file:
        params = json.load(params_file)
    params["dataset"] = dataset
    params["Num epochs"] = n_epochs
    with open(model_dir + "/params.json", 'w') as params_file:
        json.dump(params, params_file, indent=1)

def write_auc(fpr, tpr, roc_auc, Y):
    #write the AUC values for later visualization
    with open("auc_" + str(Y) + ".csv", 'w') as outfile:
        outfile.write(','.join(['label', 'measure', 'values']) + "\n")
        for label in fpr.keys():
            fpr_line = [str(label), 'fpr']
            fpr_line.extend([str(v) for v in fpr[label]])
            outfile.write(','.join(fpr_line) + "\n")
    
            tpr_line = [str(label), 'tpr']
            tpr_line.extend([str(v) for v in tpr[label]])
            outfile.write(','.join(tpr_line) + "\n")

            auc_line = [str(label), 'auc', str(roc_auc[label])]
            outfile.write(','.join(auc_line) + "\n")

def write_preds(preds, filename):
    #write predictions to a csv
    with open(filename, 'w') as outfile:
        for p in preds:
            outfile.write(','.join([str(p_i) for p_i in p]) + "\n")

def check_constants(params, Y, vocab_min):
    if int(params["Y"]) != Y or int(params["Vocab min occurrences"]) != vocab_min:
        return False, "expected (Y, vocab_min) to match previous: (" + params["Y"] + ", " + params["Vocab min occurrences"] + ")" + \
                      " but got (" + str(Y) + ", " + str(vocab_min)  + ")"
    return True, "ok"

def load_model(saved_dir, Y, vocab_min, framework):
    model_path = saved_dir + "/model.pth" if framework == "torch" else saved_dir + "/model.h5"
    params_path = saved_dir + "/params.json"
    metrics_path = saved_dir + "/metrics.json"
    if framework == "keras":
        from keras.models import load_model
        model = load_model(model_path)
    elif framework == "torch":
        import torch
        model = torch.load(model_path)
    p_file = open(params_path, 'r')
    params = json.load(p_file)
    ok, msg = check_constants(params, Y, vocab_min)
    if not ok:
        print(msg)
        sys.exit(0)
    m_file = open(metrics_path, 'r')
    metrics = json.load(m_file)
    filter_size = params["Conv window size"]
    if ";" in filter_size:
        min_filter = int(filter_size.split(";")[0])
        max_filter = int(filter_size.split(";")[-1])
        filter_size = None
    else:
        min_filter = None
        max_filter = None
        if filter_size != "n/a":
            filter_size = int(filter_size)
        else:
            filter_size = None
    if params["Conv output size"] != "n/a":
        conv_dim_factor = int(params["Conv output size"])/int(params["Y"])
    else:
        conv_dim_factor = None
    prev_epochs = int(params["Num epochs"])
    prev_dataset = params["dataset"]
    return model, filter_size, min_filter, max_filter, conv_dim_factor, prev_epochs, metrics, prev_dataset
