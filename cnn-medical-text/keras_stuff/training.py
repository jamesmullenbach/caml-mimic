"""
    Trains a model over the data to predict ICD-9 diagnosis codes
"""
import argparse
import os
import numpy as np
import random
import sys
import time

from constants import *
import datasets, evaluation, persistence
import keras_stuff.models as models

def main(Y, vocab_min, dataset, model_name, n_epochs, filter_size, min_filter, max_filter, conv_dim_factor, saved_dir):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    prev_epochs = 0
    if model_name == "cnn_multi":
        model = models.multi_window(Y, filter_size, conv_dim_factor)
    elif model_name == "lstm":
        model = models.lstm(Y)
    elif model_name == "cnn_vanilla":
        model = models.vanilla(Y, filter_size, conv_dim_factor)
    elif model_name == "saved":
        model, filter_size, min_filter, max_filter, conv_dim_factor, prev_epochs, prev_metrics, prev_dataset\
            = persistence.load_model(saved_dir, Y, vocab_min)
        print(model.summary())

    history = []
    names = ["acc", "prec", "rec", "f1", "auc"]
    names.extend(["%s_micro" % (name) for name in names])
    metrics_hist = {name: [] for name in names}
    metrics_hist_tr = {name: [] for name in names}
    for i in range(prev_epochs, prev_epochs+n_epochs):
        hist = train(model, i, dataset, Y, filter_size, max_filter)
        history.append(hist)
        metrics, fpr, tpr = test("dev", model, dataset, Y, filter_size, max_filter, print_samples=True)
        evaluation.print_metrics(metrics)
        for name in names:
            metrics_hist[name].append(metrics[name])


        print("sanity check on train")
        metrics_t, fpr_t, tpr_t = test("train", model, dataset, Y, filter_size, max_filter)
        evaluation.print_metrics(metrics_t)
        for name in names:
            metrics_hist_tr[name].append(metrics_t[name])
    #save metric history, model, params
    if model_name != "saved":
        model_dir = "../saved_models/" + '_'.join([model_name, time.strftime('%b_%d_%H:%M', time.gmtime())])
        os.mkdir(model_dir)

        persistence.save_metrics(metrics_hist, metrics_hist_tr, model_dir)
        persistence.save_params(model_dir, Y, vocab_min, dataset, model_name, n_epochs, "keras", filter_size, 
                                min_filter, max_filter, conv_dim_factor)
        model.save(model_dir + "/model.h5")
    else:
        if dataset != prev_dataset:
            response = raw_input("***WARNING*** you ran the saved model on a different dataset than it was previously trained on. Continue? (y/n) > ")
            if "y" not in response:
                print("not saving any results (model, params, or metrics)")
                sys.exit(0)
            dataset = ",".join([prev_dataset, dataset])
        #overwrite old files w/ new values, rename folder
        persistence.rewrite_metrics(prev_metrics, metrics_hist, metrics_hist_tr, saved_dir)
        persistence.rewrite_params(saved_dir, dataset, prev_epochs+n_epochs)
        model.save(saved_dir + "/model.h5")


def train(model, epoch, dataset, Y, filter_size, max_filter, print_weights=False):
    """
        Trains the model for one epoch on train data. Batches by length
        params:
            model: the model
            epoch: number of epochs previously trained + 1
            dataset: full or single
            Y: size of label space
        returns:
            history of losses (one per batch)
    """
    hist = []
    if filter_size is not None:
        min_size = max_filter if max_filter is not None else filter_size
    else:
        min_size = 1
    for idx, (X_batch, Y_batch) in enumerate(datasets.data_generator('%s/notes_%s_train_%s_sorted.csv'\
                                                                % (DATA_DIR, str(Y), dataset),\
                                                            BATCH_SIZE, Y)):
        if (X_batch.shape[1] < min_size):
            continue
        loss = model.train_on_batch(X_batch, Y_batch)
        if idx % 500 == 0:
            print('Train Epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {}'.format(
                epoch+1, idx, X_batch.shape[0], X_batch.shape[1], loss))
        if idx % 500 == 0 and print_weights:
            print("last layer weights: " + str(model.layers[-3].get_weights()[0][:5]))
        hist.append(loss)
    return hist

def test(fold, model, dataset, Y, filter_size, max_filter, max_iter = 1e9, print_samples=False):
    """
        Runs the model on dev data, computes a bunch of metrics
        params:
            fold: dev or train or test
            model: the model
            dataset: full or single
            Y: size of the label space
            max_iter: (for debugging) max number of batches to predict on
        returns:
            metrics: contains acc, prec, rec, f1, auc measures
            fpr: false positive rate vector
            tpr: true positive rate vector
    """
    Y_tot = []
    Y_hat_tot = []
    if filter_size is not None:
        min_size = max_filter if max_filter is not None else filter_size
    else:
        min_size = 1
    for idx,(X_batch, Y_batch) in enumerate(datasets.data_generator('%s/notes_%s_%s_%s_sorted.csv'\
                                                               % (DATA_DIR, str(Y), fold, dataset),\
                                                           BATCH_SIZE, Y)):
        if idx >= max_iter:
            break
        if X_batch.shape[1] < min_size:
            continue
        Y_hat = model.predict_on_batch(X_batch)

        if np.random.rand() > 0.999 and print_samples:
            print("sample prediction")
            print("Y_true: " + str(Y_batch[0]))
            print("Y_hat: " + str(Y_hat[0]))
            Y_hat = np.round(Y_hat)
            print("Y_hat: " + str(Y_hat[0]))
            print

        Y_hat = np.round(Y_hat)

        Y_hat_tot.extend(Y_hat)
        Y_tot.extend(Y_batch)
  
    metrics = {}
    Y_tot = np.array(Y_tot)
    Y_hat_tot = np.array(Y_hat_tot)
    metrics, fpr_tot, tpr_tot = evaluation.all_metrics(Y_hat_tot, Y_tot)

    return metrics, fpr_tot, tpr_tot

def plot_auc(fpr, tpr, roc_auc):
    #plot the AUC values for current visualization
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro ROC (area={0:0.3f})'.format(roc_auc["micro"]))
    plt.plot(fpr["macro"], tpr["macro"], label='macro ROC (area={0:0.3f})'.format(roc_auc["macro"]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.show()


def check_args(args):
    if args.model == "saved" and args.saved_dir is None:
        return False, "Specified 'saved' but no model path given"
    if args.model == "cnn_vanilla" and args.filter_size is None:
        return False, "Specified 'cnn_vanilla' but no filter size given"
    if args.model == "cnn_multi" and (args.min_filter is None or args.max_filter is None):
        return False, "Specified 'cnn_multi', but (min_filter, max_filter) not fully specified"
    if (args.model == "cnn_vanilla" or args.model == "cnn_multi") and args.conv_dim_factor is None:
        return False, "Specified a cnn model but no conv_dim_factor given"
    else:
        return True, "OK"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Y", type=int, help="size of label space")
    parser.add_argument("vocab_min", type=int, help="vocab parameter (min # occurrences)")
    parser.add_argument("dataset", type=str, choices=["single", "full"], help="which dataset to learn ('single' or 'full')")
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "cnn_multi", "lstm", "saved"], 
                        help="model ('cnn_vanilla', 'cnn_multi', 'lstm')")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--filter-size", type=int, required=False, dest="filter_size",
                        help="size of convolution filter to use (cnn_vanilla only)")
    parser.add_argument("--min-filter", type=int, required=False, dest="min_filter",
                        help="min size of filter range to use (cnn_multi only)")
    parser.add_argument("--max-filter", type=int, required=False, dest="max_filter",
                        help="max size of filter range to use (cnn_multi only)")
    parser.add_argument("--conv-dim-factor", type=int, required=False, dest="conv_dim_factor",
                        help="size of conv output (divided by Y e.g. Y=10, conv-dim-factor=3, conv output size is 30)")
    parser.add_argument("--saved-model", type=str, required=False, dest="saved_dir",
                        help="path to a directory containing a saved model (and params and metrics) to load instead of building one")
    args = parser.parse_args()
    ok, msg = check_args(args)
    if ok:
        main(args.Y, args.vocab_min, args.dataset, args.model, args.n_epochs, args.filter_size, args.min_filter,
             args.max_filter, args.conv_dim_factor, args.saved_dir)
    else:
        print(msg)
        sys.exit(0)

