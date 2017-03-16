"""
    Train a model with PyTorch
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from constants import *
import evaluation
import persistence
import torch_stuff.models as models
import torch_stuff.objectives as objectives

import csv
import argparse
import os
import numpy as np
import random
import sys
import time

def main(Y, vocab_min, dataset, model_name, n_epochs, objective, filter_size,
         min_filter, max_filter, conv_dim_factor, saved_dir, gpu):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    if model_name == "cnn_multi":
        print("no torch model for cnn_multi yet")
        sys.exit(0)
    elif model_name == "lstm":
        print("no torch model for lstm yet")
        sys.exit(0)
    elif model_name == "cnn_vanilla":
        model = models.VanillaConv(Y, filter_size, conv_dim_factor)
        if gpu:
            model.cuda()
    elif model_name == "saved":
        model, filter_size, min_filter, max_filter, conv_dim_factor, prev_epochs, prev_metrics, prev_dataset\
            = persistence.load_model(saved_dir, Y, vocab_min)
        sys.exit(0)

    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    names = ["acc", "prec", "rec", "f1", "auc"]
    names.extend(["%s_micro" % (name) for name in names])
    metrics_hist = {name: [] for name in names}
    metrics_hist_tr = {name: [] for name in names}
    for epoch in range(n_epochs):
        train(model, optimizer, Y, epoch, dataset, filter_size, gpu, objective)
        print("evaluating on dev")
        metrics, fpr, tpr = test(model, Y, epoch, dataset, "dev", filter_size, gpu)
        for name in names:
            metrics_hist[name].append(metrics[name])

        print("sanity check on train")
        metrics_t, _, _ = test(model, Y, epoch, dataset, "train", filter_size, gpu, print_samples=False)
        for name in names:
            metrics_hist_tr[name].append(metrics_t[name])

    #save metric history, model, params
    if model_name != "saved":
        model_dir = MODEL_DIR + '_'.join([model_name, time.strftime('%b_%d_%H:%M', time.gmtime())])
        os.mkdir(model_dir)
    
        persistence.save_metrics(metrics_hist, metrics_hist_tr, model_dir)
        persistence.save_params(model_dir, Y, vocab_min, dataset, model_name, n_epochs, "torch", filter_size, 
                                min_filter, max_filter, conv_dim_factor)
        torch.save(model, model_dir + "/model.pth")
    else:
        if dataset != prev_dataset:
            response = raw_input("***WARNING*** you ran the saved model on a different dataset than it was previously trained on. Save? (y/n) > ")
            if "y" not in response:
                print("not saving any results (model, params, or metrics)")
                sys.exit(0)
            dataset = ",".join([prev_dataset, dataset])
        #overwrite old files w/ new values, rename folder
        persistence.rewrite_metrics(prev_metrics, metrics_hist, metrics_hist_tr, saved_dir)
        persistence.rewrite_params(saved_dir, dataset, prev_epochs+n_epochs)
        torch.save(model, saved_dir + "/model.h5")


def train(model, optimizer, Y, epoch, dataset, filter_size, gpu, objective):
    filename = DATA_DIR + "/notes_" + str(Y) + "_train_" + dataset + "_sorted.csv"
    #put model in "train" mode
    model.train()
    losses = []
    if objective == "warp":
        batch_size = 1
        print_every = 5000
    else:
        batch_size = BATCH_SIZE
        print_every = 500
    for batch_idx, (data, target) in enumerate(data_generator(filename, batch_size, Y)):
        data, target = Variable(data), Variable(target)
        if data.size()[1] < filter_size:
            continue
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #clear gradients
        optimizer.zero_grad()
        model.zero_grad()
        #forward computation
        output = model(data)
        if objective == "warp":
            output = output.squeeze()
            target = target.squeeze()
            loss = objectives.warp_loss(output, target)
            if loss.size()[0] > 1:
                loss = loss.sum()
                loss.backward()
                optimizer.step()
        else:
            output = F.sigmoid(output)
            loss = F.binary_cross_entropy(output, target)
            #backward pass
            loss.backward()
            optimizer.step()
        losses.append(loss.data[0])
        if batch_idx % print_every == 0:
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.size()[0], data.size()[1], np.mean(losses)))

def test(model, Y, epoch, dataset, fold, filter_size, gpu, print_samples=True):
    filename = DATA_DIR + "/notes_" + str(Y) + "_" + fold + "_" + dataset + "_sorted.csv"
    #put model in "test" mode
    model.eval()
    y = []
    yhat = []
    yhat_raw = []
    for data, target in data_generator(filename, BATCH_SIZE, Y):
        data, target = Variable(data, volatile=True), Variable(target)
        if data.size()[1] < filter_size:
            continue
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #clear gradients
        model.zero_grad()
        #predict
        output = F.sigmoid(model(data))
        output = output.data.cpu().numpy()
        target_data = target.data.cpu().numpy()

        if np.random.rand() > 0.999 and print_samples:
            print("sample prediction")
            print("Y_true: " + str(target_data[0]))
            print("Y_hat: " + str(output[0]))
            output = np.round(output)
            print("Y_hat: " + str(output[0]))
            print

        output = np.round(output)
        y.append(target_data)
        yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    metrics, fpr, tpr = evaluation.all_metrics(yhat, y)
    evaluation.print_metrics(metrics)
    return metrics, fpr, tpr

#yield some tensors. file should hold data sorted by sequence length, for batching
def data_generator(filename, batch_size, Y):
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_insts = []
        cur_labels = []
        cur_length = 0
        for row in r:
            #find the next batch_size instances with the same length
            text = row[1]
            length = int(row[3])
            if length > cur_length:
                if len(cur_insts) > 0:
                    #create the tensors
                    yield torch.LongTensor(cur_insts), torch.FloatTensor(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[2].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])
                #reset length
                cur_length = length
            else:
                if len(cur_insts) == batch_size:
                    #create the tensors
                    yield torch.LongTensor(cur_insts), torch.FloatTensor(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[2].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])

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
    parser.add_argument("objective", type=str, choices=["warp", "bce"], help="which objective")
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
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True)
    args = parser.parse_args()
    ok, msg = check_args(args)
    if ok:
        main(args.Y, args.vocab_min, args.dataset, args.model, args.n_epochs, args.objective, args.filter_size, args.min_filter,
             args.max_filter, args.conv_dim_factor, args.saved_dir, args.gpu)
    else:
        print(msg)
        sys.exit(0)

