"""
    Train a model with PyTorch
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from constants import *
import datasets
import evaluation
import persistence
import next_codes.models as models

import csv
import argparse
import os
import numpy as np
import random
import sys
import time

def main(Y, model_name, n_epochs, objective, min_filter, max_filter,
         num_filter_maps, saved_dir, data_path, gpu, stochastic):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)
    if model_name == "text_only":
        model = models.MultiConv(Y, min_filter, max_filter, num_filter_maps)
        if gpu:
            model.cuda()
    elif model_name == "codes_only":
        model = models.CodesMLP(Y, Y)
        if gpu:
            model.cuda()
    elif model_name == "combined":
        model = models.Combiner(Y, min_filter, max_filter, num_filter_maps)
        if gpu:
            model.cuda()
    elif model_name == "saved":
        model, filter_size, min_filter, max_filter, num_filter_maps, prev_epochs, prev_metrics, prev_dataset\
            = persistence.load_model(saved_dir, Y, 3)
        sys.exit(0)

    #optimizer = optim.SGD(model.parameters(), lr=.001)
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.Adadelta(model.parameters(), rho=0.95)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adagrad(model.parameters(), lr=.01)

    model_dir = os.path.join(MODEL_DIR, '_'.join(["next_codes", model_name, time.strftime('%b_%d_%H:%M', time.gmtime())]))

    #load vocab
    dicts = None
    if model != "codes_only":
        v_dict, c_dict = datasets.load_lookups("full", Y, 3)
        desc_dict = datasets.load_code_descriptions()
        dicts = (v_dict, c_dict, desc_dict)

    #initialize metrics stuff
    min_size = max_filter if model_name != "codes_only" else 0
    names = ["acc", "prec", "rec", "f1", "auc"]
    names.extend(["%s_micro" % (name) for name in names])
    metrics_hist = {name: [] for name in names}
    metrics_hist_tr = {name: [] for name in names}
    metrics_hist["loss"] = []

    #preload if doing stochastic
    if stochastic:
        insts = load_insts(model, Y, data_path, BATCH_SIZE)

    for epoch in range(n_epochs):
        if stochastic:
            loss = train_stochastic(model, insts, optimizer, Y, epoch, data_path, min_size, gpu, objective)
        else:
            loss = train(model, optimizer, Y, epoch, data_path, min_size, gpu, objective)
        print("epoch loss: " + str(loss))
        metrics_hist["loss"].append(loss)
        print("evaluating on dev")
        metrics, fpr, tpr = test(model, Y, epoch, data_path, "dev", min_size, gpu, dicts=dicts, print_samples=False)
        for name in names:
            #print(metrics)
            if name in metrics.keys():
                metrics_hist[name].append(metrics[name])

        print("sanity check on train")
        metrics_t, _, _ = test(model, Y, epoch, data_path, "train", min_size, gpu, print_samples=False)
        for name in names:
            if name in metrics_t.keys():
                metrics_hist_tr[name].append(metrics_t[name])
        #save metric history, model, params
        if model_name != "saved":
            if epoch == 0:
                os.mkdir(model_dir)
            persistence.save_metrics(metrics_hist, metrics_hist_tr, model_dir)
            persistence.save_params(model_dir, Y, 3, data_path, model_name, n_epochs, "torch", 0, 
                                    min_filter, max_filter, num_filter_maps)
            torch.save(model, model_dir + "/model.pth")

    if model == "saved" and data_path != prev_dataset:
        response = raw_input("***WARNING*** you ran the saved model on a different dataset than it was previously trained on. Overwrite? (y/n) > ")
        if "y" not in response:
            print("not saving any results (model, params, or metrics)")
            sys.exit(0)
        dataset = ",".join([prev_dataset, dataset])
        #overwrite old files w/ new values, rename folder
        persistence.rewrite_metrics(prev_metrics, metrics_hist, metrics_hist_tr, saved_dir)
        persistence.rewrite_params(saved_dir, dataset, prev_epochs+n_epochs)
        torch.save(model, saved_dir + "/model.h5")


def train(model, optimizer, Y, epoch, data_path, min_size, gpu, objective):
    filename = "%s/future_codes_text_%s_train.csv" % (DATA_DIR, Y) if data_path is None else data_path
    #put model in "train" mode
    model.train()
    losses = []
    if objective == "warp":
        batch_size = 1
        print_every = 5000
    else:
        batch_size = BATCH_SIZE
        print_every = 1000
    if type(model) == models.Combiner or type(model) == models.MultiConv:
        gen = datasets.next_codes_text_generator(filename, batch_size, Y)
        print_every = 50
    else:
        gen = datasets.codes_only_generator(filename, batch_size, Y)
    for batch_idx, (data, target) in enumerate(gen):
        if type(model) == models.Combiner or type(model) == models.MultiConv:
            data = (Variable(torch.LongTensor(data[0])), Variable(torch.LongTensor(data[1])))
            target = Variable(torch.FloatTensor(target))
            batch_size = data[1].size()[0]
            seq_len = data[1].size()[1]
        else:
            data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
            batch_size = data.size()[0]
            seq_len = data.size()[1]
#        if type(model) == models.Combiner or type(model) == models.MultiConv and data.size()[1] < min_size:
#            continue
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #clear gradients
        optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()
        losses.append(loss.data[0])
        if batch_idx % print_every == 0 and type(model) != models.CodesMLP:
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, batch_size, seq_len, np.mean(losses)))
    return np.mean(losses)

def load_insts(model, Y, data_path, batch_size):
    filename = "%s/future_codes_text_%s_train.csv" % (DATA_DIR, Y) if data_path is None else data_path
    print("loading all instances...")
    insts = []
    if type(model) == models.Combiner or type(model) == models.MultiConv:
        gen = datasets.next_codes_text_generator(filename, batch_size, Y)
    else:
        gen = datasets.codes_only_generator(filename, batch_size, Y)
    for batch_idx, (data, target) in enumerate(gen):
        insts.append((data, target))
    return insts 

def train_stochastic(model, insts, optimizer, Y, epoch, dataset, min_size, gpu, objective):
    #put model in "train" mode
    model.train()
    losses = []
    if objective == "warp":
        print_every = 5000
    else:
        print_every = 1000
    np.random.shuffle(insts)
    for batch_idx, (data, target) in enumerate(insts):
        if type(model) == models.Combiner or type(model) == models.MultiConv:
            data = (Variable(torch.LongTensor(data[0])), Variable(torch.LongTensor(data[1])))
            target = Variable(torch.FloatTensor(target))
        else:
            data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        if data.size()[1] < min_size:
            continue
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #clear gradients
        optimizer.zero_grad()
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
            loss.backward()
            optimizer.step()
        losses.append(loss.data[0])
        if batch_idx % print_every == 0 and type(model) != models.CodesMLP:
            print("Train epoch: {} [inst #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.size()[0], data.size()[1], np.mean(losses)))
    return np.mean(losses)

def test(model, Y, epoch, data_path, fold, min_size, gpu, dicts=None, print_samples=True):
    filename = "%s/future_codes_text_%s_%s.csv" % (DATA_DIR, Y, fold) if data_path is None else data_path
    #put model in "test" mode
    model.eval()
    y = []
    yhat = []
    yhat_raw = []
    if type(model) == models.Combiner or type(model) == models.MultiConv:
        gen = datasets.next_codes_text_generator(filename, BATCH_SIZE, Y)
    else:
        gen = datasets.codes_only_generator(filename, 1, Y)
    for batch_idx, (data, target) in enumerate(gen):
        if type(model) == models.Combiner or type(model) == models.MultiConv:
            data = (Variable(torch.LongTensor(data[0]), volatile=True), Variable(torch.LongTensor(data[1]), volatile=True))
            target = Variable(torch.FloatTensor(target))
        else:
            data, target = Variable(torch.LongTensor(data), volatile=True), Variable(torch.FloatTensor(target))
            #print("data: " + str(data))
            #print("target: " + str(target))
        if data.size()[1] < min_size:
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
        if dicts is not None and print_samples:
            v_dict, c_dict, desc_dict = dicts
            if margin_worse_than(-0.5, output[0], target_data[0]):
                print("did bad on this one")
                print("Y_true: " + str(target_data[0]))
                print("Y_hat: " + str(output[0]))
                print("first 100 words:")
                words = [v_dict[w] for w in data.data[0,:]]
                print(" ".join(words[:100]))
                codes = [str(c_dict[code]) for code in np.where(target_data[0] == 1)[0]]
                print("codes / descriptions")
                print(", ".join([code + ": " + desc_dict[code] for code in codes]))
                print
            if margin_better_than(0.5, output[0], target_data[0]):
                print("did good on this one")
                print("Y_true: " + str(target_data[0]))
                print("Y_hat: " + str(output[0]))
                print("first 100 words:")
                words = [v_dict[w] for w in data.data[0,:]]
                print(" ".join(words[:100]))
                codes = [str(c_dict[code]) for code in np.where(target_data[0] == 1)[0]]
                print("codes / descriptions")
                print(", ".join([code + ": " + desc_dict[code] for code in codes]))
                print

        output = np.round(output)
        y.append(target_data)
        yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    metrics, fpr, tpr = evaluation.all_metrics(yhat, y)
    if epoch % 1 == 0 and fpr is not None and tpr is not None:
        evaluation.print_metrics(metrics)
    return metrics, fpr, tpr

def margin_worse_than(margin, output, target):
    min_true = sys.maxint
    max_false = -1*sys.maxint
    for i in range(len(target)):
        if target[i] == 1 and output[i] < min_true:
            min_true = output[i]
        elif target[i] == 0 and output[i] > max_false:
            max_false = output[i]
    return (min_true - max_false) < margin 

def margin_better_than(margin, output, target):
    min_true = sys.maxint
    max_false = -1*sys.maxint
    for i in range(len(target)):
        if target[i] == 1 and output[i] < min_true:
            min_true = output[i]
        elif target[i] == 0 and output[i] > max_false:
            max_false = output[i]
    return (min_true - max_false) > margin

def check_args(args):
    if args.model == "saved" and args.saved_dir is None:
        return False, "Specified 'saved' but no model path given"
    if (args.model == "text_only" or args.model == "combined") \
        and (args.min_filter is None or args.max_filter is None or args.num_filter_maps is None):
        return False, "Specified 'cnn_multi', but (min_filter, max_filter, num_filter_maps) not fully specified"
    else:
        return True, "OK"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Y", type=int, help="size of label space")
    parser.add_argument("model", type=str, choices=["codes_only", "text_only", "combined", "saved"], 
                        help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("objective", type=str, choices=["warp", "bce"], help="which objective")
    parser.add_argument("--min-filter", type=int, required=False, dest="min_filter",
                        help="min size of filter range to use (cnn_multi only)")
    parser.add_argument("--max-filter", type=int, required=False, dest="max_filter",
                        help="max size of filter range to use (cnn_multi only)")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps",
                        help="size of conv output")
    parser.add_argument("--saved-model", type=str, required=False, dest="saved_dir",
                        help="path to a directory containing a saved model (and params and metrics) to load instead of building one")
    parser.add_argument("--data-path", type=str, required=False, dest="data_path",
                        help="optional path to a file containing sorted data. will go to DATA_DIR/notes_Y_train.csv by default")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--stochastic", dest="stochastic", action="store_const", required=False, const=True,
                        help="optional flag to randomly sample data instead of running through it sequentially")
    args = parser.parse_args()
    ok, msg = check_args(args)
    if ok:
        main(args.Y, args.model, args.n_epochs, args.objective, args.min_filter, args.max_filter,
                args.num_filter_maps, args.saved_dir, args.data_path, args.gpu, args.stochastic)
    else:
        print(msg)
        sys.exit(0)

