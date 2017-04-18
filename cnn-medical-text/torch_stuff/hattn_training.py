"""
    Separate training method for hattn model... cause
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
import torch_stuff.models as models
import torch_stuff.objectives as objectives

import csv
import argparse
import os
import numpy as np
import random
import sys
import time

def main(Y, vocab_min, n_epochs, objective, word_lstm_dim, doc_lstm_dim, saved_dir, data_path, gpu, stochastic):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)
    model = models.HAN(Y, word_lstm_dim, doc_lstm_dim, BATCH_SIZE, gpu)
    if gpu:
        model.cuda()

    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.Adadelta(model.parameters(), rho=0.95)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adagrad(model.parameters(), lr=.01)

    model_dir = os.path.join(MODEL_DIR, '_'.join(["hattn", time.strftime('%b_%d_%H:%M', time.gmtime())]))

    #load vocab
    v_dict, c_dict = datasets.load_lookups("full", Y, vocab_min)
    desc_dict = datasets.load_code_descriptions()
    dicts = (v_dict, c_dict, desc_dict)

    names = ["acc", "prec", "rec", "f1", "auc"]
    names.extend(["%s_micro" % (name) for name in names])
    metrics_hist = {name: [] for name in names}
    metrics_hist_tr = {name: [] for name in names}
    metrics_hist["loss"] = []
    model_name = "hattn"
    for epoch in range(n_epochs):
        loss = train(model, optimizer, Y, epoch, data_path, gpu, objective)
        print("epoch loss: " + str(loss))
        metrics_hist["loss"].append(loss)
        print("evaluating on dev")
        metrics, fpr, tpr = test(model, Y, epoch, data_path, "dev", gpu, dicts=dicts)
        for name in names:
            metrics_hist[name].append(metrics[name])

        print("sanity check on train")
        metrics_t, _, _ = test(model, Y, epoch, data_path, "train", gpu, print_samples=False)
        for name in names:
            metrics_hist_tr[name].append(metrics_t[name])
        #save metric history, model, params
        if epoch == 0 and saved_dir is None: 
            os.mkdir(model_dir)
        persistence.save_metrics(metrics_hist, metrics_hist_tr, model_dir)
        params = {"model_dir": model_dir, "Y": Y, "vocab_min": vocab_min, "data_path": data_path, "doc_lstm_dim": doc_lstm_dim,
                  "model_name": model_name, "word_lstm_dim": word_lstm_dim, "stochastic": stochastic, "n_epochs": n_epochs,
                  "objective": objective}
        persistence.save_params_dict(params)
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

def train(model, optimizer, Y, epoch, data_path, gpu, objective):
    filename = DATA_DIR + "/notes_" + str(Y) + "_train_attn.csv" if data_path is None else data_path
    #put model in "train" mode
    model.train()
    losses = []
    if objective == "warp":
        batch_size = 1
        print_every = 5000
    else:
        batch_size = BATCH_SIZE
        print_every = 1
    notes_proc = 0
    for batch_idx, (data, target) in enumerate(datasets.attn_generator(filename, batch_size, Y)):
        #data is a list of lists of notes
        target = Variable(torch.FloatTensor(target))
        if gpu:
            target = target.cuda()
        #clear gradients
        optimizer.zero_grad()
        #model.zero_grad()
        #forward computation
        model.refresh(data.shape[0])
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
        notes_proc += data.shape[0] * data.shape[1]
        if batch_idx % print_every == 0:
            print("Train epoch: {} [batch #{}, batch_size {}, num notes {}, notes processed {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.shape[0], data.shape[1], notes_proc,  np.mean(losses)))
    return np.mean(losses)

def test(model, Y, epoch, data_path, fold, gpu, dicts=None, print_samples=True):
    filename = DATA_DIR + "/notes_" + str(Y) + "_" + fold + "_attn.csv" if data_path is None else data_path.replace("train", fold)
    #put model in "test" mode
    model.eval()
    y = []
    yhat = []
    yhat_raw = []
    notes_proc = 0
    if fold == "train":
        max_to_generate = 100
    else:
        max_to_generate = sys.maxint
    for batch_idx, (data, target) in enumerate(datasets.attn_generator(filename, BATCH_SIZE, Y, max_to_generate)):
        #data is a list of lists of notes
        target = Variable(torch.FloatTensor(target))
        if gpu:
            target = target.cuda()
        #clear gradients
        model.zero_grad()
        #predict
        model.refresh(data.shape[0])
        word_importances, doc_importances = None, None
        if random.random() > 0.999999999999:
            output, word_importances, doc_importances = model(data, get_importances=True, volatile=True)
            output = F.sigmoid(output)
        else:
            output = F.sigmoid(model(data, volatile=True))
        output = output.data.cpu().numpy()
        target_data = target.data.cpu().numpy()

        if dicts is not None and print_samples:
            print_examples(dicts, data, output, target_data, batch_idx, word_importances, doc_importances)

        output = np.round(output)
        y.append(target_data)
        yhat.append(output)

    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    metrics, fpr, tpr = evaluation.all_metrics(yhat, y)
    if epoch % 1 == 0:
        evaluation.print_metrics(metrics)
    return metrics, fpr, tpr

def print_examples(dicts, data, output, target_data, batch_idx, word_imps=None, doc_imps=None):
    v_dict, c_dict, desc_dict = dicts
    if margin_worse_than(-1.0, output[0], target_data[0]):
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
    if margin_better_than(1.0, output[0], target_data[0]):
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
        if doc_imps is not None and word_imps is not None:
            print_important_words(word_imps, doc_imps, v_dict, data)
    if random.random() > 0.99 or batch_idx == 0:
        print("random sample")
        print("Y_true: " + str(target_data[0]))
        print("Y_hat: " + str(output[0]))
        output = np.round(output)
        print("Y_hat: " + str(output[0]))
        
        if doc_imps is not None and word_imps is not None and v_dict is not None:
            print_important_words(word_imps, doc_imps, v_dict, data)

def print_important_words(word_imps, doc_imps, v_dict, data):
    doc_imps = doc_imps.data.cpu().numpy()
    imp_doc = np.unravel_index(doc_imps.argmax(), doc_imps.shape)
    imp_words = word_imps[imp_doc[0]][imp_doc[1]].data.cpu().numpy()
    a = imp_words.argmax()
    print("most important word and 5 words surrounding it")
    imps = imp_words[a-5:a+6]
    words = [v_dict[d] for d in data[imp_doc[0]][imp_doc[1]][a-5:a+6]]
    for i in range(len(imps)):
        print("{:0.6f}  ".format(imps[i])),
    print
    for i in range(len(words)):
        print("{0: <9} ".format(words[i][:9])),
    print


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Y", type=int, help="size of label space")
    parser.add_argument("vocab_min", type=int, help="vocab parameter (min # occurrences)")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("objective", type=str, choices=["warp", "bce"], help="which objective")
    parser.add_argument("--word-lstm-dim", type=int, required=False, dest="word_lstm_dim",
                        help="size of word lstm dim")
    parser.add_argument("--doc-lstm-dim", type=int, required=False, dest="doc_lstm_dim",
                        help="size of doc lstm dim")
    parser.add_argument("--saved-model", type=str, required=False, dest="saved_dir",
                        help="path to a directory containing a saved model (and params and metrics) to load instead of building one")
    parser.add_argument("--data-path", type=str, required=False, dest="data_path",
                        help="optional path to a file containing sorted data. will go to DATA_DIR/notes_Y_train.csv by default")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--stochastic", dest="stochastic", action="store_const", required=False, const=True,
                        help="optional flag to randomly sample data instead of running through it sequentially")
    args = parser.parse_args()
    main(args.Y, args.vocab_min, args.n_epochs, args.objective, args.word_lstm_dim, args.doc_lstm_dim,
            args.saved_dir, args.data_path, args.gpu, args.stochastic)

