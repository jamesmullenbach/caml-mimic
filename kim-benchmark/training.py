import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from constants import *
import data_helpers
import evaluation
import models

from collections import defaultdict
import cPickle
import csv
import argparse
import os
import numpy as np
import sys
import time

def main(model_name, n_epochs, norm_constraint, filter_size, min_filter, max_filter, num_filter_maps, gpu):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """

    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    idx_word_map = {idx: word for word, idx in word_idx_map.iteritems()}

    if model_name == "cnn_multi":
        model = models.MultiConv(torch.FloatTensor(W2), min_filter, max_filter, num_filter_maps, norm_constraint)
        if gpu:
            model.cuda()
    elif model_name == "cnn_vanilla":
        model = models.VanillaConv(torch.FloatTensor(W2), filter_size, num_filter_maps)
        if gpu:
            model.cuda()

    optimizer = optim.Adadelta(model.parameters(), rho=0.95)
    batch_size = BATCH_SIZE
  
    test_bows = []
    train_bows = []
    dev_bows = []
    for fold in range(10-1):
        datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=300, filter_h=5)
#        datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=300, filter_h=5)
        img_h = len(datasets[0][0]) - 1
        test_set_x = datasets[2][:,:img_h]
        test_set_y = np.asarray(datasets[2][:,-1],"int64")
        np.random.seed(3435)
#        if datasets[0].shape[0] % batch_size > 0:
#            print("padding with multiple data")
#            extra_data_num = batch_size - datasets[0].shape[0] % batch_size
#            #permute rows
#            train_set = np.random.permutation(datasets[0])   
#            extra_data = train_set[:extra_data_num]
#            new_data=np.append(datasets[0],extra_data,axis=0)
#        else:
#            new_data = datasets[0]
#        new_data = datasets[0]
        new_data = np.random.permutation(datasets[0])

        n_batches = new_data.shape[0]/batch_size
        n_train_batches = int(np.round(n_batches*0.9))

#        train_set = new_data[:n_train_batches*batch_size,:]
#        val_set = new_data[n_train_batches*batch_size:,:]
        train_set = new_data
        val_set = np.random.permutation(datasets[1])
        print("size of train set: " + str(train_set.shape))
        print("size of val set: " + str(val_set.shape))
        train_set_x, train_set_y = train_set[:,:img_h], train_set[:,-1]
        val_set_x, val_set_y = val_set[:,:img_h], val_set[:,-1]
#        for i in range(0, val_set_x.shape[0], BATCH_SIZE):
#            if np.equal(val_set_y[i:i+BATCH_SIZE], 0).all():
#                print("val labels all 0. size: " + str(val_set_y[i:i+BATCH_SIZE]))
#        for i,sent in enumerate(train_set_x):
#            for i2, sent2 in enumerate(val_set_x):
#                if np.equal(sent, sent2).all():
#                    print("match w/ train and dev: fold: %d" % (fold))
#        for i, sent in enumerate(test_set_x):
#            for i2, sent2 in enumerate(train_set_x):
#                if np.equal(sent, sent2).all():
#                    print("match w/ test and train: fold: %d" % (fold))
#            for i3, sent3 in enumerate(val_set_x):
#                if np.equal(sent, sent3).all():
#                    print("match w/ test and dev: fold: %d" % (fold))
        best_val_perf = 0.
        for epoch in range(n_epochs): 
    
            tr_acc, tr_loss = train(train_set_x, train_set_y, model, optimizer, epoch, fold, gpu)
            print("epoch: %d, fold: %d, tr_acc: %f, 'perf': %.2f" % (epoch, fold, tr_acc, 100.*(1.-tr_loss)))
            dv_acc, dv_loss, failed_examples = test(val_set_x, val_set_y, model, epoch, gpu, idx_word_map)
            dv_perf = 100.*(1.-dv_loss)
            print("epoch: %d, fold: %d, dv_acc: %f, 'perf': %.2f" % (epoch, fold, dv_acc, dv_perf))
            if len(failed_examples) > 0:
                print("failed example: " + str(failed_examples[0]))
            if dv_perf > best_val_perf:
                best_val_perf = dv_perf
                test_acc, test_loss, _ = test(test_set_x, test_set_y, model, epoch, gpu, idx_word_map)
                print("epoch: %d, test_acc: %f, 'perf': %.2f" % (epoch, test_acc, 100.*(1.-test_loss)))

        test_acc, test_loss, failed_examples = test(test_set_x, test_set_y, model, epoch, gpu, idx_word_map)
        print("\nFINAL TEST fold: %d, test_acc: %f, 'perf': %.2f\n" % (fold, test_acc, 100.*(1.-test_loss)))
        if len(failed_examples) > 0:
            print("failed example: " + str(failed_examples[0]))

def train(x, y, model, optimizer, epoch, fold, gpu):
    #put model in "train" mode
    model.train()
    y_true = []
    y_hat = []
    losses = []
    for i in range(0, x.shape[0], BATCH_SIZE):
        data, target = torch.LongTensor(x[i:i+BATCH_SIZE,:]), torch.LongTensor(y[i:i+BATCH_SIZE])
        data, target = Variable(data), Variable(target)
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #clear gradients
        optimizer.zero_grad()
        #forward computation
        output = model(data)
        loss = F.cross_entropy(output, target)
        #backward pass
        loss.backward()
        optimizer.step()

        output = output.data.cpu().numpy()
        target_data = target.data.cpu().numpy()
        y_true.append(target_data)
        y_hat.append(np.argmax(output, axis=1))
        losses.append(loss.data[0])

        model.enforce_norm_constraint()

    print("Train epoch: {} fold: {} \tLoss: {:.6f}".format(epoch+1, fold, np.mean(losses)))
    y_true = np.concatenate(y_true, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)
    acc = np.equal(y_true, y_hat).sum() / float(len(y_true))
    return acc, np.mean(losses)

def test(x, y, model, epoch, gpu, idx_word_map):
    #put model in "test" mode (sets dropout prob to 0)
    model.eval()
    y_true = []
    y_hat = []
    losses = []
    failed_examples = []
    for i in range(0, x.shape[0], BATCH_SIZE):
        if x[i:i+BATCH_SIZE,:].dtype != np.int64 or y[i:i+BATCH_SIZE].dtype != np.int64:
            continue
        data, target = torch.LongTensor(x[i:i+BATCH_SIZE,:]), torch.LongTensor(y[i:i+BATCH_SIZE])
        #volatile flag is an optimizer for inference mode. model won't do backward step when volatile=True
        data, target = Variable(data, volatile=True), Variable(target)
        if gpu:
            #gpu-ify
            data = data.cuda()
            target = target.cuda()
        #predict
        output = model(data)
        loss = F.cross_entropy(output, target)
        output = output.data.cpu().numpy()
        target_data = target.data.cpu().numpy()

        output = np.argmax(output, axis=1)
        if np.random.rand() > 0.999999:
            print("output: " + str(output))
            print("target: " + str(target_data))
        if np.not_equal(output, target_data).any() and np.random.rand() > 0.999999:
            print("output: " + str(output))
            print("target: " + str(target_data))
            inds = np.where(np.not_equal(output, target_data))[0]
            example = data.data.cpu().numpy()[inds[0]]
            words = [idx_word_map[w] for w in example[np.nonzero(example)]]
            print("words: " + str(words))
            print("target_data[inds[0]]: " + str(target_data[inds[0]]))
            print("predicted output: " + str(output[inds[0]]))
            failed_examples.append((words, target_data[inds[0]]))
        y_true.append(target_data)
        y_hat.append(output)
        losses.append(loss.data[0])

    y_true = np.concatenate(y_true, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)
    if np.random.rand() > 0.99999:
        print("ytru: " + str(y_true))
        print("yhat: " + str(y_hat))
    acc = np.equal(y_true, y_hat).sum() / float(len(y_true))
    return acc, np.mean(losses), failed_examples

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            val.append(sent)
        elif rev["split"] == 9:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, val, test]

def check_args(args):
    if args.model == "saved" and args.saved_dir is None:
        return False, "Specified 'saved' but no model path given"
    if args.model == "cnn_vanilla" and args.filter_size is None:
        return False, "Specified 'cnn_vanilla' but no filter size given"
    if args.model == "cnn_multi" and (args.min_filter is None or args.max_filter is None):
        return False, "Specified 'cnn_multi', but (min_filter, max_filter) not fully specified"
    if (args.model == "cnn_vanilla" or args.model == "cnn_multi") and args.num_filter_maps is None:
        return False, "Specified a cnn model but no conv_dim_factor given"
    else:
        return True, "OK"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["cnn_vanilla", "cnn_multi", "lstm", "saved"], 
                        help="model ('cnn_vanilla', 'cnn_multi', 'lstm')")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("norm_constraint", type=int, help="l2 norm of weight vectors should be less than this value")
    parser.add_argument("--filter-size", type=int, required=False, dest="filter_size",
                        help="size of convolution filter to use (cnn_vanilla only)")
    parser.add_argument("--min-filter", type=int, required=False, dest="min_filter",
                        help="min size of filter range to use (cnn_multi only)")
    parser.add_argument("--max-filter", type=int, required=False, dest="max_filter",
                        help="max size of filter range to use (cnn_multi only)")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps",
                        help="size of conv output")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True)
    args = parser.parse_args()
    ok, msg = check_args(args)
    if ok:
        main(args.model, args.n_epochs, args.norm_constraint, args.filter_size, args.min_filter,
                args.max_filter, args.num_filter_maps, args.gpu)
    else:
        print(msg)
        sys.exit(0)

