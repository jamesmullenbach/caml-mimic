"""
    Reads (or writes) BOW-formatted notes and performs scikit-learn logistic regression
"""
import csv
import numpy as np
import os
import pickle
import sys
import time

from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from constants import *
import datasets
import evaluation
from learn import tools
import persistence

import nltk

#Constants
C = 1.0
MAX_ITER = 20

def main(Y, train_fname, dev_fname, vocab_file, version, n):
    n = int(n)
 
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize)   

    #get lookups from non-BOW data
    data_path = train_fname.replace('_bows', '') if "_bows" in train_fname else train_fname
    dicts = datasets.load_lookups(data_path, vocab_file=vocab_file, Y=Y, version=version)
    w2ind, ind2c, c2ind = dicts['w2ind'], dicts['ind2c'], dicts['c2ind']

    X, yy_tr, hids_tr = read_bows(Y, train_fname, c2ind, version)
    X_dv, yy_dv, hids_dv = read_bows(Y, dev_fname, c2ind, version)

    print("X.shape: " + str(X.shape))
    print("yy_tr.shape: " + str(yy_tr.shape))
    print("X_dv.shape: " + str(X_dv.shape))
    print("yy_dv.shape: " + str(yy_dv.shape))

    #deal with labels that don't have any positive examples
    #drop empty columns from yy. keep track of which columns kept
    #predict on test data with those columns. guess 0 on the others
    labels_with_examples = yy_tr.sum(axis=0).nonzero()[0]
    yy = yy_tr[:, labels_with_examples]

    # build the classifier
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=MAX_ITER, solver='sag'), n_jobs=-1)

    # train
    print("training...")
    clf.fit(X, yy)

    #predict
    print("predicting...")
    yhat = clf.predict(X_dv)
    yhat_raw = clf.predict_proba(X_dv)

    #deal with labels that don't have positive training examples
    print("reshaping output to deal with labels missing from train set")
    labels_with_examples = set(labels_with_examples)
    yhat_full = np.zeros(yy_dv.shape)
    yhat_full_raw = np.zeros(yy_dv.shape)
    j = 0
    for i in range(yhat_full.shape[1]):
        if i in labels_with_examples:
            yhat_full[:,i] = yhat[:,j]
            yhat_full_raw[:,i] = yhat_raw[:,j]
            j += 1

    #evaluate
    metrics, fpr, tpr = evaluation.all_metrics(yhat_full, yy_dv, k=[8, 15], yhat_raw=yhat_full_raw)
    evaluation.print_metrics(metrics)

    #save metric history, model, params
    print("saving predictions")
    model_dir = os.path.join(MODEL_DIR, '_'.join(["log_reg", time.strftime('%b_%d_%H:%M', time.localtime())]))
    os.mkdir(model_dir)
    preds_file = tools.write_preds(yhat_full, model_dir, hids_dv, 'test', yhat_full_raw)

    print("sanity check on train")
    yhat_tr = clf.predict(X)
    yhat_tr_raw = clf.predict_proba(X)

    #reshape output again
    yhat_tr_full = np.zeros(yy_tr.shape)
    yhat_tr_full_raw = np.zeros(yy_tr.shape)
    j = 0
    for i in range(yhat_tr_full.shape[1]):
        if i in labels_with_examples:
            yhat_tr_full[:,i] = yhat_tr[:,j]
            yhat_tr_full_raw[:,i] = yhat_tr_raw[:,j]
            j += 1

    #evaluate
    metrics_tr, fpr_tr, tpr_tr = evaluation.all_metrics(yhat_tr_full, yy_tr, k=[8, 15], yhat_raw=yhat_tr_full_raw)
    evaluation.print_metrics(metrics_tr)

    if n > 0:
        print("generating top important ngrams")
        if 'bows' in dev_fname:
            dev_fname = dev_fname.replace('_bows', '')
        print("calculating top ngrams using file %s" % dev_fname)
        calculate_top_ngrams(dev_fname, clf, c2ind, w2ind, labels_with_examples, n)

    #Commenting this out because the models are huge (11G for mimic3 full)
    #print("saving model")
    #with open("%s/model.pkl" % model_dir, 'wb') as f:
    #    pickle.dump(clf, f)

    print("saving metrics")
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])
    for name in metrics.keys():
        metrics_hist[name].append(metrics[name])
    for name in metrics_tr.keys():
        metrics_hist_tr[name].append(metrics_tr[name])
    metrics_hist_all = (metrics_hist, metrics_hist, metrics_hist_tr)
    persistence.save_metrics(metrics_hist_all, model_dir)


def write_bows(data_fname, X, hadm_ids, y, ind2c):
    out_name = data_fname.split('.csv')[0] + '_bows.csv'
    with open(out_name, 'w') as of:
        w = csv.writer(of)
        w.writerow(['HADM_ID', 'BOW', 'LABELS'])
        for i in range(X.shape[0]):
            bow = X[i].toarray()[0]
            inds = bow.nonzero()[0]
            counts = bow[inds]
            bow_str = ' '.join(['%d:%d' % (ind, count) for ind,count in zip(inds,counts)])
            code_str = ';'.join([ind2c[ind] for ind in y[i].nonzero()[0]])
            w.writerow([str(hadm_ids[i]), bow_str, code_str])

def read_bows(Y, bow_fname, c2ind, version):
    num_labels = len(c2ind)
    data = []
    row_ind = []
    col_ind = []
    hids = []
    y = []
    with open(bow_fname, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for i,row in tqdm(enumerate(r)):
            hid = int(row[0])
            bow_str = row[1]
            code_str = row[2]
            for pair in bow_str.split():
                split = pair.split(':')
                ind, count = split[0], split[1]
                data.append(int(count))
                row_ind.append(i)
                col_ind.append(int(ind))
            label_set = set([c2ind[c] for c in code_str.split(';')])
            y.append([1 if j in label_set else 0 for j in range(num_labels)])
            hids.append(hid)
        X = csr_matrix((data, (row_ind, col_ind)))
    return X, np.array(y), hids


def construct_X_Y(notefile, Y, w2ind, c2ind, version):
    """
        Each row consists of text pertaining to one admission
        INPUTS:
            notefile: path to file containing note data
            Y: size of the output label space
            w2ind: dictionary from words to integers for discretizing
            c2ind: dictionary from labels to integers for discretizing
            version: which (MIMIC) dataset
        OUTPUTS: 
            csr_matrix where each row is a BOW
                Dimension: (# instances in dataset) x (vocab size)
    """
    Y = len(c2ind)
    yy = []
    hadm_ids = []
    with open(notefile, 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)
        i = 0

        subj_inds = []
        indices = []
        data = []

        for i,row in tqdm(enumerate(reader)):
            label_set = set()
            for l in str(row[3]).split(';'):
                if l in c2ind.keys():
                    label_set.add(c2ind[l])
            subj_inds.append(len(indices))
            yy.append([1 if j in label_set else 0 for j in range(Y)])
            text = row[2]
            for word in text.split():
                if word in w2ind:
                    index = w2ind[word]
                    if index != 0:
                        #ignore padding characters
                        indices.append(index)
                        data.append(1)
                else:
                    #OOV
                    indices.append(len(w2ind))
                    data.append(1)
            i += 1
            hadm_ids.append(int(row[1]))
        subj_inds.append(len(indices))

    return csr_matrix((data, indices, subj_inds)), np.array(yy), hadm_ids

def calculate_top_ngrams(inputfile, clf, c2ind, w2ind, labels_with_examples, n):
    
    #Reshape the coefficients matrix back into having 0's for columns of codes not in training set.
    labels_with_examples = set(labels_with_examples)
    mat = clf.coef_
    mat_full = np.zeros((8922, mat.shape[1]))
    j = 0
    for i in range(mat_full.shape[0]):
        if i in labels_with_examples:
            mat_full[i,:] = mat[j,:]
            j += 1

    #write out to csv
    f = open("%s/top_ngrams.csv" % DATA_DIR, 'wb')
    writer = csv.writer(f, delimiter = ',')
    #write header
    writer.writerow(['SUBJECT_ID', 'HADM_ID', 'LABEL', 'INDEX', 'NGRAM', 'SCORE'])
            
    #get text as list of strings for each record in dev set
    with open("%s" % (inputfile), 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)

        all_rows = []
        for i,row in tqdm(enumerate(reader)):
                        
            text = row[2]
            hadm_id = row[1]
            subject_id = row[0]
            labels = row[3].split(';')
            
            #for each text, label pair, calculate heighest weighted n-gram in text
            for label in labels:
                myList = []
                
                #subject id
                myList.append(subject_id)
                #hadm id
                myList.append(hadm_id)

                #augmented coefficients matrix has dims (5000, 51918) (num. labels, size vocab.)
                #get row corresponding to label:
                word_weights = mat_full[c2ind[label]]
                
                #get each set of n grams in text
                #get ngrams
                fourgrams = nltk.ngrams(text.split(), n)
                fourgrams_scores = []
                for grams in fourgrams:
                    #calculate score
                    sum_weights = 0
                    for word in grams:
                        if word in w2ind:
                            inx = w2ind[word]
                            #add coeff from logistic regression matrix for given word
                            sum_weights = sum_weights + word_weights[inx]
                        else:
                            #else if word not in vocab, adds 0 weight
                            pass
                    fourgrams_scores.append(sum_weights)
                 
                #get the fourgram itself
                w = [word for word in text.split()][fourgrams_scores.index(max(fourgrams_scores)):fourgrams_scores.index(max(fourgrams_scores))+n]
                    
                #label
                myList.append(label)
                #start index of 4-gram
                myList.append(fourgrams_scores.index(max(fourgrams_scores)))
                #4-gram
                myList.append(" ".join(w))
                #sum weighted score (highest)
                myList.append(max(fourgrams_scores))
            
                writer.writerow(myList)
                
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [train_dataset] [dev_dataset] [vocab_file] [version] [size of ngrams (0 if do not wish to generate)]"))
        sys.exit(0)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])

