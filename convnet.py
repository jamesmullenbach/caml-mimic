"""
    Runs a ConvNet over the data to predict ICD-9 diagnosis codes
    options for single window or multi-window
    
    Framework: Keras
"""
from collections import defaultdict
import csv
import os
import numpy as np
import sys

import evaluation

from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence

from scipy import interp
from sklearn.metrics import roc_curve, auc

#Embedding constants
VOCAB_SIZE = 40000
EMBEDDING_SIZE = 64
DROPOUT_EMBED = 0.5
DROPOUT_DENSE = 0.5

#Convolution constants
FILTER_SIZE = 3
STRIDE = 1
CONV_DIM_FACTOR = 10
POOL_LENGTH = 3

#training constants
BATCH_SIZE = 32
NUM_EPOCHS = 10


def main(Y, vocab_min, model_path, dataset):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    cnn = build_model(Y)
    #cnn = build_model_multiwindow(Y, 3, 5, 1)
    #cnn = build_model_multiconv(Y)

    history = []
    for i in range(NUM_EPOCHS):
        hist = train(cnn, i, dataset, Y)
        history.append(hist)
        metrics, fpr, tpr = test(cnn, dataset, Y)
    	print("[MACRO] accuracy, precision, recall, f-measure, AUC")
    	print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc_macro"])
    	print("[MICRO] accuracy, precision, recall, f-measure, AUC")
    	print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])
    	print
    
    cnn.save(model_path)

def build_model(Y):
    """
        Builds the single window CNN model
        params:
            Y: size of the label space
        returns:
            cnn: the CNN model
    """
    cnn = Sequential()
    #no input length bc it's not constant
    cnn.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED))
    cnn.add(Convolution1D(Y*CONV_DIM_FACTOR, FILTER_SIZE, activation='tanh'))
    from keras.layers.pooling import GlobalMaxPooling1D
    cnn.add(GlobalMaxPooling1D())
    #model.add(Dense(Y*CONV_DIM_FACTOR/2))
    #model.add(Dropout(DROPOUT_DENSE))
    cnn.add(Dense(Y))
    cnn.add(Dropout(DROPOUT_DENSE))
    cnn.add(Activation('sigmoid'))
    cnn.compile(optimizer='rmsprop', loss='binary_crossentropy')
    print(cnn.summary())
    return cnn 

def build_model_multiwindow(Y, s, l, step):
    """
        Builds the multi-window CNN model
        params:
            Y: size of the label space
            s: the smallest filter size
            l: the largest filter size
            step: size difference between consecutive filters
        returns:
            cnn_multi: the model
    """
    from keras.layers import Input, Merge
    from keras.models import Model
    embed = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED)

    convs = []

    #set up first embedding layer
    for i,sz in enumerate(range(s, l+1, step)):
        convs.append(Sequential())
    base_embed = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED)
    #convs[0].add(base_embed)

    #set up other embedding layers to share params with the first
    for i,sz in enumerate(range(s, l+1, step)):
        convs[i].add(embed)

    #add the conv layers
    for i,sz in enumerate(range(s, l+1, step)):
        convs[i].add(Convolution1D(Y*CONV_DIM_FACTOR, sz, activation='tanh'))
        from keras.layers.pooling import GlobalMaxPooling1D
        convs[i].add(GlobalMaxPooling1D())

    merged = Merge(convs, mode='concat', concat_axis=1) 

    cnn_multi = Sequential()
    cnn_multi.add(merged)
    cnn_multi.add(Dense(Y))
    cnn_multi.add(Dropout(DROPOUT_DENSE))
    cnn_multi.add(Activation('sigmoid'))
    
    cnn_multi.compile(optimizer='rmsprop', loss='binary_crossentropy')
    print(cnn_multi.summary())
    return cnn_multi

def train(cnn, epoch, dataset, Y):
    """
        Trains the model for one epoch on train data. Batches by length
        params:
            cnn: the model
            epoch: number of epochs previously trained + 1
            dataset: full or single
            Y: size of label space
        returns:
            history of losses (one per batch)
    """
    hist = []
    for idx, (X_batch, Y_batch) in enumerate(data_generator('../mimicdata/notes_%s_train_%s_sorted.csv'\
                                                                % (str(Y), dataset),\
                                                            BATCH_SIZE, Y)):
        loss = cnn.train_on_batch(X_batch, Y_batch)
        if idx % 100 == 0:
            print('Train Epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {}'.format(
                epoch+1, idx, X_batch.shape[0], X_batch.shape[1], loss))
        hist.append(loss)
    return hist

def test(cnn, dataset, Y, max_iter = 1e9):
    """
        Runs the model on dev data, computes a bunch of metrics
        params:
            cnn: the model
            dataset: full or single
            Y: size of the label space
            max_iter: (for debugging) max number of batches to predict on
        returns:
            metrics: contains acc, prec, rec, f1, auc measures
            fpr: false positive rate vector
            tpr: true positive rate vector
    """
    accs, precs, recs, f1s = [], [], [], []
    accs_micro, precs_micro, recs_micro, f1s_micro = [], [], [], []
    fprs, tprs = [], []
    batch_sizes = []
    all_fpr = np.array([])
    Y_tot = []
    Y_hat_tot = []
    fpr_tot = {i: [] for i in range(Y)}
    tpr_tot = {i: [] for i in range(Y)}
    for idx,(X_batch, Y_batch) in enumerate(data_generator('../mimicdata/notes_%s_dev_%s_sorted.csv'\
                                                               % (str(Y), dataset),\
                                                           BATCH_SIZE, Y)):
        if idx >= max_iter:
            break
        Y_hat = cnn.predict_on_batch(X_batch)
        Y_hat = np.round(Y_hat)

        Y_hat_tot.extend(Y_hat)
        Y_tot.extend(Y_batch)

        metrics, fpr, tpr = evaluation.all_metrics(Y_hat, Y_batch)
        batch_sizes.append(Y_batch.shape[0])

        accs.append(metrics['acc'])
        precs.append(metrics['prec'])
        recs.append(metrics['rec'])
        f1s.append(metrics['f1'])

        accs_micro.append(metrics['acc_micro'])
        precs_micro.append(metrics['prec_micro'])
        recs_micro.append(metrics['rec_micro'])
        f1s_micro.append(metrics['f1_micro'])

        if fpr is not None and tpr is not None:
            batch_fpr = np.unique(np.concatenate([fpr[i] for i in range(Y)]))
            all_fpr = np.concatenate((all_fpr, batch_fpr))
            for i in range(Y):
                fpr_tot[i].extend(fpr[i])
                tpr_tot[i].extend(tpr[i])
  
    metrics = {}
    Y_tot = np.array(Y_tot)
    Y_hat_tot = np.array(Y_hat_tot)
    #micro-AUC: just looks at each individual prediction
    fpr_micro, tpr_micro, _ = roc_curve(Y_tot.ravel(), Y_hat_tot.ravel())
    fpr_tot["micro"] = fpr_micro
    tpr_tot["micro"] = tpr_micro
    metrics["auc_micro"] = auc(fpr_tot["micro"], tpr_tot["micro"])

    #macro-AUC: kind of like an average of the ROC curves for all classes
    all_fpr = np.unique(all_fpr)
    all_fpr = all_fpr[~np.isnan(all_fpr)]
    mean_tpr = np.zeros_like(all_fpr)
    denom = 0
    for i in range(Y):
        val = interp(all_fpr, fpr_tot[i], tpr_tot[i])
        if not np.isnan(val).any():
            denom += 1
            mean_tpr += val
    if denom > 0:
        mean_tpr /= denom
    fpr_tot["macro"] = all_fpr
    tpr_tot["macro"] = mean_tpr
    metrics["auc_macro"] = auc(fpr_tot["macro"], tpr_tot["macro"])

    metrics['acc'] = np.average(accs, weights=batch_sizes)
    metrics['prec'] = np.average(precs, weights=batch_sizes)
    metrics['rec'] = np.average(recs, weights=batch_sizes)
    metrics['f1'] = np.average(f1s, weights=batch_sizes)

    metrics['acc_micro'] = np.average(accs_micro, weights=batch_sizes)
    metrics['prec_micro'] = np.average(precs_micro, weights=batch_sizes)
    metrics['rec_micro'] = np.average(recs_micro, weights=batch_sizes)
    metrics['f1_micro'] = np.average(f1s_micro, weights=batch_sizes)

    return metrics, fpr_tot, tpr_tot

def evaluate(cnn, X_dv, Y_dv):
    #predict, threshold, and get (macro) metrics
    preds = cnn.predict(X_dv)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    metrics, fpr, tpr = evaluation.all_metrics(preds, Y_dv)
    return preds, metrics, fpr, tpr

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

def get_version():
    #use the older version of keras when not running on my pc
    import socket
    return ("james" not in socket.gethostname())

def write_preds(preds, filename):
    #write predictions to a csv
    with open(filename, 'w') as outfile:
        for p in preds:
            outfile.write(','.join([str(p_i) for p_i in p]) + "\n")    

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
                    yield np.array(cur_insts), np.array(cur_labels)
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
                    yield np.array(cur_insts), np.array(cur_labels)
                    #clear
                    cur_insts = []
                    cur_labels = []
                cur_insts.append([int(w) for w in text.split()])
                labels = [int(l) for l in row[2].split(';')]
                cur_labels.append([1 if i in labels else 0 for i in range(Y)])

def load_lookups(Y, vocab_min):
    v_dict = defaultdict(str)
    c_dict = defaultdict(str)
    with open('../mimicdata/vocab_lookup_' + str(vocab_min) + '.csv', 'r') as vocabfile:
        vr = csv.reader(vocabfile)
        next(vr)
        for row in vr:
            v_dict[int(row[0])] = row[1]

    with open('../mimicdata/label_lookup_' + str(Y) + '.csv', 'r') as labelfile:
        lr = csv.reader(labelfile)
        next(lr)
        for row in lr:
            c_dict[int(row[0])] = row[1]
    return (v_dict, c_dict)    

if __name__ == "__main__":
    #just take in the label set size
    if len(sys.argv) < 4:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min] [model_path] [dataset (single or full)]"))
        sys.exit(0)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])

