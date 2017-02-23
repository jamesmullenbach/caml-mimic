"""
    Runs a ConvNet over the data to predict ICD-9 diagnosis codes
    options for single window or multi-window
    
    Framework: Keras
"""
from collections import defaultdict
import csv
import json
import os
import numpy as np
import sys
import time

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
EMBEDDING_SIZE = 100
DROPOUT_EMBED = 0.5
EMBED_INIT = 'glorot_uniform'

#Convolution constants
FILTER_SIZE = 3
MULTI_WINDOW = False
MIN_FILTER = 3
MAX_FILTER = 5
CONV_DIM_FACTOR = 10
ACTIVATION_CONV = 'tanh'
WINDOW_TYPE = 'valid'

#training constants
BATCH_SIZE = 64
NUM_EPOCHS = 1

#other
DROPOUT_DENSE = 0.5
OPTIMIZER = 'rmsprop'
LOSS = 'binary_crossentropy'
LEARNING_RATE = 0.001
PADDING = "var"

def main(Y, vocab_min, dataset):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """
    if MULTI_WINDOW:
        cnn = build_model_multiwindow(Y, MIN_FILTER, MAX_FILTER)
    else:
        cnn = build_model(Y)

    history = []
    names = ["acc", "prec", "rec", "f1", "auc"]
    names.extend(["%s_micro" % (name) for name in names])
    metrics_hist = {name: [] for name in names}
    metrics_hist_tr = {name: [] for name in names}
    for i in range(NUM_EPOCHS):
        hist = train(cnn, i, dataset, Y)
        history.append(hist)
        metrics, fpr, tpr = test("dev", cnn, dataset, Y, print_samples=True)
        print_metrics(metrics)
        for name in names:
            metrics_hist[name].append(metrics[name])


        print("sanity check on train")
        metrics_t, fpr_t, tpr_t = test("train", cnn, dataset, Y)
        print_metrics(metrics_t)
        for name in names:
            metrics_hist_tr[name].append(metrics_t[name])
    #save metric history, model, params
    model_dir = "models/" + '_'.join(['cnn', time.strftime('%b_%d_%H:%M', time.gmtime())])
    os.mkdir(model_dir)

    save_metrics(metrics_hist, metrics_hist_tr, model_dir)
    save_params(model_dir, Y, vocab_min, dataset)
    cnn.save(model_dir + "/model.h5")

def build_model(Y):
    """
        Builds the single window CNN model
        params:
            Y: size of the label space
        returns:
            cnn: the CNN model
    """
    cnn = Sequential()
    cnn.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE, dropout=DROPOUT_EMBED, init=EMBED_INIT))
    cnn.add(Convolution1D(Y*CONV_DIM_FACTOR, FILTER_SIZE, activation=ACTIVATION_CONV))
    from keras.layers.pooling import GlobalMaxPooling1D
    cnn.add(GlobalMaxPooling1D())
    cnn.add(Dense(Y))
    cnn.add(Dropout(DROPOUT_DENSE))
    cnn.add(Activation('sigmoid'))
    cnn.compile(optimizer=OPTIMIZER, loss=LOSS)
    print(cnn.summary())
    print
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
        convs[i].add(Convolution1D(Y*CONV_DIM_FACTOR, sz, activation=ACTIVATION_CONV))
        from keras.layers.pooling import GlobalMaxPooling1D
        convs[i].add(GlobalMaxPooling1D())

    merged = Merge(convs, mode='concat', concat_axis=1) 

    cnn_multi = Sequential()
    cnn_multi.add(merged)
    cnn_multi.add(Dense(Y))
    cnn_multi.add(Dropout(DROPOUT_DENSE))
    cnn_multi.add(Activation('sigmoid'))
    
    cnn_multi.compile(optimizer=OPTIMIZER, loss=LOSS)
    print(cnn_multi.summary())
    return cnn_multi

def train(cnn, epoch, dataset, Y, print_weights=False):
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
        if (X_batch.shape[1] < FILTER_SIZE):
            continue
        loss = cnn.train_on_batch(X_batch, Y_batch)
        if idx % 500 == 0:
            print('Train Epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {}'.format(
                epoch+1, idx, X_batch.shape[0], X_batch.shape[1], loss))
        if idx % 500 == 0 and print_weights:
            print("last layer weights: " + str(cnn.layers[-3].get_weights()[0][:5]))
        hist.append(loss)
    return hist

def test(fold, cnn, dataset, Y, max_iter = 1e9, print_samples=False):
    """
        Runs the model on dev data, computes a bunch of metrics
        params:
            fold: dev or train or test
            cnn: the model
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
    for idx,(X_batch, Y_batch) in enumerate(data_generator('../mimicdata/notes_%s_%s_%s_sorted.csv'\
                                                               % (str(Y), fold, dataset),\
                                                           BATCH_SIZE, Y)):
        if idx >= max_iter:
            break
        if X_batch.shape[1] < FILTER_SIZE:
            continue
        Y_hat = cnn.predict_on_batch(X_batch)

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

def print_metrics(metrics):
    print
    print("[MACRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"])
    print("[MICRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])
    print

def save_metrics(metrics_hist, metrics_hist_tr, model_dir):
    with open(model_dir + "/metrics.json", 'w') as metrics_file:
        data = metrics_hist.copy()
        data.update({"%s_tr" % (name):val for (name,val) in metrics_hist_tr.items()})
        json.dump(data, metrics_file, indent=1)

def save_params(model_dir, Y, vocab_min, dataset):
    with open(model_dir + "/params.json", 'w') as params_file:
        param_names = ["Y", "dataset", "Num epochs", "Vocab size", "Embedding size", "Embed dropout", "Conv activation", "Conv window size",
                "Conv window type", "Conv output size", "Dense dropout", "Optimizer", "Loss", "Embed init", "Learning rate",
                "Padding", "Vocab min occurrences"]
        filter_size = FILTER_SIZE if not MULTI_WINDOW else ';'.join([str(i) for i in range(MIN_FILTER, MAX_FILTER + 1)])
        param_vals = [Y, dataset, NUM_EPOCHS, VOCAB_SIZE, EMBEDDING_SIZE, DROPOUT_EMBED, ACTIVATION_CONV, filter_size, WINDOW_TYPE, 
                CONV_DIM_FACTOR*Y, DROPOUT_DENSE, OPTIMIZER, LOSS, EMBED_INIT, LEARNING_RATE, PADDING, vocab_min]
        data = {name: str(val) for (name,val) in zip(param_names, param_vals)}
        json.dump(data, params_file, indent=1)


if __name__ == "__main__":
    #just take in the label set size
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min] [dataset (single or full)]"))
        sys.exit(0)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])

