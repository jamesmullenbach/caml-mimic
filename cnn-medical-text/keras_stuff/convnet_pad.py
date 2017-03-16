"""
    Runs a ConvNet over the data to predict ICD-9 diagnosis codes
    for the fixed input length case. Different file b/c data loading/evaluation are diff
    
    Framework: Keras
"""
import os
import numpy as np
import sys
import time

from constants import *
import datasets, evaluation, persistence

import keras_stuff.models as models

from keras.preprocessing import sequence

PADDING = "post"

def main(Y, vocab_min, dataset, filter_size, conv_dim_factor, arch="single"):
    """
        main function which sequentially loads the data, builds the model, trains, evaluates, writes output, etc.
    """

    if arch == "multi":
        cnn = models.multi_window(Y, filter_size, conv_dim_factor, PADDING)
    else:
        cnn = models.vanilla(Y, filter_size, conv_dim_factor, PADDING)

    (X_tr, Y_tr), (X_dv, Y_dv) = datasets.load_all_data(Y, dataset, notebook=False)

    print("padding sequences")
    X_tr = sequence.pad_sequences(X_tr, maxlen=MAX_LENGTH, padding=PADDING, truncating=PADDING)
    X_dv = sequence.pad_sequences(X_dv, maxlen=MAX_LENGTH, padding=PADDING, truncating=PADDING)

    print("getting lookups")
    v_dict, c_dict = datasets.load_lookups(Y, vocab_min)

    #shuffle before just in case
    np.random.seed(1337)
    np.random.shuffle(X_tr)
    np.random.seed(1337)
    np.random.shuffle(Y_tr)
    print("##### OTHER PARAMS ######")
    print("# Sequence length: %d, conv window: %d, dropout: %f" % (MAX_LENGTH, filter_size, DROPOUT_DENSE))
    print("training model")
    hist = train(cnn, X_tr, Y_tr, X_dv, Y_dv)
    print("evaluating on dev")
    preds,metrics,fpr, tpr = evaluate(cnn, X_dv, Y_dv)
    print("[MACRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc"], metrics["prec"], metrics["rec"], metrics["f1"], metrics["auc"])
    print("[MICRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc_micro"])
    print

    print("sanity check on train")
    preds_t,metrics_t, fpr_t, tpr_t = evaluate(cnn, X_tr, Y_tr)
    print("[MACRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics_t["acc"], metrics_t["prec"], metrics_t["rec"], metrics_t["f1"], metrics_t["auc"])
    print("[MICRO] accuracy, precision, recall, f-measure, AUC")
    print(metrics_t["acc_micro"], metrics_t["prec_micro"], metrics_t["rec_micro"], metrics_t["f1_micro"], metrics_t["auc_micro"])

    print("ROC AUC measures")
    preds = np.array(preds)
    Y_dv = np.array(Y_dv)
    roc_auc = {"macro": metrics["auc"], "micro": metrics["auc_micro"]}
    roc_auc.update({i: metrics["auc_%d" % i] for i in range(Y)})

    persistence.write_auc(fpr, tpr, roc_auc, Y)
    #plot_auc(fpr, tpr, roc_auc)

    print("writing predictions and true labels")
    preds = [[i for i in range(len(p)) if p[i] == 1] for p in preds]
    preds_t = [[i for i in range(len(p)) if p[i] == 1] for p in preds_t]
    golds = [[i for i in range(len(g)) if g[i] == 1] for g in Y_dv]
    golds_t = [[i for i in range(len(g)) if g[i] == 1] for g in Y_tr]
    persistence.write_preds(preds, 'dev.preds')
    persistence.write_preds(preds_t, 'train.preds')
    persistence.write_preds(golds, 'dev.golds')
    persistence.write_preds(golds_t, 'train.golds')

    #save metric history, model, params
    model_dir = "../saved_models/" + '_'.join(['cnn_pad', time.strftime('%b_%d_%H:%M', time.gmtime())])
    os.mkdir(model_dir)

    persistence.save_metrics(metrics, metrics_t, model_dir)
    persistence.save_params(model_dir, Y, vocab_min, dataset, filter_size, conv_dim_factor, arch, PADDING)
    cnn.save(model_dir + "/model.h5")

def train(cnn, X_tr, Y_tr, X_dv, Y_dv):
    #fit the model, validating with dev data
    hist = cnn.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_data=(X_dv, Y_dv))
    return hist

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

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("usage: python " + str(os.path.basename(__file__) + " [|Y|] [vocab_min] [dataset (single or full)] [filter_size] [conv_dim_factor] [architecture (single or multi)]"))
        sys.exit(0)
    arch = "single" if len(sys.argv) <= 6 else sys.argv[6]
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), arch)

