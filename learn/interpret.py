"""
    Code to extract some examples of where the attention was focusing for input documents
"""
import operator
import random
import sys

import numpy as np

import learn.models as models

def save_samples(data, output, target_data, s, filter_size, tp_file, fp_file, dicts=None):
    """
        save important spans of text from attention
        INPUTS:
            data: input data (text) to the model
            output: model prediction
            target_data: ground truth labels
            s: attention vector from attn model
            filter_size: size of the convolution filter, and length of phrase to extract from source text
            tp_file: opened file to write true positive results
            fp_file: opened file to write false positive results
            dicts: hold info for reporting results in human-readable form
    """
    tgt_codes = np.where(target_data[0] == 1)[0]
    true_str = "Y_true: " + str(tgt_codes)
    output_rd = np.round(output)
    pred_codes = np.where(output_rd[0] == 1)[0]
    pred_str = "Y_pred: " + str(pred_codes)
    if dicts is not None:
        if s is not None and len(pred_codes) > 0:
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, tp_file, fps=False)
            important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, fp_file, fps=True)

def important_spans(data, output, tgt_codes, pred_codes, s, dicts, filter_size, true_str, pred_str, spans_file, fps=False):
    """
        looks only at the first instance in the batch
    """
    ind2w, ind2c, desc_dict = dicts['ind2w'], dicts['ind2c'], dicts['desc']
    for p_code in pred_codes:
        #aww yiss, xor... if false-pos mode, save if it's a wrong prediction, otherwise true-pos mode, so save if it's a true prediction
        if output[0][p_code] > .5 and (fps ^ (p_code in tgt_codes)):
            confidence = output[0][p_code]

            #some info on the prediction
            code = ind2c[p_code]
            conf_str = "confidence of prediction: %f" % confidence
            typ = "false positive" if fps else "true positive"
            prelude = "top three important windows for %s code %s (%s: %s)" % (typ, str(p_code), code, desc_dict[code])
            if spans_file is not None:
                spans_file.write(conf_str + "\n")
                spans_file.write(true_str + "\n")
                spans_file.write(pred_str + "\n")
                spans_file.write(prelude + "\n")

            #find most important windows
            attn = s[0][p_code].data.cpu().numpy()
            #merge overlapping intervals
            imps = attn.argsort()[-10:][::-1]
            windows = make_windows(imps, filter_size, attn)
            kgram_strs = []
            i = 0
            while len(kgram_strs) < 3 and i < len(windows):
                (start,end), score = windows[i]
                words = [ind2w[w] if w in ind2w.keys() else 'UNK' for w in data[0][start:end].data.cpu().numpy()]
                kgram_str = " ".join(words) + ", score: " + str(score)
                #make sure the span is unique
                if kgram_str not in kgram_strs:
                    kgram_strs.append(kgram_str)
                i += 1
            for kgram_str in kgram_strs:
                if spans_file is not None:
                    spans_file.write(kgram_str + "\n")
            spans_file.write('\n')

def make_windows(starts, filter_size, attn):
    starts = sorted(starts)
    windows = []
    overlaps_w_next = [starts[i+1] < starts[i] + filter_size for i in range(len(starts)-1)]
    overlaps_w_next.append(False)
    i = 0
    get_new_start = True
    while i < len(starts):
        imp = starts[i]
        if get_new_start:
            start = imp
        overlaps = overlaps_w_next[i]
        if not overlaps:
            windows.append((start, imp+filter_size))
        get_new_start = not overlaps
        i += 1
    #return windows sorted by decreasing importance
    window_scores = {(start,end): attn[start] for (start,end) in windows}
    window_scores = sorted(window_scores.items(), key=operator.itemgetter(1), reverse=True)
    return window_scores
