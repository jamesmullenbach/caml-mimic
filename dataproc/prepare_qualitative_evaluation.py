"""
    Read four files with format (HADM_ID, CODE, INDEX)
        These specify four choices for the most important 4-gram in a discharge summary from the test set for a given code
    Write these out in markdown format to be presented to physician for evaluation
"""
import csv
from collections import Counter

import numpy as np
from tqdm import tqdm

from constants import *
import datasets

CONTEXT_SIZE = 10
NUM_QUESTIONS = 100
FILTER_SIZE = 4
MAX_CODE_OCCURRENCES = 5
INSTRUCTIONS = """
We have a dataset of discharge summaries, along with ICD-9 codes associated with those summaries. You will be presented with a series of codes and their descriptions, along with four supporting texts, each drawn from the same discharge summary which has that code associated. Please make a check mark by all supporting texts that you believe adequately explain the presence of the given code. If one supporting text clearly provides the best explanation, make a double check mark by that text. 
"""

ATTN_FILENAME = "foo"
CONV_FILENAME = "bar"
LR_FILENAME = "baz"
SIM_FILENAME = "hax"

def main():
    desc_dict = datasets.load_code_descriptions()

    print("loading attn windows")
    attn_windows = {}
    attn_window_szs = {}
    with open(ATTN_FILENAME, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            attn_windows[(int(row[0]), row[1])] = int(row[2])
            attn_window_szs[(int(row[0]), row[1])] = int(row[3])

    print("loading conv windows")
    conv_windows = {}
    with open(CONV_FILENAME, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            conv_windows[(int(row[0]), row[1])] = int(row[2])

    print("loading lr windows")
    lr_windows = {}
    with open(LR_FILENAME, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            lr_windows[(int(row[1]), row[2])] = int(row[3])

    print("loading sim windows")
    sim_windows = {}
    sim_vals = {}
    with open(SIM_FILENAME, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            sim_windows[(int(row[1]), row[2])] = int(row[3])
            sim_vals[(int(row[1]), row[2])] = float(row[-1])

    attn_keys = set(attn_windows.keys())
    conv_keys = set(conv_windows.keys())
    lr_keys = set(lr_windows.keys())
    sim_keys = set(sim_windows.keys())
    valid_texts = []
    print("building evaluation document")
    with open('%s/qualitative_eval_full.md' % (MIMIC_3_DIR), 'w') as of:
        with open('%s/qualitative_eval_full_key.md' % (MIMIC_3_DIR), 'w') as kf:
            code_counts = Counter()
            of.write('### Instructions\n')
            of.write(INSTRUCTIONS + '\n\n')
            with open('%s/test_full.csv' % MIMIC_3_DIR, 'r') as f:
                r = csv.reader(f)
                #header
                next(r)
                num_pairs = 0
                for idx,row in tqdm(enumerate(r)):
                    codes = str(row[3]).split(';')
                    toks = row[2].split()
                    hadm_id = int(row[1])
                    for code in codes:
                        num_pairs += 1
                        key = (hadm_id, code)
                        if key in conv_keys and key in lr_keys and key in sim_keys and key in attn_keys and code_counts[code] < MAX_CODE_OCCURRENCES:
                            if sim_vals[key] == 0:
                                continue
                            code_counts[code] += 1
                            valid_texts.append((key, toks))

            valid_texts = np.random.permutation(valid_texts)
            opts = 'ABCD'
            for i, (key, toks) in enumerate(valid_texts[:NUM_QUESTIONS]):
                hadm_id, code = key
                of.write("### Question %d\n" % (i + 1))
                kf.write("### Question %d\n" % (i + 1))
                of.write("Code: %s\n" % code)
                kf.write("Code: %s\n" % code)
                of.write("Full descriptions: %s\n\n" % desc_dict[code])
                kf.write("Full descriptions: %s\n\n" % desc_dict[code])
                for i,(method,window) in enumerate(np.random.permutation([('attn', attn_windows[key]), ('conv', conv_windows[key]), ('lr', lr_windows[key]), ('sim', sim_windows[key])])):
                    window = int(window)
                    if method == 'attn':
                        filter_size = attn_window_szs[key]
                    else:
                        filter_size = FILTER_SIZE
                    pre = toks[window-(CONTEXT_SIZE/2):window]
                    mid = toks[window:window+filter_size]
                    post = toks[window+filter_size:window+filter_size+(CONTEXT_SIZE/2)]
                    md_out = ' '.join(pre) + ' **' + ' '.join(mid) + '** ' + ' '.join(post)
                    of.write('%s) %s\n\n' % (opts[i], md_out))
                    kf.write('%s (%s) %s\n\n' % (opts[i], method, md_out))
            print("percentage of valid document-code pairs: %f" % (len(valid_texts) / float(num_pairs)))

if __name__ == "__main__":
    main()
