import csv
import itertools
import math
import operator
import pickle
import string
import sys

import nltk
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import datasets
from constants import *

'''
Learning idf weights for vocab over all descriptions and text, then using these weights to transform each ngram and description. For each record, output the ngram with the most string overlap, e.g. most informative, in predicting a given true code that the record has.'''

def main(inputfile, outputfile, n):
    n = int(n)
    
    csv.field_size_limit(sys.maxsize)
    
    calculate_string_overlap_ngrams(inputfile, outputfile, n)

def calculate_string_overlap_ngrams(inputfile, file_to_write, n):

    descs_dict = datasets.load_code_descriptions(mimic=3)
    print "Loading code descriptions"

    print "Loading corpus of text for which we have code descriptions"
    #get text for computing idf
    k = 0
    j = 0
    with open("%s/%s" % (DATA_DIR, inputfile), 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)

        corpus = []

        for i,row in tqdm(enumerate(reader)):

            text = row[2]
            labels = row[3].split(';')
            c = False

            for label in labels:
                #check that we have a description for the label:
                if label in descs_dict:
                    k = k + 1
                    c = True
                    #add description to corpus
                    name = clean(descs_dict[label])
                    corpus.append(name)
                else:
                    j = j + 1

            if c==True:
                #append text- only want to add to corpus once
                text = stem(text)
                corpus.append(text)

    print "No descriptions for how many code/text pairs:", j
    print "Descriptions for how many code/text pairs:", k

    #calculate idfs based off of corpus of text
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf_scores = dict(zip(vectorizer.get_feature_names(), idf))
    print "Idf weights based on generated corpus"
    print dict(itertools.islice(idf_scores.iteritems(), 10))
    
    #write out to csv
    f = open("%s/%s" % (DATA_DIR, file_to_write), 'wb')
    writer = csv.writer(f, delimiter = ',')
    #write header
    writer.writerow(['SUBJECT_ID', 'HADM_ID', 'LABEL', 'INDEX', 'NGRAM', 'SCORE'])

    z = 0
    print "Calculating tfs for each ngram and computing cosine similarity with description"
    print "Updated to just consider idf weights for each unique word in ngram, no longer doing term frequency for efficiency purposes"
    with open("%s/%s" % (DATA_DIR, inputfile), 'r') as notesfile:
        reader = csv.reader(notesfile)
        next(reader)

        cnt = 0
        for i,row in tqdm(enumerate(reader)):

            text = row[2]
            text = stem(text)
            labels = row[3].split(';') 
            hadm_id = row[1]
            subject_id = row[0]

            #for each text, label pair, calculate heighest weighted n-gram in text
            for label in labels:
                cnt = cnt + 1
                if cnt % 100 == 0:
                    print cnt
                if label in descs_dict:

                    desc = clean(descs_dict[label])
                    myList = []

                    #subject id
                    myList.append(subject_id)
                    #hadm id
                    myList.append(hadm_id)

                    #get each set of n grams in text
                    fourgrams = ngrams(text.split(), n)
                    fourgrams_scores = []
                    for grams in fourgrams:
                        
                        #get size
                        vocab = set(desc.split() + list(grams))
                        word_lookup = {k: v for v, k in enumerate(vocab)}
                        
                        #build description representation
                        tf_desc = np.zeros(len(vocab))
                        tf_gram = np.zeros(len(vocab))
                        
                        #add idf values to array
                        for word in desc.split():
                            if word in vectorizer.vocabulary_:
                                tf_desc[word_lookup[word]] = idf_scores[word]
                                
                        for word in grams:
                            if word in vectorizer.vocabulary_:
                                tf_gram[word_lookup[word]] = idf_scores[word]

                        #calculate cosine similarity with description
                        score = cosine_sim(tf_gram, tf_desc)
                        fourgrams_scores.append(score)

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

def clean(record):
    result = ''.join([i for i in record if not i.isdigit()])
    result = result.lower().translate(None, string.punctuation)
    return stem(result)

def stem(record):
    ps = PorterStemmer()
    return str(' '.join([ps.stem(i) for i in record.split()]))

def cosine_sim(u, v):
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v))) 

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python " + str(os.path.basename(__file__) + " [inputfile] [outputfile] [n]"))
        sys.exit(0)
 
    main(sys.argv[1], sys.argv[2], sys.argv[3])
