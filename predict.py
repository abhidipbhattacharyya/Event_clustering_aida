from featureExtraction.feature import Feature
from featureExtraction.model import My_Model
from word_vector import word_vec_wrapper
from training_pair_preparation.classes import Pair
import json
import spacy
import os
import random
import numpy as np
import tensorflow as tf
from input_reading.input_reader import Data
import configuration.config as cfg
from postprocessing.key_generation import generate_key, write_key
nlp = spacy.load('en')
w2v = word_vec_wrapper(cfg.W2V_PATH ,nlp)

def feature_extraction_caller(event_pair_list, npa):
    X1 = list()
    X2 = list()
    Y = list()
    feat = Feature()

    for p in event_pair_list:
        Y.append(p.same)
        f1= feat.extract_feature(p.ev1, w2v)
        f2= feat.extract_feature(p.ev2, w2v)
        X1.append(f1)
        X2.append(f2)

    if npa ==1:
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y = np.array(Y)
    return X1, X2, Y


df = Data()
fname_pair = df.read_jsons(cfg.TESTING_json_DATA)
print('loading model....')
model1 =  My_Model(373, 50)
model1.load_model(cfg.MODEL_PATH)
fname_cluster_pair = list()

for fname in fname_pair:
    print('predicting {}'.format(fname))
    list_of_pairs = fname_pair[fname]
    test_X1, test_X2, test_Y = feature_extraction_caller(list_of_pairs,1)
    predicted_y = model1.predict(test_X1, test_X2)
    #print(predicted_y)
    #print('len of predicted = {}\n len of actual={} '.format(len(predicted_y),len(test_Y)))
    for i in range(len(predicted_y)):
        list_of_pairs[i].same = predicted_y[i]
        #print('{} vs {} = {}'.format(list_of_pairs[i].ev1['event']['mentionid'],list_of_pairs[i].ev2['event']['mentionid'],predicted_y[i]))
    fname, cluster = generate_key([fname,list_of_pairs])
    fname_cluster_pair.append([fname,cluster])

for ff in fname_cluster_pair:
    write_key(ff,'key')
