from featureExtraction.feature import Feature
from featureExtraction.model import My_Model
from word_vector import word_vec_wrapper
from training_pair_preparation.classes import Pair
from input_reading.reader_cross_doc import Data_Reader
import json
import spacy
import os
import random
import numpy as np
import tensorflow as tf
import configuration.cross_doc_cfg as cfg
from keras.utils import to_categorical

nlp = spacy.load('en')
w2v = word_vec_wrapper(cfg.W2V_PATH ,nlp)

def feature_extraction_caller(event_pair_list, npa):
    X1 = list()
    X2 = list()
    S = list()
    Y = list()
    feat = Feature()

    for p in event_pair_list:
        Y.append(p.same)
        f1= feat.extract_feature(p.ev1, w2v)
        f2= feat.extract_feature(p.ev2, w2v)
        l1 = p.ev1['event']['lemma']
        l2 = p.ev2['event']['lemma']
        sim =w2v.similarity2(l1,l2)
        X1.append(f1)
        X2.append(f2)
        S.append(sim)
    if npa ==1:
        X1 = np.array(X1)
        X2 = np.array(X2)
        S = np.array(S)
        Y = np.array(Y)
    return X1, X2, S,Y,

#fname_pair = read_yyy(cfg.JSON_DATA)
if __name__ == '__main__':
    print('loading model1....')
    model1 =  My_Model(411, 50)#411 with ere event #373
    model1.load_model(cfg.MODEL)
    print('loading model2....')
    model2 = My_Model(411, 50)
    model2.load_model(cfg.IND_MODEL)


    df = Data_Reader(cfg.JSON_DATA, cfg.TESTING_LABEL_DATA,0)
    list_of_pair = df.list_of_pairs
    print('extracting features for training')
    test_X1, test_X2, test_S, test_Y = feature_extraction_caller(list_of_pair,1)

    #predicted_y = model1.predict(test_X1, test_X2, test_S)


    predicted_y2 = model2.predict(test_X1, test_X2, test_S)
    predicted_y = predicted_y2# (predicted_y+predicted_y2)/2

    confusion_mat = matrix = [[0]*2 for i in range(2)]
    ## cols are original labels, rows are predicted_y
    for p,o in zip(predicted_y, test_Y ):
        p = int(p+0.5)# convert to 0 or 1 if greater than 0.5
        o = int(o)
        confusion_mat[p][o]+=1

    cm = np.array(confusion_mat)

    true_total_count = np.sum(cm,0)
    pred_total_count = np.sum(cm,1)

    correct_pred = np.diag(cm)

    recall = correct_pred/(true_total_count)
    precision = correct_pred /(pred_total_count)
    print('======== confusion matrix============')
    print(' \t{}\t{}'.format(0,1))
    print('----------------------')
    for i in range(2):
        print('{}|\t{}\t{}'.format(i,confusion_mat[i][0],confusion_mat[i][1],))

    print('recall= {}'.format(recall))
    print('precision = {}'.format(precision))
