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
    df = Data_Reader(cfg.JSON_DATA, cfg.TRAINING_LABEL_DATA,1)
    list_of_pair = df.list_of_pairs
    print('extracting features for training')
    train_X1, train_X2, train_S, train_Y = feature_extraction_caller(list_of_pair,1)
    model = My_Model(train_X1.shape[1], 50)
    model.train_model(train_X1,train_X2,train_S,train_Y,epch=cfg.epch)
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)
    saved_fname = model.save_model(cfg.MODEL_PATH)
    print('dimention of input is {} x {}'.format(train_X1.shape[1], 50))
    print('model saved as {}'.format(saved_fname))
