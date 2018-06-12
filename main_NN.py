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



nlp = spacy.load('en')
label_data_path = 'cluster/all.cluster'
data_folders= ['']
label_data_training = 'cluster/train1.cluster'
label_data_testing = 'cluster/testing1.cluster'
testfileName='/Users/abhipubali/Public/DropBox/AIDA_Paper/work/data/010aaf594ae6ef20eb28e3ee26038375.rich_ere.xml.inputs.json'
#w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt')
w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt',nlp)
def read_lable_data(file):
    list_line = list()
    list_pairs = list()
    with open(file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        #tokens = nlp(line)
        tokens = line.split()
        #0 file namae 2-ev1 4-ev2 6-cluster
        #print(tokens[0])
        list_pairs.append(Pair(str(tokens[1]),str(tokens[2]),str(tokens[3]),str(tokens[0])))
        #list_pairs.append(Pair(str(tokens[2]),str(tokens[4]),str(tokens[6]),str(tokens[0])))
    return list_pairs

def read_events_pairs(filepath, list_of_pair, augmentation =0):
    actual_event_pairs = list()
    one_count =0
    zero_count=0
    aug_one_count=0
    aug_zero_count =0
    for p in list_of_pair:
        fname = p.fname + '.inputs.json'
        fname = os.path.join(filepath, fname)
        with open(fname) as json_data:
            data= json.load(json_data)
        for d in data:
            if d['event']['mentionid'] == p.ev1:
                ev1 = d
            elif d['event']['mentionid'] == p.ev2:
                ev2 = d
        newp = Pair(ev1,ev2,p.same)
        actual_event_pairs.append(newp)
        #print(p.same)
        if p.same==1:
            one_count = one_count+1
        else:
            zero_count = zero_count+1

        #data augmentation
        if augmentation ==1:
        # data augmentation with associativity
            prob = random.uniform(0, 1)
            if prob > 0.6 and p.same==1:
                newp = Pair(ev2,ev1,p.same)
                aug_one_count = aug_one_count+1
                actual_event_pairs.append(newp)
            elif prob >0.9 and p.same==0:
                newp = Pair(ev2,ev1,p.same)
                aug_zero_count = aug_zero_count +1
                actual_event_pairs.append(newp)
        #data augmentation with reflex
            prob = random.uniform(0, 1)
            if prob > 0.9:
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    newp = Pair(ev1,ev1,1)
                    aug_one_count = aug_one_count+1
                else:
                    newp = Pair(ev2,ev2,1)
                    aug_zero_count = aug_zero_count +1
                actual_event_pairs.append(newp)
    print('oneC={}, zeroC={}, augOne={}, augZero={}'.format(one_count, zero_count, aug_one_count, aug_zero_count))
    return actual_event_pairs

def data_augmentation(list_pair):
    indices = [x for x in range(len(list_pair))]
    new_pair = list()
    random.shuffle(indices)
    aug_zero_count = 0
    aug_one_count = 0
    for i in indices:
        p = list_pair[i]
        prob = random.uniform(0, 1)
        if prob > 0.6 and p.same==1:
            newp = Pair(p.ev2,p.ev1,p.same)
            aug_one_count = aug_one_count+1
            new_pair.append(newp)
        elif prob >0.9 and p.same==0:
            newp = Pair(p.ev2,p.ev1,p.same)
            aug_zero_count = aug_zero_count +1
            new_pair.append(newp)
    #data augmentation with reflexive
        prob = random.uniform(0, 1)
        if prob > 0.9:
            prob = random.uniform(0, 1)
            if prob > 0.5:
                newp = Pair(p.ev1,p.ev1,1)
                aug_one_count = aug_one_count+1
                new_pair.append(newp)
            else:
                newp = Pair(p.ev2,p.ev2,1)
                aug_zero_count = aug_zero_count +1
                new_pair.append(newp)
    list_pair.extend(new_pair)
    print(' augOne={}, augZero={}'.format( aug_one_count, aug_zero_count))
    return list_pair

def train_test_split(actual_event_pairs):
    indices = [i for i in range(len(actual_event_pairs))]
    train_set = list()
    test_set = list()
    random.shuffle(indices)
    for i in indices:
        train_prob = random.uniform(0, 1)
        if train_prob >.8:
            test_set.append(actual_event_pairs[i])
        else:
            train_set.append(actual_event_pairs[i])
    return train_set, test_set

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

if __name__ == '__main__':
    '''
    print('processing .cluster files for training')
    list_of_pair_train = read_lable_data(label_data_training )
    #print(list_of_pair[0].fname)
    #actual_event_pairs = read_events_pairs('data', list_of_pair)

    print('processing input files for training')
    act_train_p = read_events_pairs('data/Inputs', list_of_pair_train)
    #train_set, vali_set = train_test_split(act_train_p)
    train_set = act_train_p
    train_set = data_augmentation(train_set)

    print('extracting features for training')
    train_X1, train_X2, train_Y = feature_extraction_caller(train_set,1)

    model = My_Model(train_X1.shape[1], 50)
    model.train_model(train_X1,train_X2,train_Y,epch=2)
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    model.save_model('trained_model')
    '''
    print('loading model')
    model1 =  My_Model(373, 50)
    model1.load_model('trained_model/model_2.h5')

    print('processing .cluster files for testing')
    list_of_pair_test = read_lable_data(label_data_testing )
    print('processing input files for testing')
    test_set = read_events_pairs('data/Inputs', list_of_pair_test)
    print('processing input files for testing')
    test_X1, test_X2, test_Y = feature_extraction_caller(test_set,1)


    print(model1.evaluate(test_X1, test_X2, test_Y ))


    #t1 = np.array(test_X1[0]).reshape(1,373)
    #t2 = np.array(test_X2[0]).reshape(1,373)
    #print(t1.shape)
    #p_y = model.predict(test_X1, test_X2)
    #print(p_y.shape)
    #print(test_Y.shape)
    #p= model.predict(t1, t2)
    #for i,p  in zip(test_Y, p_y):
        #print('{} vs {}'.format(p,i))
# augment 1 labeled data by - swapping ev1 and ev2
#augment i labeld data by putting ev1=ev2 from any label data
