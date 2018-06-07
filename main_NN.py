from featureExtraction.feature import Feature
from featureExtraction.model import My_Model
from word_vector import word_vec_wrapper
from training_pair_preparation.classes import Pair
import json
import spacy
import os
import random
import numpy as np

nlp = spacy.load('en')
label_data_path = 'cluster/all.cluster'
data_folders= ['']
testfileName='/Users/abhipubali/Public/DropBox/AIDA_Paper/work/data/010aaf594ae6ef20eb28e3ee26038375.rich_ere.xml.inputs.json'
#w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt')

def read_leable_data(file):
    list_line = list()
    list_pairs = list()
    with open(file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        tokens = nlp(line)
        #0 file namae 2-ev1 4-ev2 6-cluster
        #print(tokens[0])
        list_pairs.append(Pair(str(tokens[2]),str(tokens[4]),str(tokens[6]),str(tokens[0])))
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

def feature_extraction_caller(event_pair_list):
    X1 = list()
    X2 = list()
    Y = list()
    feat = Feature()
    w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt',nlp)
    for p in event_pair_list:
        Y.append(p.same)
        f1= feat.extract_feature(p.ev1, w2v)
        f2= feat.extract_feature(p.ev2, w2v)
        X1.append(f1)
        X2.append(f2)
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)
    return X1, X2, Y

if __name__ == '__main__':
    list_of_pair = read_leable_data(label_data_path)
    #print(list_of_pair[0].fname)
    actual_event_pairs = read_events_pairs('data', list_of_pair)
    train_set, test_set = train_test_split(actual_event_pairs)
    print(len(actual_event_pairs))
    print(len(train_set))
    print(len(test_set))

    train_set = data_augmentation(train_set)


    train_X1, train_X2, train_Y = feature_extraction_caller(train_set)

    model = My_Model(train_X1.shape[1], 50)
    model.train_model(train_X1,train_X2,train_Y,2)

    test_X1, test_X2, test_Y = feature_extraction_caller(test_set)
    print(model.model.evaluate([test_X1, test_X2], test_Y))
# augment 1 labeled data by - swapping ev1 and ev2
#augment i labeld data by putting ev1=ev2 from any label data
'''
with open(testfileName) as json_data:
    data= json.load(json_data)
ev= data[0]
print(ev)
feat = Feature()
w2v = word_vec_wrapper('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt')
f= feat.extract_feature(ev, w2v)
print(f.shape)
print(f)
'''
