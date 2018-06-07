#import sys
#sys.path.append("..")
#from ..fileReading import Data
from .entity_dic import entity_dict
import gensim
import json
import os
import numpy as np
#import  .word_vector.word_vec_wrapper
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
realismap ={'actual':0, 'generic':1, 'other':2, 'non-event':4}
nermap = {'sentence':0, 'commodity':1, 'time':2, 'crime':3, 'LOC':4, 'vehicle':5, 'PER':6, 'money':7, 'GPE':8, 'weapon':9, 'ORG':10, 'title':11, 'FAC':12}
arg_specificity = {'nonspecific': 1, 'specificGroup': 3, 'specific': 0, 'specificIndeterminate': 5, 'specificIndividual': 4, 'UNK': 2}
arg_ere= {'ere::Beneficiary': 46, 'adjudicator': 38, 'person': 8, 'ere::Giver': 43, 'ere::Agent': 24, 'destination': 15, 'audience': 17, 'place': 18, 'ere::Destination': 42, 'ere::Audience': 45, 'ere::Adjudicator': 48, 'giver': 36, 'ere::Person': 23, 'defendant': 11, 'ere::Origin': 41, 'ere::Org': 25, 'recipient': 20, 'ere::Defendant': 33, 'ere::Position': 39, 'beneficiary': 40, 'attacker': 6, 'instrument': 7, 'ere::Money': 44, 'ere::Target': 2, 'entity': 9, 'artifact': 28, 'ere::Place': 3, 'target': 14, 'money': 32, 'ere::Victim': 27, 'crime': 19, 'position': 26, 'ere::Crime': 35, 'origin': 22, 'thing': 21, 'victim': 0, 'ere::Entity': 12, 'ere::Thing': 31, 'time': 13, 'prosecutor': 10, 'ere::Prosecutor': 37, 'agent': 1, 'ere::Instrument': 5, 'ere::Attacker': 4, 'ere::Sentence': 34, 'ere::Artifact': 16, 'ere::Recipient': 30, 'ere::Time': 29, 'ere::Plaintiff': 47}
class Feature:
    def extract_feature(self, event,w2v):
        #---- event features-----#
        realis_1hot = [0]*len(realismap)
        realis_1hot[realismap[event['event']['modality']]]=1
        word2vec_lemma = w2v.vector(event['event']['lemma'])#w2v[event['event']['lemma']]
        #----- argument features-----#
        args = event['arguments']
        no_of_args = len(args)
        arg_ere_presence = [0]*len(arg_ere)
        arg_specificity_presence = [0]*len(arg_specificity)
        arg_ner_presence = [0]*len(nermap)
        #arg_entity_presence = [0]*len(entity_dict)
        for a in args:
            ere = a['ere']
            sp = a['entity-specificity']
            ner = a['entity-ner']
            #entity = a['entity']
            arg_ere_presence[arg_ere[ere]]+=1
            arg_specificity_presence[arg_specificity[sp]]+=1
            arg_ner_presence[nermap[ner]]+=1
            #arg_entity_presence[entity_dict[entity]]+=1
        #--create the feature---#
        feature = list()
        feature.extend(realis_1hot)
        feature.extend(word2vec_lemma)
        feature.extend([no_of_args])
        feature.extend(arg_ere_presence)
        feature.extend(arg_specificity_presence)
        feature.extend(arg_ner_presence)
        return np.array(feature)

if __name__ == '__main__':
    testfileName='/Users/abhipubali/Public/DropBox/AIDA_Paper/work/data/010aaf594ae6ef20eb28e3ee26038375.rich_ere.xml.inputs.json'
    with open(testfileName) as json_data:
        data= json.load(json_data)
    ev= data[0]
    print(ev)
    feat = Feature()
    #w2v = KeyedVectors.load_word2vec_format('/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt', binary=False)
    f= feat.extract_feature(ev, w2v)
    print(f.shape)
    print(f)
