import json
import os
from collections import defaultdict

class Data:
    def __init__(self, dirnames):
        self.data= list()
        self.event_ere={}
        self.ner_dic = defaultdict()
        self.ner_type_dic = defaultdict()
        self.en_ere={}
        self.en_sp={}
        self.dirnames = dirnames


    def readFiles(self):
        for dirname in self.dirnames:
            for jasonfile in os.listdir(dirname):
                print('processing {}'.format(jasonfile))
                #self.fname.append(jasonfile)
                jasonfile = os.path.join(dirname, jasonfile)
                with open(jasonfile) as json_data:
                    self.data.append(json.load(json_data))

    def process_data(self):
        count=0
        count_sp=0
        count_ere=0
        count_ev_ere=0
        for d in self.data:
            for e in d:
                event = e['event']
                args = e['arguments']
                event_ere = event['ere']
                if event_ere not in self.event_ere:
                    self.event_ere[event_ere] = count_ev_ere
                    count_ev_ere = count_ev_ere+1
                for a in args:
                    ner_type = a['entity-ner']
                    ner = a['entity']
                    ere= a['ere']
                    sp = a['entity-specificity']
                    self.ner_type_dic[ner_type]=1
                    if ner not in self.ner_dic:
                        self.ner_dic[ner] = count
                        count = count +1
                    if ere not in self.en_ere:
                        self.en_ere[ere] = count_ere
                        count_ere = count_ere +1
                    if sp not in self.en_sp:
                        self.en_sp[sp] = count_sp
                        count_sp = count_sp +1


df = Data(['/home/verbs/shared/aida/resources/EventCorefData/Inputs'])
df.readFiles()
df.process_data()
#print(df.ner_type_dic.keys())
#print(df.ner_dic)
#print(df.en_sp)
#print(df.en_ere)
print(df.event_ere)
