import os
from collections import defaultdict
import spacy
from classes import Line, File, Pair
import random

nlp = spacy.load('en')
class read_file:
    def __init__(self, dirnames):
        self.dirnames = dirnames
        self.data = list()
        self.process_data= list()
        self.process_train_data = list()
        self.process_test_data= list()

    def readFiles(self):
        for dirname in self.dirnames:
            for file in os.listdir(dirname):
                print('processing {}'.format(file))
                #self.fname.append(file)
                file = os.path.join(dirname, file)
                list_line = list()
                with open(file) as f:
                    lines = f.readlines()
                for i in range(1,len(lines)-1):
                    line = lines[i]
                    tokens = nlp(line)
                    fname_ = str(tokens[0])
                    event_men = str(tokens[2])
                    clus = str(tokens[5]).replace('(','').replace(')','')
                    l = Line(fname_,event_men,clus)
                    list_line.append(l)
                ff = File(list_line[0].filename,list_line)
                #print(ff.data)
                self.data.append(ff)

    def processdata(self):
        for ff in self.data:
            for key in ff.data:
                train_test = random.uniform(0,1)
                value = ff.data[key]
                fname = key
                list_pairs = list()
                for i in range(len(value)-1):
                    for j in range(i+1, len(value)):
                        p = Pair(value[i].event_mention, value[j].event_mention, value[i].same_cluster(value[j]))
                        list_pairs.append(p)
                d = [fname,list_pairs ]
                self.process_data.append(d)
                if train_test > 0.8:
                    self.process_test_data.append(d)
                    print('{} goes to testing'.format(key))
                else:
                    self.process_train_data.append(d)
                    print('{} goes to training'.format(key))


    def writeOP(self):
        for ff in self.process_data:
            fname = ff[0]+'.cluster'
            file= os.path.join('../cluster',fname)
            opfile = open(file,'w')
            for pair in ff[1]:
                opfile.write('{}\t{}\t{}\t{}\n'.format(ff[0],pair.ev1,pair.ev2,pair.same))

    def writeOP_all(self):
        file= os.path.join('../cluster','all.cluster')
        opfile = open(file,'w')
        for ff in self.process_data:
            for pair in ff[1]:
                opfile.write('{}\t{}\t{}\t{}\n'.format(ff[0],pair.ev1,pair.ev2,pair.same))

    def writeOP_train_test(self):
        file_train= os.path.join('../cluster','train.cluster')
        opfile = open(file_train,'w')
        for ff in self.process_train_data:
            for pair in ff[1]:
                opfile.write('{}\t{}\t{}\t{}\n'.format(ff[0],pair.ev1,pair.ev2,pair.same))

        file_test= os.path.join('../cluster','testing.cluster')
        opfile = open(file_test,'w')
        for ff in self.process_test_data:
            for pair in ff[1]:
                opfile.write('{}\t{}\t{}\t{}\n'.format(ff[0],pair.ev1,pair.ev2,pair.same))






df = read_file(['../key'])
df.readFiles()
df.processdata()
df.writeOP_train_test()
