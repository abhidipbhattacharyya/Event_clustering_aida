import json
import os
from collections import defaultdict

class Data:
    def __init__(self, dirnames):
        self.data= list()
        self.fname = list()
        self.dirnames = dirnames


    def readFiles(self):
        for dirname in self.dirnames:
            for jasonfile in os.listdir(dirname):
                print('processing {}'.format(jasonfile))
                self.fname.append(jasonfile)
                jasonfile = os.path.join(dirname, jasonfile)
                with open(jasonfile) as json_data:
                    self.data.append(json.load(json_data))
                #print(self.data[0][0]['event']['ere'])

    def similarity(self,d1,d2):
        search = '::'
        lens = len(search)
        ere1 = d1['event']['ere']
        ere2 = d2['event']['ere']
        search_inx1 = ere1.find(search)
        search_inx2 = ere2.find(search)

        if search_inx1>0:
            ere1 = ere1[search_inx1+lenS:]
        if search_inx2 > 0:
            ere2 = ere2[search_inx2+lenS:]

        pb1 = d1['event']['propbank']
        ix1 = pb1.find('.')
        if ix1 > 0:
            pb1 = pb1[:ix1-1]

        pb2 = d2['event']['propbank']
        ix2 = pb2.find('.')
        if ix2 > 0:
            pb2 = pb2[:ix2-1]

        sim = int(ere1==ere2)+ 0.5* int(pb1==pb2)
        cluster_name = ere1+'_'+pb1
        return sim, ere1, pb1

    def processdata_sim(self):
        self.cluster = [None]*len(self.data)#defaultdict()
        c_no=0
        index=-1
        for f,d in zip(self.fname , self.data):
            index =index+1
            no_of_data = len(d)
            self.cluster[index] = [None] * no_of_data
            for i in range(no_of_data):
                self.cluster[index][i] = [d[i],[d[i]]] #mean
                #self.cluster[index][i][1]= [d[i]] #cluster initialization
            change = 0
            for i in range(len(self.cluster[index] )-1):
                to_be_deleted = list()
                for j in range(i+1, len(self.cluster[index] )):
                    sim, ere, pb = self. similarity(self.cluster[index] [i][0], self.cluster[index] [j][0])
                    if(sim > 1):
                        #merge
                        self.cluster[index] [i][1].extend(self.cluster[index][j][1])
                        to_be_deleted.append(self.cluster[index][j])
                        #update mean
                        self.cluster[index][i][0]['event']['ere'] = ere
                        self.cluster[index][i][0]['event']['propbank']=pb
                        change = 1
                for tbd in to_be_deleted:
                    self.cluster[index].remove(tbd)

    def processdata(self):
        self.cluster = defaultdict()
        c_no=0
        for f,d in zip(self.fname , self.data):
            no_of_data = len(d)
            for i in range(no_of_data):
                if d[i]['event']['ere'] not in self.cluster:
                    #continue
                #else:
                    self.cluster[d[i]['event']['ere']]= c_no
                    c_no = c_no +1
                print('{}\t{}\t({})'.format(f, d[i]['event']['mentionid'],self.cluster[d[i]['event']['ere']]))

    def writeOP(self):
        for f,d in zip(self.fname , self.data):
            file= os.path.join('key',f.replace('inputs.json','key'))
            opfile = open(file,'w')
            no_of_data = len(d)
            for i in range(no_of_data):
                opfile.write('{}\t{}\t({})\n'.format(f.replace('.inputs.json',''), d[i]['event']['mentionid'],self.cluster[d[i]['event']['ere']]))
            opfile.close()

    def writeOP2(self):
        index = -1
        for f,d in zip(self.fname , self.data):
            index = index+1
            file= os.path.join('key',f.replace('inputs.json','key'))
            opfile = open(file,'w')
            opfile.write('#begin document ({}); part 000\n'.format(f.replace('.inputs.json','')))
            no_of_data = len(d)
            no_of_clus = len(self.cluster[index] )
            #print(no_of_clus)

            for i in range(no_of_clus):
                clus = self.cluster[index][i][1]
                #print(len(clus))
                for c in clus:
                    opfile.write('{}\t{}\t({})\n'.format(f.replace('.inputs.json',''), c['event']['mentionid'],i))
            opfile.write('#end document')
            opfile.close()

df = Data(['data'])
df.readFiles()
df.processdata_sim()
df.writeOP2()
