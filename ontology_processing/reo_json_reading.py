import json
import os
from collections import defaultdict
from word_vector import word_vec_wrapper
jasonfile = 'reo_relation.json'
class Reo:
    def __init__(self,jasonfile):
        #self.data = list()
        with open(jasonfile) as json_data:
            self.data = json.load(json_data)

    def findRelation_ordered(self, reo_key, reo_val):
        if reo_key.find("reo::")<0:
            reo_key = "reo::"+reo_key
        if reo_val.find("reo::")<0:
            reo_val = "reo::"+reo_val

        if reo_key in self.data:
            list_of_rel = self.data[reo_key]
            for rel, values in list_of_rel.items():
                if reo_val in values:
                    return rel
        return None

    def findRelation(self, r1, r2):
        rel = self.findRelation_ordered(r1,r2)
        if rel == None:
            rel = self.findRelation_ordered(r2,r1)
            if rel != None:
                dir = -1
            else:
                dir = None
        else:
            dir=1
        return rel, dir

    def word_vec_checking(self,w2v):

reo = Reo('reo_relations.json')
print(reo.findRelation("reo::Appeal","Verdict"))
print(reo.findRelation("Verdict","Appeal"))
print(reo.findRelation("Verdict","Alive"))
