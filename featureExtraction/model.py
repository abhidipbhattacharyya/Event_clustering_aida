import numpy as np
from keras.models import Model
from keras.layers.core import  Dropout
from keras.layers import Dense, Input, merge
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import load_model
import os
import json
#https://statcompute.wordpress.com/2017/01/08/an-example-of-merge-layer-in-keras/
#https://stackoverflow.com/questions/41603357/model-ensemble-with-shared-layers

class My_Model:
    def __init__(self,input_shape, dimention):
        self.ip1 = Input(shape=(input_shape,))
        self.ip2 = Input(shape=(input_shape,))
        #self.sh_dense1 = Dense(1500, activation='relu')
        self.sh_dense = Dense(dimention, activation='softmax')#(Dropout(0.5)(Dense(1500, activation='relu')))
        self.sh_dense1_op = self.sh_dense(self.ip1)
        self.sh_dense2_op = self.sh_dense(self.ip2)

        self.merged_layer = merge([self.sh_dense1_op, self.sh_dense2_op], mode = 'concat')
        #self.model.add(Dense(1, init = 'normal', activation = 'sigmoid'))
        self.prediction = Dense(1, activation='sigmoid')(Dense(250, activation='relu')(self.merged_layer))
        self.model = Model(input=[self.ip1, self.ip2], output=self.prediction)
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        self.model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
        #seed(2017)
        #self.model.fit([X1, X2], Y.values, batch_size = 2000, nb_epoch = 100, verbose = 1)
    def train_model(self,train_X1,train_X2,train_y,batch_size_=50,epch=15):
        self.model.fit([train_X1,train_X2], train_y, batch_size=batch_size_, nb_epoch=epch, verbose=1, shuffle=True)
        self.epch = epch

    def predict(self,test_x1,test_x2):
        return self.model.predict([test_x1,test_x2])

    def evaluate(self,X1,X2,y):
        return self.model.evaluate([X1,X2],y)

    def save_Model_separately(self, path):
        weights_file = os.path.join(path,'weight.h5')
        model_file = os.path.join(path, 'model_archi.json')
        # Save the weights
        self.model.save_weights(weights_file)
        with open(model_file, 'w') as f:
            f.write(self.model.to_json())

    def save_model(self, path):
        model_file = os.path.join(path,'model_'+str(self.epch)+'.h5')
        self.model.save(model_file)

    def load_model(self, path):
        self.model= load_model(path)



if __name__ == '__main__':
    train_x1 = np.random.rand(100,40)
    train_x2 = np.random.rand(100,40)
    print(train_x1.shape)
    print(train_x2.shape)
    train_y = np.random.randint(2, size=train_x1.shape[0])
    print(train_y.shape)
    model = My_Model(train_x1.shape[1],50)
    model.train_model(train_x1,train_x2,train_y,epch=2)
