# in python3, cPickle has been replaced by _pickle 
# see https://blog.csdn.net/CaoMei_HuaCha/article/details/82899662
import _pickle as cPickle
import gzip 
import numpy as np 

def load_data():
    f = gzip.open('mnist.pkl.gz','rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data()



