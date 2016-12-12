
import numpy 
import scipy.io
import h5py
import pandas as pd
import cPickle as pickle
from keras.utils import np_utils
import pdb

#from math import floor

class dataFactory_pyprep(object):
    
    def __init__(self, years_dict,data_path,data_tags,exchange_ids=None):
               
        full_data = pd.concat(pd.read_hdf(cur_path,'table') for cur_path in data_path)
             
        #full_data.set_index('year',drop=True,inplace=True)
        train_data = full_data[full_data.year<=years_dict['train_top']]
        test_data = full_data[full_data.year>=years_dict['test_bottom']]
        test_data = test_data[test_data.year<=years_dict['test_top']]
        
#        if train_data.shape[0] != 0:
#            train_labels = [numpy.where(train_data['class_label'] == 1)[0],
#                            numpy.where(train_data['class_label'] == 2)[0]]          
#        
#            # level datas
#            num_samples = [len(train_labels[0]),len(train_labels[1])]
#            sorted_numsamples = numpy.argsort(num_samples)
#            new_labels = numpy.random.choice(train_labels[sorted_numsamples[1]],size=num_samples[sorted_numsamples[0]],replace=False)            
#            res_train_labels = numpy.concatenate((numpy.asarray(train_labels[sorted_numsamples[0]]),new_labels),axis=0)
#            
#            train_data = train_data.iloc[res_train_labels]
#            train_data.reset_index(inplace=True,drop=False)
#        
#        test_data.reset_index(inplace=True,drop=False)
        
        self.train_data = train_data
        self.test_data = test_data
        self.data_tags = data_tags
        

    def get_train_data(self):
        #y = np_utils.to_categorical(self.train_data['class_label'].values -1)
        #X = self.train_data[self.data_tags].values.astype(numpy.float32)
        return self.train_data[self.data_tags].values.astype(numpy.float32)

    def get_test_data(self):          
        return self.test_data

				
