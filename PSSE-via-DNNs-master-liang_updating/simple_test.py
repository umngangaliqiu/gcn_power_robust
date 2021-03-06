from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
from matplotlib import pyplot as plt
#np.random.seed(SEED)
import keras
from keras import backend as K
import tensorflow as tf
import os, shutil, scipy.io
from sklearn.metrics import mean_squared_error

####################
'''
 * @author [Liang Zhang]
 * @email [zhan3523@umn.edu]
Different NN models for PSSE provided in this file
'''
import tensorflow as tf

from keras import optimizers
from keras import regularizers

from keras.models import Model
from keras.layers import Dense, Activation, add, Dropout, Lambda
from keras.layers import Input, average
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from model import *
###################

# configure args
SEED = 1234
tf.set_random_seed(SEED)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_learning_phase(1)

# data loading part
caseNo = 118
weight_4_mag = 1
weight_4_ang = 1#2*math.pi/360

psse_data = scipy.io.loadmat('dataset_118bus.mat')
print(psse_data['input_all'].shape, psse_data['labels'].shape)

data_x = psse_data['input_all']
data_x = data_x
data_y = psse_data['labels']#236*18528
print(data_x.shape[0])

# scale the mags,
data_y[0:caseNo,:] = weight_4_mag*data_y[0:caseNo,:] #first 118 is magnitude
data_y[caseNo:,:] = weight_4_ang*data_y[caseNo:,:]  # second 118 is angle

# seperate them into training 80%, test 20%
split_train = int(0.8*psse_data['input_all'].shape[1]) #18528*0.8=14822  for trainning
split_val = psse_data['input_all'].shape[1] - split_train # 3706 for value
train_x = np.transpose(data_x[:, :split_train]) #14822*490
train_y = np.transpose(data_y[:, :split_train]) # 14822*236
val_x   = np.transpose(data_x[:, split_train:split_train+split_val]) # the remaining is for value  490, 14822:18528    3706*490
val_y   = np.transpose(data_y[:, split_train:split_train+split_val]) # 3706*236
test_x  = np.transpose(data_x[:, split_train+split_val:]) # 0
test_y  = np.transpose(data_y[:, split_train+split_val:]) # 0

print(train_x.shape, val_x.shape)
#Train the model
input_shape = (train_x.shape[1],)  ########################?????  (490,)
print(train_x.shape[1])
print(train_x)
epoch_num = 200
# psse_model = nn1_8H_psse(input_shape, train_y.shape[1])
psse_model = lav_psse(input_shape, train_y.shape[1])


print(train_y.shape[1])
psse_model.fit(train_x, train_y, epochs=epoch_num, batch_size=64)


save_file = '_'.join([str(caseNo), 'nn1_8H_PSSE',
                      'epoch', str(epoch_num)]) + '.h5'

if not os.path.exists('model_logs'):
    os.makedirs('model_logs')
save_path = os.path.join('model_logs', save_file)
print('\nSaving model weights to {:s}'.format(save_path))
psse_model.save_weights(save_path)


# evaluate the model
K.set_learning_phase(0)
val_predic = psse_model.predict(val_x)
scores = psse_model.evaluate(val_x, val_y)
print("\n%s: %.2f%%" % (psse_model.metrics_names[1], scores[1]*100))

#the self.defined distance metric since, to access the distance between predicted and the true
test_no = split_val
def rmse(val_predic, val_y, voltage_distance = np.zeros((test_no,caseNo)), voltage_norm = np.zeros((test_no,1))):
    for i in range(test_no):
        for j in range(caseNo):
            predic_r, predic_i = (1/weight_4_mag)* val_predic[i, j]*math.cos(val_predic[i, j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_predic[i,j]*math.sin(val_predic[i, j+caseNo]*2*math.pi/360)
            val_r, val_i = (1/weight_4_mag)*val_y[i,j]*math.cos(val_y[i,j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_y[i, j]*math.sin(val_y[i][j+caseNo]*2*math.pi/360)
            voltage_distance[i,j] = (predic_r-val_r)**2 + (predic_i-val_i)**2
        voltage_norm[i,] = np.sum(voltage_distance[i,:])

    return np.sqrt(np.mean(voltage_norm))

print("\n distance from the true states in terms of \|\|_2: %.4f" % rmse(val_y,val_predic))






