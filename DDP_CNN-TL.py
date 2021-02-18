import tensorflow as tf
tf.random.set_seed(10)
import random
random.seed()

#import matplotlib
import os
#import pickle
import numpy as np
#import pandas as pd
##import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
#import xarray as xr
#import seaborn as sns
from keras import layers

from keras.backend.tensorflow_backend import clear_session

from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
from keras.layers import Dense
from keras import Sequential
import h5py
import keras
#from pylab import plt
#from matplotlib import cm
from scipy.io import loadmat,savemat

# Memory usage
import psutil
process = psutil.Process(os.getpid())
print('Memory used by the process:')
print(process.memory_info().rss)  # in bytes 

import gc

## Data set has 100k data points

trainN=1500
testN=300
lead=1;
batch_size = 16
num_epochs = 50
pool_size = 2
drop_prob=0.0
conv_activation='relu'
Nlat=512
Nlon=512
n_channels=2
NT = 1800 # Numer of snapshots per file
numDataset = 1 # number of dataset / 2

print('Start....')

input_normalized=np.zeros([trainN+testN,Nlon, Nlat,n_channels],np.float32)
output_normalized=np.zeros([trainN+testN,Nlon,Nlat,1],np.float32)

def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon//2,Nlat//2,n_channels)),
            #layers.MaxPooling2D(pool_size=pool_size),
            #Dropout(drop_prob),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),
            # end "encoder"
    
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            # dense layers (flattening and reshaping happens automatically)
            ] + [keras.layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +

            [

            # start "Decoder" (mirror of the encoder above)
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            layers.Convolution2D(1, kernel_size, padding='same', activation=None)
            ]
            )
    optimizer= keras.optimizers.adam(lr=lr)


    model.compile(loss='mean_squared_error', optimizer = optimizer)

    return model

def build_model2(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon,Nlat,n_channels)),
            layers.MaxPooling2D(pool_size=pool_size),
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),
            #Dropout(drop_prob),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),
            # end "encoder"
    
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),

            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.MaxPooling2D(pool_size=pool_size),
            #keras.layers.Flatten()
            # dense layers (flattening and reshaping happens automatically)
            ] + [keras.layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +

            [
            #keras.layers.Reshape((8,8,conv_depth)),
            # start "Decoder" (mirror of the encoder above)
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            #Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            #layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            
            layers.Convolution2D(1, kernel_size, padding='same', activation=None)
            ]
            )
    #optimizer= keras.optimizers.adam(lr=lr)
    optimizer= tf.optimizers.Adam(lr=lr)


    model.compile(loss='mean_squared_error', optimizer = optimizer)

    return model

params = {'conv_depth': 64, 'hidden_size': 5000,
              'kernel_size': 5, 'lr': 0.00001, 'n_hidden_layers': 0}


for i in range(0,numDataset):

    Filename = '/oasis/scratch/comet/yg62/temp_project/Re64k/gpu training/TL8k_NX512/Data_for_TL_NX512.mat'

    with h5py.File(Filename, 'r') as f:
        input_normalized[:,:,:,:]=np.array(f['input'],np.float32).T
        output_normalized[:,:,:,0]=np.array(f['output'],np.float32).T
        f.close()

    index=np.random.permutation(trainN+testN)
    input_normalized=input_normalized[index,:,:,:]
    output_normalized=output_normalized[index,:,:,:]

    print('Finish Initialization')
    print(np.shape(input_normalized))
    print('Memory taken by input:')
    print(input_normalized.nbytes)
    print('Memory taken by output:')
    print(np.shape(output_normalized))
    print(output_normalized.nbytes)


    # Reset and free GPU memory
    #tf.keras.backend.clear_session()
    reset_keras()

    model = build_model(**params)
    model2 = build_model2(**params)

    #if i != 0:
    model.load_weights('./weights_cnn_KT') # load model weight from last time

    # Load the pre-trained model and fix the weights
    for layer in range(1,8):
        print(layer)
        print(model.layers[layer])
        print(model2.layers[layer+1])
        extracted_weights = model.layers[layer].get_weights()
        model2.layers[layer+1].set_weights(extracted_weights)

    model2.load_weights('./weights_cnn_KT_ext') # load model weight from last time
    for layer in model2.layers[2:-3]:
      layer.trainable = False
    
    optimizer= tf.optimizers.Adam(lr=0.00001)
    model.compile(loss='mean_squared_error', optimizer = optimizer)

    print(model2.summary())

    hist = model2.fit(input_normalized[0:trainN,:,:,:], output_normalized[0:trainN,:,:,:],
                 batch_size = batch_size,shuffle='True',
                 verbose=1,
                 epochs = num_epochs,
                 validation_data=(input_normalized[trainN:,:,:,:],output_normalized[trainN:,:,:,:]))

    model2.save_weights('./weights_cnn_KT_ext')
    
    #loss = hist.history['loss']
    #val_loss = hist.history['val_loss']
    #savemat('loss' + str(i) + '.mat' ,dict([('trainLoss',loss),('valLoss',val_loss)]))

    #del input_normalized
    #del output_normalized
    del hist
    #del f
    if i != numDataset-1:
        del model
    gc.collect()
    process = psutil.Process(os.getpid())
    print('Memory used by the process:')
    print(process.memory_info().rss)  # in bytes 
    print('finished training dataset' + str(i+1) + '/' + str(numDataset))


prediction=model2.predict(input_normalized[trainN:,:,:,:])

#print(np.shape(output_normalized[trainN:,:,:,:]))

#input_normalized[trainN:trainN+100,:,:,:]),

savemat('prediction_KT.mat',dict([('test',output_normalized[trainN:trainN+100,:,:,:]),('input',input_normalized[trainN:trainN+100,:,:,:]),('prediction',prediction[0:100,:,:])])) 
#savemat('Normalization_parameters.mat',dict([('SDEV_S',SDEV_S),('SDEV_W',SDEV_W),('SDEV_O',SDEV_O)]))