# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:39:29 2019

@author: jingr

Read the compiled modules from .py file 
"""

import importlib
import sys,os,re
#import keras.callbacks
from keras import optimizers
from keras.models import model_from_json

        
def getModelFromPyFile(pyFilePath, weightFile = None, input_length = None, loss = 'binary_crossentropy', optimizer = 'optimizers.Adam()', metrics = ['acc'], verbose=False):
    '''
    Read a .py file which contain a keras neural network.
    The templates is available in /Path of src/models
    
    Parameters:
        pyFilePath: the name of .py file
        weightFile: the name of weight file. The weight is the initial weight for the built model, if not provide, keras will generate it randomly.
        input_length: int, A keras parameter for 2D layer. This parameter is added 
                        to modify the size of the built model before compiling. 
                        The "batch_input_shape" and "input_length" will be changed 
                        according to this parameter. If not provided, program will change 
                        the size to the current shape automaticly if a 2D convolution 
                        layer is used as the first layer.
        loss:    string, Keras parameter, available candidates are 'mean_squared_error', 
                'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
                'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 
                'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 
                'poisson', 'cosine_proximity'
        optimizer: string, Keras parameters. Available candidates are SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, 
        metrics: string, Keras parameters. Available candidates are 'acc', 'mae', 
                'binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy',
                'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy'.
                 Note: The loss function is available here.
    '''
    (folderPath, fileName) = os.path.split(pyFilePath)
    moduleName = re.sub('\.[^\.]+$','',fileName)
    sys.path.append(folderPath)
    obj=importlib.import_module(moduleName)
    model = obj.model
    if not weightFile is None:
        model.load_weights(weightFile)
    subLayer = model.layers[0]
    if not input_length is None:
        if 'input_length' in dir(subLayer):
            subLayer.input_length = input_length
            subLayer.batch_input_shape = (None,input_length)
    #model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])
    try:
        if verbose:
            print('Compling loaded model for training')
        model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
    except:
        if verbose:
            print('Compling fialed, cleaning the weight for recompiling')
        model = model_from_json(model.to_json())
        model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
#    model.summary()
    return model

#pyFilePath = 'D:\\workspace\\proteinPredictionUsingDeepLearning\\models\\CNN_Conv1D+MaxPooling+bidirectional_LSTM.py'
#model = getModelFromPyFile(pyFilePath)

def getModelFromJsonFile(jsonFile,weightFile = None, input_length = None, loss = 'binary_crossentropy', optimizer = 'optimizers.Adam()', metrics = ['acc'], verbose=False):
    '''
    Kears is able to save a built module in JSON format, thus this function is to read the .json model.
    NOTE: The module will be recompiled when load.
    
    Parameters: the same as getModelFromPyFile
    '''
    # load json and create model
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    if not weightFile is None:
        loaded_model.load_weights(weightFile)
#    print("Loaded model from disk")
    model = loaded_model
    subLayer = model.layers[0]
    if not input_length is None:
        if 'input_length' in dir(subLayer):
            subLayer.input_length = input_length
            subLayer.batch_input_shape = (None,input_length)
    try:
        if verbose:
            print('Compling loaded model for training')
        model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
    except:
        if verbose:
            print('Compling fialed, cleaning the weight for recompiling')
        model = model_from_json(model.to_json())
        model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
#    model = model_from_json(model.to_json())
#    model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
#    model.compile()
#    model.summary()
    return model

def readModelFromJsonFileDirectly(jsonFile,weightFile):
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightFile)
    return loaded_model


def modifyModelFirstKernelSize(model, firstKernelSize, loss = 'binary_crossentropy', optimizer = 'optimizers.Adam()', metrics = ['acc']):
    '''
    Changing the kernel size of the first layer. Since the shape of input dataset might be not fit for the first layer,
    this function is added to modify the size of the built model before compiling.

    Parameters:
        model: a keras model which could be load from .py or .json file
        firstKernelSize: int, the keras parameter which will be changed
        others: the same with getModelFromPyFile
    '''
    for subLayer in model.layers:
        if 'kernel_size' in dir(subLayer):
            subLayer.kernel_size = firstKernelSize
            break
    model = model_from_json(model.to_json())
    model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
    return model

def saveBuiltModel(model, savePath, weightPath = None):
    # serialize model to JSON
    model_json = model.to_json()
    with open(savePath, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    if not weightPath is None:
        model.save_weights(weightPath)
    print("Saved model to disk")
################################################
## serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")
# 
## later...
# 
## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
##################################################