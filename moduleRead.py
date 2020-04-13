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
import keras
import numpy as np
from utils import TextDecorate
td = TextDecorate()
        
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
            print('Compling failed, cleaning the weight for recompiling')
        model = model_from_json(model.to_json())
        model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
#    model.compile()
#    model.summary()
    return model

def readModelFromJsonFileDirectly(jsonFile,weightFile=None):
    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    if not weightFile is None:
        # load weights into new model
        loaded_model.load_weights(weightFile)
    return loaded_model

def readModelFromPyFileDirectly(pyFilePath, weightFile=None):
    (folderPath, fileName) = os.path.split(pyFilePath)
    moduleName = re.sub('\.[^\.]+$','',fileName)
    sys.path.append(folderPath)
    obj=importlib.import_module(moduleName)
    model = obj.model
    if not weightFile is None:
        model.load_weights(weightFile)
    return model

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

def modelMerge(models,activation='sigmoid'):
    outputs = []
    inputs = []
    for model in models:
        outputs.append(model.output)
        inputs.append(model.input)
    concatenated = keras.layers.concatenate(outputs)
    out = keras.layers.Dense(1,activation=activation)(concatenated)    
    modelOut = keras.models.Model(inputs=inputs,outputs=out)
    return modelOut

def getPrimeFactor(intIn):
    '''
    Algorism is referring from:
    https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
    '''
    n = int(intIn)
    symbol = 1
    if n < 0:
        symbol = -1
        n = np.abs(n)
    outList = []  
    # Print the number of two's that divide n 
    while n % 2 == 0: 
        outList.append(2)
        n = n / 2
          
    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(np.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
        while n % i== 0: 
            outList.append(i) 
            n = n / i 
              
    # Condition if n is a prime 
    # number greater than 2 
    if n > 2: 
        outList.append(n) 
    return np.sort(outList),symbol

def reshapeSingleModelLayer(model,dataMat,reshapeSize=None,verbose=False,td=td):
    subLayer = model.layers[0]
#    if subLayer.name.lower().startswith('embedding'):
##        td.printC('Embedding layer detected as the input layer, will not change the size.','g')
#        subLayer = model.layers.pop(0)
#        newModel = keras.models.Sequential()
#        newModel.add(keras.layers.Embedding(subLayer.input_dim,subLayer.output_dim,input_length=dataMat.shape[1]))
#        newModel.add(model)
#        newModel.summary()
#        return newModel
    try:
        baseShape = subLayer.input_shape
#        print(1,baseShape)
    except:
        baseShape = [None]+list(subLayer.kernel_size)
#        print(2,baseShape)
        while len(baseShape) < 4:
            baseShape += [1]
    if len(baseShape) == 2:
        if verbose:
            td.printC('The input shape %s contains only 2 dimensions, which is unable to make the reshape. Thus one additional dimention will be added, please modify the model file if you don\'t want to change the input shape' %str(baseShape),'p')
        baseShape = list(baseShape) + [1]
    dataFeaLength = dataMat.shape[1]
    primeFactors,symbol = getPrimeFactor(dataFeaLength)
    primeFactors = list(primeFactors)
    #start from the 2nd dimention since the 1st is None set by keras
    assert dataFeaLength >= np.prod(baseShape[1:])
    if reshapeSize is None:
        if verbose:
            td.printC('No reshape size provided, will generate the reshape size according to the datashape', 'b')
        tmpShape = []
        for j in range(1,len(baseShape)):
            baseNum = baseShape[j]
            if  baseNum == 1:
                tmpShape.append(1)
            else:
                tmpNum = 1
                while tmpNum < baseNum:
                    assert len(primeFactors) > 0
                    tmpNum *= primeFactors.pop(0)
                tmpShape.append(tmpNum)
        if len(primeFactors) > 0:
            prodPosition = 0
#                    print(tmpShape)
            if np.sum(np.array(tmpShape)>1)>1:
                #if at least two values larger than 1, use the first value larger than 1 as the number for production
                for tmpVal in tmpShape:
                    if tmpVal > 1:
                        break
                    prodPosition += 1
            else:
                #if only one value larger than 1, use the first '1' as the number for production
                for tmpVal in tmpShape:
                    if tmpVal == 1:
                        break
                    prodPosition += 1
            tmpShape[prodPosition] *= np.prod(primeFactors)
                
        tmpShape = tuple(np.array(tmpShape,dtype=int))
        if verbose:
            td.printC('New shape %s generated and will be added as the input layer to the current model' %(str(tmpShape)),'g')
    else:
        if verbose:
            td.printC('Reshape size %s provided by user, will use it directly' %(str(reshapeSize)), 'g')
        tmpShape = reshapeSize
        tmpShape = tuple(np.array(tmpShape,dtype=int))
    newModel = keras.models.Sequential()
    newModel.add(keras.layers.Reshape(tuple(tmpShape), input_shape=(dataFeaLength,)))
    newModel.add(model)
    return newModel
    
def modelMergeByAddReshapLayer(models, dataMats, label, activation='sigmoid', reshapeSizes=None, verbose=False, td=td):
    dataLabel = np.array(label)
    outputs = [None] * len(models)
    inputs = [None] * len(models)
    for i,model in enumerate(models):
        if verbose:
            td.printC('Preparing to merge model %d ...' %i, 'b')
        if 'input_length' in dir(model.layers[0]):
            if not model.layers[0].input_length == dataMats[i].shape[1]:
                if verbose:
                    td.printC('The input length will be changed to %d as the related matrix', 'g')
                model.layers[0].input_length = dataMats[i].shape[1]
        try:
            modelInputShape = model.input.shape
    #        print(modelInputShape)
            shapeProdNum = None
            for tmpVal in modelInputShape:
                if 'value' in dir(tmpVal):
                    tmpVal = tmpVal.value
                if tmpVal is None:
                    continue
                else:
                    if shapeProdNum is None:
                        shapeProdNum = 1
                    shapeProdNum *= tmpVal
    #        assert (shapeProdNum is None) or (shapeProdNum == trainDataMats[0].shape[1])
            if not shapeProdNum is None:
    #            td.printC(str(shapeProdNum),'r')
                assert shapeProdNum == dataMats[0].shape[1]        
#            assert np.prod(model.layers[0].input_shape[1:]) == dataMats[i].shape[1]
            outputs[i] = model.output
            inputs[i] = model.input
            if verbose:
                td.printC('Model %d added successful' %i, 'g')
        except:
            if verbose:
                td.printC('Model %d added failed, trying to add a reshape layer' %i, 'b')
            if reshapeSizes is None:
                reshapeSize = None
            else:
                reshapeSize = reshapeSizes[i]
            newModel = reshapeSingleModelLayer(model,dataMats[i],reshapeSize=reshapeSize,verbose=verbose,td=td)
            
            
#            newModel.summary()
            outputs[i] = newModel.output
            inputs[i] = newModel.input
            models[i] = newModel
            if verbose:
                td.printC('Model %d with reshape layer added successful' %i, 'g')
    #check the shape for outputs, the dimensions should be the same
    outSizes = []
    outShapes = []
    for output in outputs:
        outSizes.append(len(output.shape))
        outShapes.append(output.shape)
    outSizes = np.array(outSizes)
    minSize = np.min(outSizes)
    if np.sum(outSizes > minSize) > 0:
        if verbose:
            td.printC('The shapes of the outputs %s are not consistent, the minimal is %d, others will be reduced to the same dimension.' %(str(list(outShapes)), minSize), 'b')
        for i, model in enumerate(models):
            outShapeOld = model.output.shape
            if len(outShapeOld) > minSize:
                outShapeNew = [1] * (minSize - 1) #excluding none
                for j,tmpVal in enumerate(outShapeOld[1:]):
                    tmpPos = j
                    while tmpPos >= (minSize - 1):
                        tmpPos -= minSize - 1
                    outShapeNew[tmpPos] *= int(tmpVal)
#                model.add(keras.layers.Reshape(outShapeNew, input_shape=outShapeOld))
                model.add(keras.layers.Reshape(outShapeNew))
                #only update the output
                outputs[i] = model.output
                inputs[i] = model.input
                if verbose:
                    td.printC('New shape %s generated and will be added as the out layer of model %d' %(str(outShapeNew), i),'g')
    #the name of the input should be different
    tmpCount = 0
    nameSet = set()
    for i,model in enumerate(models):
        changed = False
        for subLayer in model.layers:
            name = subLayer.name
            if name in nameSet:
                subLayer.name += '_%d' %tmpCount
                tmpCount += 1
                nameSet.add(subLayer.name)
                changed = True
            else:
                nameSet.add(name)
#        name = model.input.name
#        if name in nameSet:
#            model.input.name += ':%d' %tmpCount
#            tmpCount += 1
#            nameSet.add(model.input.name)
#            changed = True
        if changed:
            newModel = model_from_json(model.to_json())
            for layer in newModel.layers:
                try:
                    layer.set_weights(model.get_layer(name=layer.name).get_weights())
                except:
                    if verbose:
                        td.printC("Could not transfer weights for layer {}".format(layer.name),'p')
            outputs[i] = newModel.output
            inputs[i] = newModel.input
            models[i] = newModel
#        model.summary()
    #    concatenate the neural network by dense    
    concatenated = keras.layers.concatenate(outputs)
    outDim = int(np.prod(dataLabel.shape) / dataMats[0].shape[0])
    out = keras.layers.Dense(outDim,activation=activation)(concatenated)  

    modelOut = keras.models.Model(inputs=inputs,outputs=out)
#    modelOut.summary()
    return modelOut
        
def modelMergeByAddReshapLayer_old(models, dataMats, activation='sigmoid', reshapeSizes=None, verbose=False, td=td):
    outputs = [None] * len(models)
    inputs = [None] * len(models)
    for i,model in enumerate(models):
        if verbose:
            td.printC('Preparing to merge model %d ...' %i, 'b')
        if 'input_length' in dir(model.layers[0]):
            if not model.layers[0].input_length == dataMats[i].shape[1]:
                if verbose:
                    td.printC('The input length will be changed to %d as the related matrix', 'g')
                model.layers[0].input_length = dataMats[i].shape[1]
        try:
            assert np.prod(model.layers[0].input_shape[1:]) == dataMats[i].shape[1]
            outputs[i] = model.output
            inputs[i] = model.input
            if verbose:
                td.printC('Model %d added successful' %i, 'g')
        except:
            if verbose:
                td.printC('Model %d added failed, trying to add a reshape layer' %i, 'b')
            subLayer = model.layers[0]
            try:
                baseShape = subLayer.input_shape
            except:
                baseShape = [None]+list(subLayer.kernel_size)
                while len(baseShape) < 4:
                    baseShape += [1]
            if len(baseShape) == 2:
                if verbose:
                    td.printC('The input shape %s contains only 2 dimensions, which is unable to make the reshape. Thus one additional dimention will be added, please modify the model file if you don\'t want to change the input shape','p')
                baseShape = list(baseShape) + [1]
            dataFeaLength = dataMats[i].shape[1]
            primeFactors,symbol = getPrimeFactor(dataFeaLength)
            primeFactors = list(primeFactors)
            #start from the 2nd dimention since the 1st is None set by keras
            assert dataFeaLength >= np.prod(baseShape[1:])
            if reshapeSizes is None:
                if verbose:
                    td.printC('No reshape size provided, will generate the reshape size according to the datashape', 'b')
                tmpShape = []
                for j in range(1,len(baseShape)):
                    baseNum = baseShape[j]
                    if  baseNum == 1:
                        tmpShape.append(1)
                    else:
                        tmpNum = 1
                        while tmpNum < baseNum:
                            assert len(primeFactors) > 0
                            tmpNum *= primeFactors.pop(0)
                        tmpShape.append(tmpNum)
                if len(primeFactors) > 0:
                    prodPosition = 0
#                    print(tmpShape)
                    if np.sum(np.array(tmpShape)>1)>1:
                        #if at least two values larger than 1, use the first value larger than 1 as the number for production
                        for tmpVal in tmpShape:
                            if tmpVal > 1:
                                break
                            prodPosition += 1
                    else:
                        #if only one value larger than 1, use the first '1' as the number for production
                        for tmpVal in tmpShape:
                            if tmpVal == 1:
                                break
                            prodPosition += 1
                    tmpShape[prodPosition] *= np.prod(primeFactors)
                        
                tmpShape = tuple(np.array(tmpShape,dtype=int))
                if verbose:
                    td.printC('New shape %s generated and will be added as the input layer to the current model' %(str(tmpShape)),'g')
                newModel = keras.models.Sequential()    
            else:
                if verbose:
                    td.printC('Reshape size %r provided by user, will use it directly' %i, 'g')
                tmpShape = reshapeSizes[i]
                tmpShape = tuple(np.array(tmpShape,dtype=int))
            newModel.add(keras.layers.Reshape(tuple(tmpShape), input_shape=(dataFeaLength,)))
            newModel.add(model)
#            newModel.summary()
            outputs[i] = newModel.output
            inputs[i] = newModel.input
            models[i] = newModel
            if verbose:
                td.printC('Model %d with reshape layer added successful' %i, 'g')
    #check the shape for outputs, the dimensions should be the same
    outSizes = []
    outShapes = []
    for output in outputs:
        outSizes.append(len(output.shape))
        outShapes.append(output.shape)
    outSizes = np.array(outSizes)
    minSize = np.min(outSizes)
    if np.sum(outSizes > minSize) > 0:
        if verbose:
            td.printC('The shapes of the outputs %s are not consistent, the minimal is %d, others will be reduced to the same dimension.' %(str(list(outShapes)), minSize), 'b')
        for i, model in enumerate(models):
            outShapeOld = model.output.shape
            if len(outShapeOld) > minSize:
                outShapeNew = [1] * (minSize - 1) #excluding none
                for j,tmpVal in enumerate(outShapeOld[1:]):
                    tmpPos = j
                    while tmpPos >= (minSize - 1):
                        tmpPos -= minSize - 1
                    outShapeNew[tmpPos] *= int(tmpVal)
#                model.add(keras.layers.Reshape(outShapeNew, input_shape=outShapeOld))
                model.add(keras.layers.Reshape(outShapeNew))
                #only update the output
                outputs[i] = model.output
                inputs[i] = model.input
                if verbose:
                    td.printC('New shape %s generated and will be added as the out layer of model %d' %(str(outShapeNew), i),'g')
    #    concatenate the neural network by dense    
    concatenated = keras.layers.concatenate(outputs)
    out = keras.layers.Dense(1,activation=activation)(concatenated)    
    modelOut = keras.models.Model(inputs=inputs,outputs=out)
    return modelOut

def modifyInputLengths(models,inputLengths,layerNum=0,verbose=False,td=td):
    for i,model in enumerate(models):
        input_length = inputLengths[i]
        subLayer = model.layers[layerNum]
        if not input_length is None:
            if 'input_length' in dir(subLayer):
                if not subLayer.input_length == input_length:
                    if verbose:
                        td.printC('The input_length is not consistent with datashape, will be changed','b')
                    subLayer.input_length = input_length
                    subLayer.batch_input_shape = (None,input_length)
                    newModel = model_from_json(model.to_json())
                    if verbose:
                        td.printC('The input_length is changed, will trying to load the weights if possible (note that few layers might failed since the input_length was changed)','b')
                    for layer in newModel.layers:
                        try:
                            layer.set_weights(model.get_layer(name=layer.name).get_weights())
                        except:
                            if verbose:
                                td.printC("Could not transfer weights for layer {}".format(layer.name),'p')
                    models[i] = newModel

                
def modifyFirstKenelSizes(models,firstKernelSizes):
    for i,model in enumerate(models):
        firstKernelSize = firstKernelSizes[i]
        for subLayer in model.layers:
            if 'kernel_size' in dir(subLayer):
                subLayer.kernel_size = firstKernelSize
                break
            
def modifyFirstKenelSizeDirectly(model,firstKernelSize):
    for subLayer in model.layers:
        if 'kernel_size' in dir(subLayer):
            subLayer.kernel_size = firstKernelSize
            break
        
def modelCompile(model, loss = 'binary_crossentropy', optimizer = 'optimizers.Adam()', metrics = ['acc']):
    model.compile(loss = loss,optimizer = eval(optimizer),metrics = metrics)
#    return model