# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:06:26 2019

@author: jingr

processing the modeling 
"""

import os, sys, re
sys.path.append(os.path.curdir)
sys.path.append(sys.argv[0])

helpDoc = '''
The predicting script for using built model. 

Usage python predicting.py --dataTestFilePaths File1 File2 ... --paraFile filepath --predictionSavePath name [--verbose 0/1]

The ‘--dataTestFilePaths’ is the testdataset, the format is the same as the dataset for training.

The ‘--paraFile’ is the file recording the related parameters (could be extract by adding --paraSaveName when using running.py), this script will load the model according to the parameters.

The '--predictionSavePath' is the outfile which contain the predictions, otherwise the predictions will be printed to STDOUT


'''
import paraParser
if '--help' in sys.argv:
    print(helpDoc)
    exit()

import moduleRead
import dataProcess
#import analysisPlot
import numpy as np
#from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 

paraDictCMD = paraParser.parseParameters(sys.argv[1:])

paraFile = paraDictCMD['paraFile']
if not paraFile is None:
    paraDict = paraParser.parseParametersFromFile(paraFile)
    paraDict['dataTestFilePaths'] = paraDictCMD['dataTestFilePaths']
else:
    paraDict = paraDictCMD.copy()


#parameters
dataType = paraDict['dataType']
dataEncodingType = paraDict['dataEncodingType']
spcLen = paraDict['spcLen']
firstKernelSize = paraDict['firstKernelSize']
#dataTrainFilePaths = paraDict['dataTrainFilePaths']
#dataTrainLabel = paraDict['dataTrainLabel']
dataTestFilePaths = paraDict['dataTestFilePaths']
dataTestLabel = paraDict['dataTestLabel']
#modelLoadFile = paraDict['modelLoadFile']
#weightLoadFile = paraDict['weightLoadFile']
dataSplitScale = paraDict['dataSplitScale']
outSaveFolderPath = paraDict['outSaveFolderPath']
showFig = paraDict['showFig']
saveFig = paraDict['saveFig']
savePrediction = paraDict['savePrediction']

loss = paraDict['loss']
optimizer = paraDict['optimizer']
if not optimizer.startswith('optimizers.'):
    optimizer = 'optimizers.' + optimizer
if not optimizer.endswith('()'):
    optimizer = optimizer + '()'
metrics = paraDict['metrics']

shuffleDataTrain = paraDict['shuffleDataTrain']
shuffleDataTest = paraDict['shuffleDataTest']
batch_size = paraDict['batch_size']
epochs = paraDict['epochs']
useKMer = paraDict['useKMer']
KMerNum = paraDict['KMerNum']
inputLength = paraDict['inputLength']
modelSaveName = paraDict['modelSaveName']
weightSaveName = paraDict['weightSaveName']
noGPU = paraDict['noGPU']


#if not modelSaveName is None:
#    if not modelSaveName.endswith('.json'):
#        modelSaveName += '.json'
#if not weightSaveName is None:
#    if not weightSaveName.endswith('.bin'):
#        weightSaveName += '.bin'

modelLoadFile = outSaveFolderPath + os.path.sep + modelSaveName
weightLoadFile = outSaveFolderPath + os.path.sep + weightSaveName

verbose = paraDict['verbose']

predictionSavePath = None
for i,k in enumerate(sys.argv):
    if k == '--predictionSavePath':
        predictionSavePath = sys.argv[i+1]
    elif k == '--verbose':
        verbose = sys.argv[i+1]



#################################################
#for debug
#batch_size = 40
#epochs = 250

#useKMer = False
#KMerNum = 3

#inputLength = -1
#inputLength = None
#dataTrainFilePaths = ['D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\train\\train_pos.txt', 
#                 'D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\train\\train_neg.txt']
#dataTrainLabel = [1,0]
#dataTestFilePaths = ['D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\test\\test_pos.txt', 
#                 'D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\test\\test_neg.txt']
#dataTestLabel = [1,0]
#modelLoadFile = 'D:\\workspace\\proteinPredictionUsingDeepLearning\\models\\LSTM.py'
#verbose = True
#outSaveFolderPath = 'D:\\workspace\\proteinPredictionUsingDeepLearning\\tmpOut'
#savePrediction = True
#saveFig = True
################################################

#if verbose:
#    print('Parameters:')
#    paraParser.printParameters(paraDict)
#    print('Generating dataset...')
#    print('Checking the number of train files, which should be larger than 1 (e.g. at least two labels)...')
#assert len(dataTrainFilePaths) > 1

if noGPU:
    if verbose:
        print('As set by user, gpu will be disabled.')
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

#if verbose:
#    print('Checking the number of the train files and the labels, they should be the same')    
#assert len(dataTrainFilePaths) == len(dataTrainLabel)
    

if dataType is None:
    if verbose:
        print('NO data type provided, please provide a data type suce as \'protein\', \'dna\' or\'rna\'')
assert not dataType is None

if dataType.lower() == 'protein':
    if verbose:
        print('Enconding protein data...')
    featureGenerator = dataProcess.ProteinFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
elif dataType.lower() == 'dna':
    if verbose:
        print('Enconding DNA data...')
    featureGenerator = dataProcess.DNAFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
elif dataType.lower() == 'rna':
    if verbose:
        print('Enconding RNA data...')
    featureGenerator = dataProcess.RNAFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
else:
    print('Unknow dataType %r, please use \'protein\', \'dna\' or\'rna\'' %dataType)
assert dataType.lower() in ['protein','dna','rna']

if verbose:
    print('test datafiles should be provided, the test dataset will be generated from the test datafiles...')
    print('Checking the number of test files, which should be larger than 1 (e.g. at least two labels)...')
assert len(dataTestFilePaths) > 0
#if verbose:
#    print('Checking the number of the test files and the labels, they should be the same')  
#assert len(dataTestFilePaths) == len(dataTestLabel)
#if verbose:
#    print('Begin to generate train dataset...')
#
#trainDataLoaders = []
#for i,dataPath in enumerate(dataTrainFilePaths):
#    dataLoader = dataProcess.DataLoader(label = dataTrainLabel[i], featureGenerator=featureGenerator)
#    dataLoader.readFile(dataPath, spcLen = spcLen)
#    trainDataLoaders.append(dataLoader)
#trainDataSetCreator = dataProcess.DataSetCreator(trainDataLoaders)
#trainDataMat, trainLabelArr = trainDataSetCreator.getDataSet(toShuffle=shuffleDataTrain)

if verbose:
    print('Begin to generate test dataset...')
    
testDataLoaders = []
for i,dataPath in enumerate(dataTestFilePaths):
    #The label is set to 0, since we do not need the label for testing (only for accuracy calculating)
    dataLoader = dataProcess.DataLoader(label = 0, featureGenerator=featureGenerator)
    dataLoader.readFile(dataPath, spcLen = spcLen)
    testDataLoaders.append(dataLoader)
testDataSetCreator = dataProcess.DataSetCreator(testDataLoaders)
testDataMat, testLabelArr = testDataSetCreator.getDataSet(toShuffle=shuffleDataTest)

    
if verbose:
#    print('Datasets generated, the scales are:\n\ttraining: %d x %d\n\ttest: %d x %d' %(trainDataMat.shape[0],trainDataMat.shape[1],testDataMat.shape[0],testDataMat.shape[1]))    
    print('begin to prepare model...')
#    print('Loading keras model from .py files...')
    
if not inputLength is None:
    if inputLength == 0:
        inputLength = testDataMat.shape[1]



if verbose:
    print('Checking module file for modeling')
if modelLoadFile is None:
    if verbose:
        print('please provide a model file in a json file.')
if weightLoadFile is None:
    if verbose:
        print('the weight file is necessary for predicting, otherwise the model will be with initialized weight')
assert not modelLoadFile is None
assert not weightLoadFile is None
#if modelLoadFile.endswith('.py'):
#    if verbose:
#        print('Loading module from python file')
#        if not weightLoadFile is None:
#            print('Weights will be loaded at the same time')
#    model = moduleRead.getModelFromPyFile(modelLoadFile, weightFile=weightLoadFile, input_length=inputLength, loss=loss, optimizer=optimizer, metrics=metrics)
#else:
#    if verbose:
#        print('Loading module from Json file')
#        if not weightLoadFile is None:
#            print('Weights will be loaded at the same time')
#    model = moduleRead.getModelFromJsonFile(modelLoadFile, weightFile=weightLoadFile, input_length=inputLength, loss=loss, optimizer=optimizer, metrics=metrics)
if verbose:
    print('Loading module and weight file')
model = moduleRead.readModelFromJsonFileDirectly(modelLoadFile,weightLoadFile)
if verbose:
    print('Module loaded, generating the summary of the module')
    model.summary()
    
if '2D' in str(model.layers[0].__class__):
    if verbose:
        print('2D layer detected, data will be reshaped accroding to the \'spcLen\'')    
    if useKMer:
        reshapeLen = spcLen - KMerNum + 1
    else:            
        reshapeLen = spcLen 
    #newShape = (int(trainDataMat.shape[1]/spcLen),spcLen)
#    newShape = (int(trainDataMat.shape[1]/reshapeLen),reshapeLen)
#    trainDataMat = trainDataMat.reshape(trainDataMat.shape[0],int(trainDataMat.shape[1]/reshapeLen),reshapeLen,1)
    testDataMat = testDataMat.reshape(testDataMat.shape[0],int(testDataMat.shape[1]/reshapeLen),reshapeLen,1)
    
#    if len(firstKernelSize) == 0:
#        if verbose:
#            print('Since the --firstKernelSize is not provided, program will change it into (%d,3)' %(newShape[0]))
#        firstKernelSize = (newShape[0],3)
#        moduleRead.modifyModelFirstKernelSize(model, firstKernelSize)
#    else:
#        firstKernelSize = tuple(firstKernelSize)
#        if verbose:
#            print('--firstKernelSize %s is provided, program will use it' %(str(firstKernelSize)))
#        moduleRead.modifyModelFirstKernelSize(model, firstKernelSize)
    

#subLayer = model.layers[0]
#if 'input_length' in dir(subLayer):
#    subLayer.input_length = trainDataMat.shape[1]
#    subLayer.batch_input_shape = (None,trainDataMat.shape[1])
#    
#model = model_from_json(model.to_json())
#model.compile(loss = 'binary_crossentropy',optimizer = optimizers.Adam(),metrics = ['acc'])

#if verbose:
#    print('Start training...')
#history = analysisPlot.LossHistory()
#model.fit(trainDataMat, trainLabelArr,batch_size = batch_size,epochs = epochs,validation_split = 0.1,callbacks = [history])
#if verbose:
#    print('Training finished, generating the summary of the module')
#    model.summary()

if not outSaveFolderPath is None:
    if not os.path.exists(outSaveFolderPath):        
        os.makedirs(outSaveFolderPath, exist_ok=True)
    else:
        if verbose:
            print('outpath %s is exists, the outputs might be overwirten' %outSaveFolderPath)
#if not modelSaveName is None:
#    tmpModelOutPath = outSaveFolderPath + os.path.sep + modelSaveName
#    tmpWeightOutPath = None
#    if not weightSaveName is None:
#        tmpWeightOutPath = outSaveFolderPath + os.path.sep + weightSaveName
#    if verbose:
#        print('\'modelSaveName\' provided, module will be saved at %s' %tmpModelOutPath)
#        if not tmpWeightOutPath is None:
#            print('Weights will be saved at %s' %tmpWeightOutPath)
#    moduleRead.saveBuiltModel(model, tmpModelOutPath, weightPath=tmpWeightOutPath)

#if verbose:
#    tmpFigSavePath = None
#    if showFig:
#        print('Drawing figure recording loss...')    
#    if saveFig:
#        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'epoch_loss.pdf'
#        print('Saving figure recording loss at %s' %tmpFigSavePath)
#    history.loss_plot('epoch',showFig=showFig,savePath=tmpFigSavePath)


predicted_Probability = model.predict(testDataMat)
prediction = model.predict_classes(testDataMat)


#print('Showing the confusion matrix')
#cm=confusion_matrix(testLabelArr,prediction)
#print(cm)
#print("ACC: %f "%accuracy_score(testLabelArr,prediction))
#print("F1: %f "%f1_score(testLabelArr,prediction))
#print("Recall: %f "%recall_score(testLabelArr,prediction))
#print("Pre: %f "%precision_score(testLabelArr,prediction))
#print("MCC: %f "%matthews_corrcoef(testLabelArr,prediction))
#print("AUC: %f "%roc_auc_score(testLabelArr,prediction))

if not predictionSavePath is None:
    tmpPredictSavePath = predictionSavePath
    if verbose:
        print('Saving predictions at %s' %tmpPredictSavePath)
    with open(tmpPredictSavePath, 'w') as FIDO:
        FIDO.write('#Label\tPrediction\tPobability\n')
        for i in range(len(testLabelArr)):
            tmpLabel = i
            tmpPrediction = prediction[i]
            while len(tmpPrediction.shape) > 0:
                tmpPrediction = tmpPrediction[0]
            tmpProbability = predicted_Probability[i]
            tmpStr = '%r\t%r\t%f\n' %(tmpLabel,tmpPrediction,tmpProbability)
            FIDO.write(tmpStr)
else:
    if verbose:
        print('No save path prvided, the predictions will be listed in STDOUT')
    print('\n\n')
    print('#Instance\tPrediction\tPobability')
    for i in range(len(testLabelArr)):
        tmpLabel = i
        tmpPrediction = prediction[i]
        while len(tmpPrediction.shape) > 0:
            tmpPrediction = tmpPrediction[0]
        tmpProbability = predicted_Probability[i]
        tmpStr = '%r\t%r\t%f' %(tmpLabel,tmpPrediction,tmpProbability)
        print(tmpStr)
    print('\n\n')

#    tmpCMPath = outSaveFolderPath + os.path.sep + 'performance'
#    if verbose:
#        print('Saving confusion matrix and predicting performance at %s' %tmpPredictSavePath)
#    with open(tmpCMPath, 'w') as FIDO:
#        FIDO.write('Confusion Matrix:\n')
#        for i in range(cm.shape[0]):
#            tmpStr = ''
#            for j in range(cm.shape[1]):
#                tmpStr += '%d\t' %cm[i,j]
#            tmpStr += '\n'
#            FIDO.write(tmpStr)
#        FIDO.write('Predicting Performance:\n')
#        FIDO.write("ACC: %f \n"%accuracy_score(testLabelArr,prediction))
#        FIDO.write("F1: %f \n"%f1_score(testLabelArr,prediction))
#        FIDO.write("Recall: %f \n"%recall_score(testLabelArr,prediction))
#        FIDO.write("Pre: %f \n"%precision_score(testLabelArr,prediction))
#        FIDO.write("MCC: %f \n"%matthews_corrcoef(testLabelArr,prediction))
#        FIDO.write("AUC: %f \n"%roc_auc_score(testLabelArr,prediction))
#if verbose:
#    tmpFigSavePath = None
#    if showFig:        
#        print('Plotting the ROC curve...')
#    if saveFig:
#        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'roc.pdf'
#        print('Saving figure recording ROC curve at %s' %tmpFigSavePath)
#    analysisPlot.plotROC(testLabelArr,predicted_Probability,showFig=showFig,savePath=tmpFigSavePath)

print('Finished')