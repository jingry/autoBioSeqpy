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
import tensorflow as tf
from utils import TextDecorate, evalStrList, mergeDict
td = TextDecorate()






def predict(paraDict):
    #parameters
    #dataType = paraDict['dataType']
    dataTypeList = paraDict['dataType']
    dataEncodingType = paraDict['dataEncodingType']
    spcLen = paraDict['spcLen']
#    firstKernelSize = paraDict['firstKernelSize']
    #dataTrainFilePaths = paraDict['dataTrainFilePaths']
    #dataTrainLabel = paraDict['dataTrainLabel']
    dataTestFilePaths = paraDict['dataTestFilePaths']
#    dataTestLabel = paraDict['dataTestLabel']
    #modelLoadFile = paraDict['modelLoadFile']
    #weightLoadFile = paraDict['weightLoadFile']
#    dataSplitScale = paraDict['dataSplitScale']
    outSaveFolderPath = paraDict['outSaveFolderPath']
#    showFig = paraDict['showFig']
#    saveFig = paraDict['saveFig']
#    savePrediction = paraDict['savePrediction']
    
#    loss = paraDict['loss']
    optimizer = paraDict['optimizer']
    if not optimizer.startswith('optimizers.'):
        optimizer = 'optimizers.' + optimizer
    if not optimizer.endswith('()'):
        optimizer = optimizer + '()'
#    metrics = paraDict['metrics']
    
#    shuffleDataTrain = paraDict['shuffleDataTrain']
#    shuffleDataTest = paraDict['shuffleDataTest']
#    batch_size = paraDict['batch_size']
#    epochs = paraDict['epochs']
    
    #useKMer = paraDict['useKMer']
    #KMerNum = paraDict['KMerNum']
#    inputLength = paraDict['inputLength']
    modelSaveName = paraDict['modelSaveName']
    weightSaveName = paraDict['weightSaveName']
    noGPU = paraDict['noGPU']
    labelToMat = paraDict['labelToMat']
    
    
    modelLoadFile = paraDict['modelLoadFile']
    useKMerList = paraDict['useKMer']
    if len(useKMerList ) == 0:
        useKMerList = [False] * len(modelLoadFile)
    KMerNumList = paraDict['KMerNum']
    if len(KMerNumList ) == 0:
        KMerNumList = [3] * len(modelLoadFile)
        
    #dataTrainModelInd = paraDict['dataTrainModelInd']
    #if len(modelLoadFile) == 1:
    #    dataTrainModelInd = [0] * len(dataTrainFilePaths)
    dataTestModelInd = paraDict['dataTestModelInd']
    if len(modelLoadFile) == 1:
        dataTestModelInd = [0] * len(dataTestFilePaths)
        
    modelPredictFile = outSaveFolderPath + os.path.sep + modelSaveName
    
    
    
    weightLoadFile = outSaveFolderPath + os.path.sep + weightSaveName
    
    verbose = paraDict['verbose']
    
    predictionSavePath = None
    for i,k in enumerate(sys.argv):
        if k == '--predictionSavePath':
            predictionSavePath = sys.argv[i+1]
        elif k == '--verbose':
            verbose = sys.argv[i+1]
    
    colorText = paraDict['colorText']
    if colorText.lower() == 'auto':
        import platform
        if 'win' in platform.system().lower():
            td.disable()
    elif not bool(eval(colorText)):
        td.disable()
        
    if noGPU:
        if verbose:
            td.printC('As set by user, gpu will be disabled.','g')
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    else:
        #check the version of tensorflow before configuration
        tfVersion = tf.__version__
        if int(tfVersion.split('.')[0]) >= 2:
    #        config = tf.compat.v1.ConfigProto(allow_growth=True)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
    
            sess =tf.compat.v1.Session(config=config)
        else:
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            sess = tf.Session(config=config)
    
    
    if not len(dataTypeList) == len(modelLoadFile):
        if verbose:
            td.printC('Please provide enough data type(s) as the number of --modelLoadFile','r')
    assert len(dataTypeList) == len(modelLoadFile)
    
    featureGenerators = []
    for i,subDataType in enumerate(dataTypeList):
        if subDataType.lower() == 'protein':
            if verbose:
                td.printC('Enconding protein data for model %d ...' %i,'b')
            featureGenerator = dataProcess.ProteinFeatureGenerator(dataEncodingType[i], useKMer=useKMerList[i], KMerNum=KMerNumList[i])
        elif subDataType.lower() == 'dna':
            if verbose:
                td.printC('Enconding DNA data for model %d ...' %i,'b')
            featureGenerator = dataProcess.DNAFeatureGenerator(dataEncodingType[i], useKMer=useKMerList[i], KMerNum=KMerNumList[i])
        elif subDataType.lower() == 'rna':
            if verbose:
                td.printC('Enconding RNA data for model %d ...' %i,'b')
            featureGenerator = dataProcess.RNAFeatureGenerator(dataEncodingType[i], useKMer=useKMerList[i], KMerNum=KMerNumList[i])
        elif subDataType.lower() == 'other':
            if verbose:
                td.printC('Reading CSV-like data for model %d ...' %i,'b')
            featureGenerator = dataProcess.OtherFeatureGenerator()
        elif subDataType.lower() == 'smiles':
            if verbose:
                td.printC('Enconding Smiles data for model %d ...' %i,'b')
            featureGenerator = dataProcess.SmilesFeatureGenerator(dataEncodingType[i], useKMer=useKMerList[i], KMerNum=KMerNumList[i])
        else:
            td.printC('Unknow dataType %r, please use \'protein\', \'dna\' ,\'rna\' or \'other\'' %subDataType, 'r')
        featureGenerators.append(featureGenerator)
        assert subDataType.lower() in ['protein','dna','rna','other','smiles']
    
    #if dataType is None:
    #    if verbose:
    #        print('NO data type provided, please provide a data type suce as \'protein\', \'dna\' or\'rna\'')
    #assert not dataType is None
    #
    #if dataType.lower() == 'protein':
    #    if verbose:
    #        print('Enconding protein data...')
    #    featureGenerator = dataProcess.ProteinFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
    #elif dataType.lower() == 'dna':
    #    if verbose:
    #        print('Enconding DNA data...')
    #    featureGenerator = dataProcess.DNAFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
    #elif dataType.lower() == 'rna':
    #    if verbose:
    #        print('Enconding RNA data...')
    #    featureGenerator = dataProcess.RNAFeatureGenerator(dataEncodingType, useKMer=useKMer, KMerNum=KMerNum)
    #else:
    #    print('Unknow dataType %r, please use \'protein\', \'dna\' or\'rna\'' %dataType)
    #assert dataType.lower() in ['protein','dna','rna']
    
    
    
    
    if verbose:
        td.printC('Checking the number of test files, which should be larger than 1 (e.g. at least two labels)...','b')
    assert len(dataTestFilePaths) > 0
    
    if verbose:
        td.printC('Begin to generate test dataset...','b')
    
    testDataLoadDict = {}    
    for modelIndex in range(len(modelLoadFile)):
        testDataLoadDict[modelIndex] = []
    #    testDataLoaders = []
    for i,dataPath in enumerate(dataTestFilePaths):
        modelIndex = dataTestModelInd[i]
        featureGenerator = featureGenerators[modelIndex]
        dataLoader = dataProcess.DataLoader(label = 0, featureGenerator=featureGenerator)
        dataLoader.readFile(dataPath, spcLen = spcLen[modelIndex])
        testDataLoadDict[modelIndex].append(dataLoader)
    
    testDataMats = []
    testLabelArrs = []
    testNameLists = []
    for modelIndex in range(len(modelLoadFile)):
        testDataLoaders = testDataLoadDict[modelIndex]
        testDataSetCreator = dataProcess.DataSetCreator(testDataLoaders)
        testDataMat, testLabelArr, nameList = testDataSetCreator.getDataSet(toShuffle=False, withNameList=True)
        testDataMats.append(testDataMat)
        testLabelArrs.append(testLabelArr)
        testNameLists.append(nameList)
    if verbose:
        td.printC('Test datasets generated.','g')
    nameTemp = testNameLists[0]    
    testDataMats, testLabelArrs, sortedIndexes = dataProcess.matAlignByName(testDataMats,nameTemp,testLabelArrs,testNameLists)
    testNameLists = [nameTemp] * len(testNameLists)
        
    tmpTempLabel = testLabelArrs[0]
    for tmpLabel in testLabelArrs:
        assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0   
    #
    #    
    #testDataLoaders = []
    #for i,dataPath in enumerate(dataTestFilePaths):
    #    #The label is set to 0, since we do not need the label for testing (only for accuracy calculating)
    #    dataLoader = dataProcess.DataLoader(label = 0, featureGenerator=featureGenerator)
    #    dataLoader.readFile(dataPath, spcLen = spcLen)
    #    testDataLoaders.append(dataLoader)
    #testDataSetCreator = dataProcess.DataSetCreator(testDataLoaders)
    #testDataMat, testLabelArr = testDataSetCreator.getDataSet(toShuffle=shuffleDataTest)
    
    if labelToMat:
        if verbose:
            td.printC('Since labelToMat is set, the labels would be changed to onehot-like matrix','g')
        testLabelArr,testLabelArrDict,testArrLabelDict = dataProcess.labelToMat(testLabelArrs[0])
    #    print(testLabelArr)
    else:
        testLabelArr = testLabelArrs[0]
    
    
        
    if verbose:
    #    print('Datasets generated, the scales are:\n\ttraining: %d x %d\n\ttest: %d x %d' %(trainDataMat.shape[0],trainDataMat.shape[1],testDataMat.shape[0],testDataMat.shape[1]))    
        td.printC('begin to prepare model...','b')
    #    print('Loading keras model from .py files...')
        
    
    #
    #if not inputLength is None:
    #    if inputLength == 0:
    #        inputLength = testDataMat.shape[1]
    
    
    
    if verbose:
        td.printC('Checking module file for modeling','b')
    if modelPredictFile is None:
        if verbose:
            td.printC('please provide a model file in a json file.','r')
    if weightLoadFile is None:
        if verbose:
            td.printC('the weight file is necessary for predicting, otherwise the model will be with initialized weight','r')
    assert not modelPredictFile is None
    assert not weightLoadFile is None
    
    if verbose:
        td.printC('Loading module and weight file','b')
    custom_objects_list = []
    for i,subModelFile in enumerate(modelLoadFile):
        if subModelFile.endswith('.py'):
            tmpObj = moduleRead.getCustomObjects(subModelFile)
            if not tmpObj is None:
                custom_objects_list.append(tmpObj)
    
    if len(custom_objects_list) > 0:
        print(custom_objects_list)
        custom_objects = mergeDict(custom_objects_list)
    else:
        custom_objects = None
        
    model = moduleRead.readModelFromJsonFileDirectly(modelPredictFile,weightLoadFile,custom_objects=custom_objects)
    if verbose:
        td.printC('Module loaded, generating the summary of the module','b')
        model.summary()
    #    
    #if '2D' in str(model.layers[0].__class__):
    #    if verbose:
    #        print('2D layer detected, data will be reshaped accroding to the \'spcLen\'')    
    #    if useKMer:
    #        reshapeLen = spcLen - KMerNum + 1
    #    else:            
    #        reshapeLen = spcLen 
    #    #newShape = (int(trainDataMat.shape[1]/spcLen),spcLen)
    ##    newShape = (int(trainDataMat.shape[1]/reshapeLen),reshapeLen)
    ##    trainDataMat = trainDataMat.reshape(trainDataMat.shape[0],int(trainDataMat.shape[1]/reshapeLen),reshapeLen,1)
    #    testDataMat = testDataMat.reshape(testDataMat.shape[0],int(testDataMat.shape[1]/reshapeLen),reshapeLen,1)
    #    
    
    if not outSaveFolderPath is None:
        if not os.path.exists(outSaveFolderPath):        
            os.makedirs(outSaveFolderPath, exist_ok=True)
        else:
            if verbose:
                td.printC('outpath %s is exists, the outputs might be overwirten' %outSaveFolderPath,'p')
    
    
    predicted_Probability = model.predict(testDataMats)
    if not 'predict_classes' in dir(model):
        prediction = np.rint(predicted_Probability)
    #    if labelToMat:
    #        prediction = dataProcess.matToLabel(np.array(prediction,dtype=int), testArrLabelDict,td=td)
    else:
        prediction = model.predict_classes(testDataMats)
    return prediction,predicted_Probability,testNameLists,testDataMats,testLabelArr,model


if __name__ == '__main__':
    paraDictCMD = paraParser.parseParameters(sys.argv[1:])

    paraFile = paraDictCMD['paraFile']
    if not paraFile is None:
        paraDict = paraParser.parseParametersFromFile(paraFile)
        paraDict['dataTestFilePaths'] = paraDictCMD['dataTestFilePaths']
        paraDict['dataTestModelInd'] = paraDictCMD['dataTestModelInd']
    else:
        paraDict = paraDictCMD.copy()
        
    prediction,predicted_Probability,testNameLists,testDataMats,testLabelArr,model = predict(paraDict)    
    verbose = paraDict['verbose']
    predictionSavePath = None
    for i,k in enumerate(sys.argv):
        if k == '--predictionSavePath':
            predictionSavePath = sys.argv[i+1]
        elif k == '--verbose':
            verbose = sys.argv[i+1]
            
    nameTemp = testNameLists[0]
    
    if not predictionSavePath is None:
        tmpPredictSavePath = predictionSavePath
        if verbose:
            td.printC('Saving predictions at %s' %tmpPredictSavePath,'g')
        with open(tmpPredictSavePath, 'w') as FIDO:
            FIDO.write('#Name\tIndex\tPrediction\tProbability\n')
            for i in range(len(prediction)):
                tmpLabel = i
                tmpPrediction = prediction[i]
                while len(tmpPrediction.shape) > 0:
                    tmpPrediction = tmpPrediction[0]
                tmpProbability = predicted_Probability[i]
    #            tmpStr = '%r\t%r\t%f\n' %(tmpLabel,tmpPrediction,tmpProbability)
                if len(tmpProbability.shape) == 0:
                    tmpStr = '%s\t%r\t%r\t%f\n' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability)
                else:
                    if len(tmpProbability) == 1:
                        tmpStr = '%s\t%r\t%r\t%f\n' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability[0])
                    else:
                        tmpStr = '%s\t%r\t%r\t[ %s ]\n' %(nameTemp[i],tmpLabel,tmpPrediction,' , '.join(tmpProbability.astype(str)))
                FIDO.write(tmpStr)
    else:
        if verbose:
            td.printC('No save path provided, the predictions will be listed in STDOUT','p')
        print('\n\n')
    #    print(predicted_Probability)
        print('#Name\tIndex\tPrediction\tProbability')
        for i in range(len(prediction)):
            tmpLabel = i
            tmpPrediction = prediction[i]
            while len(tmpPrediction.shape) > 0:
                tmpPrediction = tmpPrediction[0]
            tmpProbability = predicted_Probability[i]
            if len(tmpProbability.shape) == 0:
                tmpStr = '%s\t%r\t%r\t%f' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability)
            else:
                if len(tmpProbability) == 1:
                    tmpStr = '%s\t%r\t%r\t%f' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability[0])
                else:
                    tmpStr = '%s\t%r\t%r\t[ %s ]' %(nameTemp[i],tmpLabel,tmpPrediction,' , '.join(tmpProbability.astype(str)))
            print(tmpStr)
        print('\n\n')
    
    
    td.printC('Finished','g')
