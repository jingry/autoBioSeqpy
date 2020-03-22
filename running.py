# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:06:26 2019

@author: jingr

processing the modeling 
"""

'''
rest
new flag for checking the parameters

dataloader for different model index
function for model merge
for predicting
Kmer 
check the labels, should be the same
label to mat
the input shape of the first layer (or change the function of reshape)

'''

import os, sys, re
sys.path.append(os.path.curdir)
sys.path.append(sys.argv[0])

import paraParser
if '--help' in sys.argv:
    paraParser.showHelpDoc()
    exit()

import moduleRead
import dataProcess
import analysisPlot
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
import tensorflow as tf

paraDict = paraParser.parseParameters(sys.argv[1:])

paraFile = paraDict['paraFile']
if not paraFile is None:
    paraDict = paraParser.parseParametersFromFile(paraFile)


#parameters
dataTypeList = paraDict['dataType']
dataEncodingType = paraDict['dataEncodingType']

firstKernelSize = paraDict['firstKernelSize']
print('debug0',firstKernelSize)
tmp = ','.join(firstKernelSize)
print('debug1',tmp)
if len(tmp) > 0:
    firstKernelSizes = eval(tmp)
else:
    firstKernelSizes = []
dataTrainFilePaths = paraDict['dataTrainFilePaths']
dataTrainLabel = paraDict['dataTrainLabel']
dataTestFilePaths = paraDict['dataTestFilePaths']
dataTestLabel = paraDict['dataTestLabel']
modelLoadFile = paraDict['modelLoadFile']
dataTrainModelInd = paraDict['dataTrainModelInd']
if len(modelLoadFile) == 1:
    dataTrainModelInd = [0] * len(dataTrainFilePaths)
dataTestModelInd = paraDict['dataTestModelInd']
if len(modelLoadFile) == 1:
    dataTestModelInd = [0] * len(dataTestFilePaths)
    
spcLen = paraDict['spcLen']
if len(spcLen) < 1:
    spcLen = [100] * len(modelLoadFile)
    
weightLoadFile = paraDict['weightLoadFile']
dataSplitScale = paraDict['dataSplitScale']
outSaveFolderPath = paraDict['outSaveFolderPath']
showFig = paraDict['showFig']
saveFig = paraDict['saveFig']
savePrediction = paraDict['savePrediction']
paraSaveName = paraDict['paraSaveName']

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
useKMerList = paraDict['useKMer']
if len(useKMerList ) == 0:
    useKMerList = [False] * len(modelLoadFile)
KMerNumList = paraDict['KMerNum']
if len(KMerNumList ) == 0:
    KMerNumList = [3] * len(modelLoadFile)
inputLength = paraDict['inputLength']
modelSaveName = paraDict['modelSaveName']
weightSaveName = paraDict['weightSaveName']
noGPU = paraDict['noGPU']
labelToMat = paraDict['labelToMat']

seed = paraDict['seed']
if seed < 0:
    seed = np.random.randint(int(1e9))

if not modelSaveName is None:
    if not modelSaveName.endswith('.json'):
        modelSaveName += '.json'
if not weightSaveName is None:
    if not weightSaveName.endswith('.bin'):
        weightSaveName += '.bin'

verbose = paraDict['verbose']

if verbose:
    print('Parameters:')
    paraParser.printParameters(paraDict)
    print('Generating dataset...')
    print('Checking the number of train files, which should be larger than 1 (e.g. at least two labels)...')
assert len(dataTrainFilePaths) > 1

if noGPU:
    if verbose:
        print('As set by user, gpu will be disabled.')
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
    
if verbose:
    print('Checking the number of the train files and the labels, they should be the same')    
assert len(dataTrainFilePaths) == len(dataTrainLabel)
    

if not len(dataTypeList) == len(modelLoadFile):
    if verbose:
        print('Please provide enough data type as the number of --modelLoadFile')
assert len(dataTypeList) == len(modelLoadFile)

featureGenerators = []
for i,subDataType in enumerate(dataTypeList):
    if subDataType.lower() == 'protein':
        if verbose:
            print('Enconding protein data...')
        featureGenerator = dataProcess.ProteinFeatureGenerator(dataEncodingType, useKMer=useKMerList[i], KMerNum=KMerNumList[i])
    elif subDataType.lower() == 'dna':
        if verbose:
            print('Enconding DNA data...')
        featureGenerator = dataProcess.DNAFeatureGenerator(dataEncodingType, useKMer=useKMerList[i], KMerNum=KMerNumList[i])
    elif subDataType.lower() == 'rna':
        if verbose:
            print('Enconding RNA data...')
        featureGenerator = dataProcess.RNAFeatureGenerator(dataEncodingType, useKMer=useKMerList[i], KMerNum=KMerNumList[i])
    elif subDataType.lower() == 'other':
        if verbose:
            print('Reading data in CSV format...')
        featureGenerator = dataProcess.OtherFeatureGenerator()
    else:
        print('Unknow dataType %r, please use \'protein\', \'dna\' ,\'rna\' or \'other\'' %subDataType)
    featureGenerators.append(featureGenerator)
    assert subDataType.lower() in ['protein','dna','rna','other']

if len(dataTestFilePaths) > 0:
    if verbose:
        print('test datafiles provided, the test dataset will be generated from the test datafiles...')
        print('Checking the number of test files, which should be larger than 0 (e.g. at least one labels)...')
    assert len(dataTestFilePaths) > 0
    if verbose:
        print('Checking the number of the test files and the labels, they should be the same')  
    assert len(dataTestFilePaths) == len(dataTestLabel)
    if verbose:
        print('Begin to generate train dataset...')
    
    trainDataLoadDict = {}
    for modelIndex in range(len(modelLoadFile)):
        trainDataLoadDict[modelIndex] = []
#    trainDataLoaders = []
    for i,dataPath in enumerate(dataTrainFilePaths):
        modelIndex = dataTrainModelInd[i]
        featureGenerator = featureGenerators[modelIndex]
        dataLoader = dataProcess.DataLoader(label = dataTrainLabel[i], featureGenerator=featureGenerator)
        dataLoader.readFile(dataPath, spcLen = spcLen[modelIndex])
        trainDataLoadDict[modelIndex].append(dataLoader)
    
    trainDataMats = []
    trainLabelArrs = []
    trainNameLists = []
    for modelIndex in range(len(modelLoadFile)):
        trainDataLoaders = trainDataLoadDict[modelIndex]
        trainDataSetCreator = dataProcess.DataSetCreator(trainDataLoaders)
        trainDataMat, trainLabelArr, nameList = trainDataSetCreator.getDataSet(toShuffle=False, seed=seed, withNameList=True)
        trainDataMats.append(trainDataMat)
        trainLabelArrs.append(trainLabelArr)
        trainNameLists.append(nameList)
      
    
    if verbose:
        print('Begin to generate test dataset...')
    
    testDataLoadDict = {}    
    for modelIndex in range(len(modelLoadFile)):
        testDataLoadDict[modelIndex] = []
#    testDataLoaders = []
    for i,dataPath in enumerate(dataTestFilePaths):
        modelIndex = dataTestModelInd[i]
        featureGenerator = featureGenerators[modelIndex]
        dataLoader = dataProcess.DataLoader(label = dataTestLabel[i], featureGenerator=featureGenerator)
        dataLoader.readFile(dataPath, spcLen = spcLen[modelIndex])
        testDataLoadDict[modelIndex].append(dataLoader)
    
    testDataMats = []
    testLabelArrs = []
    testNameLists = []
    for modelIndex in range(len(modelLoadFile)):
        testDataLoaders = testDataLoadDict[modelIndex]
        testDataSetCreator = dataProcess.DataSetCreator(testDataLoaders)
        testDataMat, testLabelArr, nameList = testDataSetCreator.getDataSet(toShuffle=False, seed=seed, withNameList=True)
        testDataMats.append(testDataMat)
        testLabelArrs.append(testLabelArr)
        testNameLists.append(nameList)
else:
    if verbose:
        print('No test datafiles provided, the test dataset will be generated by spliting the train datafiles...')
        print('Checking if the scale for spliting provided...')
        assert not dataSplitScale is None
        print('Generating the train and test datasets')
    
    trainDataLoadDict = {}
    for modelIndex in range(len(modelLoadFile)):
        trainDataLoadDict[modelIndex] = []
#    trainDataLoaders = []
    for i,dataPath in enumerate(dataTrainFilePaths):
        modelIndex = dataTrainModelInd[i]
        featureGenerator = featureGenerators[modelIndex]
        dataLoader = dataProcess.DataLoader(label = dataTrainLabel[i], featureGenerator=featureGenerator)
        dataLoader.readFile(dataPath, spcLen = spcLen[modelIndex])
        trainDataLoadDict[modelIndex].append(dataLoader)
        
    trainDataMats = []
    trainLabelArrs = []
    trainNameLists = []
    testDataMats = []
    testLabelArrs = []
    testNameLists = []
    for modelIndex in range(len(modelLoadFile)):
        trainDataLoaders = trainDataLoadDict[modelIndex]
        trainDataSetCreator = dataProcess.DataSetCreator(trainDataLoaders)
        trainDataMat, testDataMat, trainLabel, testLabel, namesTrain, namesTest = trainDataSetCreator.getTrainTestSet(dataSplitScale, toShuffle=False, seed=seed, withNameList=True)
        trainDataMats.append(trainDataMat)
        trainLabelArrs.append(trainLabel)
        trainNameLists.append(namesTrain)
        testDataMats.append(testDataMat)
        testLabelArrs.append(testLabel)
        testNameLists.append(namesTest)
    
    
#    trainDataLoaders = []
#    for i,dataPath in enumerate(dataTrainFilePaths):
#        dataLoader = dataProcess.DataLoader(label = dataTrainLabel[i], featureGenerator=featureGenerator)
#        dataLoader.readFile(dataPath, spcLen = spcLen)
#        trainDataLoaders.append(dataLoader)
#    trainDataSetCreator = dataProcess.DataSetCreator(trainDataLoaders)
#    trainDataMat, testDataMat, trainLabelArr, testLabelArr = trainDataSetCreator.getTrainTestSet(dataSplitScale, toShuffle=shuffleDataTrain, seed=seed)

if shuffleDataTrain:
    nameTemp = trainNameLists[0]
    np.random.seed = seed
    np.random.shuffle(nameTemp)
    trainDataMats, trainLabelArrs, sortedIndexes = dataProcess.matAlignByName(trainDataMats,nameTemp,trainLabelArrs,trainNameLists)
#    trainDataMat, trainLabelArr = dataProcess.matSuffleByRow(trainDataMat, trainLabelArr)
    
if shuffleDataTest:
    nameTemp = testNameLists[0]
    np.random.seed = seed
    np.random.shuffle(nameTemp)
    testDataMats, testLabelArrs, sortedIndexes = dataProcess.matAlignByName(testDataMat,nameTemp,testLabelArr,trainNameLists)

#    testDataMat, testLabelArr = dataProcess.matSuffleByRow(testDataMat, testLabelArr)

tmpTempLabel = trainLabelArrs[0]
for tmpLabel in trainLabelArrs:
    assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0

tmpTempLabel = testLabelArrs[0]
for tmpLabel in testLabelArrs:
    assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0
    
if labelToMat:
    if verbose:
        print('Since labelToMat is set, the labels would be changed to matrix')
    trainLabelArr,trainLabelArrDict,trainArrLabelDict = dataProcess.labelToMat(trainLabelArrs[0])
    testLabelArr,testLabelArrDict,testArrLabelDict = dataProcess.labelToMat(testLabelArrs[0])
else:
    trainLabelArr = trainLabelArrs[0]
    testLabelArr = testLabelArrs[0]
    

    
if verbose:
    print('Datasets generated')
    for i,trainDataMat in enumerate(trainDataMats):
        testDataMat = testDataMats[i]
        print('The %dth scales are:\n\ttraining: %d x %d\n\ttest: %d x %d' %(i,trainDataMat.shape[0],trainDataMat.shape[1],testDataMat.shape[0],testDataMat.shape[1]))    
    print('begin to prepare model...')
    
if not inputLength is None:
    if inputLength == 0:
        inputLength = trainDataMat.shape[1]



if verbose:
    print('Checking module file for modeling')
if len(modelLoadFile) < 1:
    if verbose:
        print('please provide a model file in a python script or a json file. You can find some examples in the \'model\' folder')
assert not len(modelLoadFile) < 1

models = []
for i,subModelFile in enumerate(modelLoadFile):
    weightFile = None
    if len(weightLoadFile) > 1:
        weightFile = weightLoadFile[i]
    if subModelFile.endswith('.py'):
        model = moduleRead.readModelFromPyFileDirectly(subModelFile,weightFile=weightFile)
    else:
        model = moduleRead.readModelFromJsonFileDirectly(subModelFile,weightFile=weightFile)
    models.append(model)

#input_length
if len(inputLength) < 1:
    inputLength = []
    for i,trainDataMat in enumerate( trainDataMats ):
        inputLength.append(trainDataMat.shape[1])        
moduleRead.modifyInputLengths(models,inputLength)



'''
#reshape and change first kernel size
for i,model in enumerate(models):
    subSpcLen = spcLen[i]
    if '2D' in str(model.layers[0].__class__):
        useKMer = useKMerList[i]
        if useKMer:
            KMerNum = KMerNumList[i]
            reshapeLen = subSpcLen - KMerNum + 1
        else:            
            reshapeLen = subSpcLen
        trainDataMat = trainDataMats[i]
        testDataMat = testDataMats[i]
        newShape = (int(trainDataMat.shape[1]/reshapeLen),reshapeLen)
        trainDataMats[i] = trainDataMat.reshape(trainDataMat.shape[0],int(trainDataMat.shape[1]/reshapeLen),reshapeLen,1)
        testDataMats[i] = testDataMat.reshape(testDataMat.shape[0],int(testDataMat.shape[1]/reshapeLen),reshapeLen,1)

    if len(firstKernelSizes) == 0:
        if verbose:
            print('Since the --firstKernelSizes is not provided, program will change it into (%d,3)' %(newShape[0]))
        firstKernelSize = (newShape[0],3)
        moduleRead.modifyFirstKenelSizeDirectly(model, firstKernelSize)
    else:
        firstKernelSize = tuple(firstKernelSizes[i])
        if verbose:
            print('--firstKernelSize %s is provided, program will use it' %(str(firstKernelSize)))
        moduleRead.modifyFirstKenelSizeDirectly(model, firstKernelSize)
'''
#first kernel size
if len(firstKernelSizes) > 0:
    for i,model in enumerate(models):
        firstKernelSize = tuple(firstKernelSizes[i])
        if verbose:
            print('--firstKernelSize %s is provided, program will use it' %(str(firstKernelSize)))
        moduleRead.modifyFirstKenelSizeDirectly(model, firstKernelSize)

#merge model
if len(modelLoadFile) > 1:
    try:
        model = moduleRead.modelMerge(models)
    except:
        model = moduleRead.modelMergeByAddReshapLayer(models, dataMats=trainDataMats, reshapeSize=None, verbose=verbose)
else:
    model = models[0]

moduleRead.modelCompile(model,loss = loss,optimizer = optimizer,metrics = metrics)

if verbose:
    print('Start training...')
#print(len(trainDataMats))
history = analysisPlot.LossHistory()
if len(trainDataMats) == 1:
    model.fit(trainDataMats[0], trainLabelArr,batch_size = batch_size,epochs = epochs,validation_split = 0.1,callbacks = [history])
else:
    model.fit(trainDataMats, trainLabelArr,batch_size = batch_size,epochs = epochs,validation_split = 0.1,callbacks = [history])
if verbose:
    print('Training finished, generating the summary of the module')
    model.summary()

if not outSaveFolderPath is None:
    if not os.path.exists(outSaveFolderPath):        
        os.makedirs(outSaveFolderPath, exist_ok=True)
    else:
        if verbose:
            print('outpath %s is exists, the outputs might be overwirten' %outSaveFolderPath)
if not modelSaveName is None:
    tmpModelOutPath = outSaveFolderPath + os.path.sep + modelSaveName
    tmpWeightOutPath = None
    if not weightSaveName is None:
        tmpWeightOutPath = outSaveFolderPath + os.path.sep + weightSaveName
    if verbose:
        print('\'modelSaveName\' provided, module will be saved at %s' %tmpModelOutPath)
        if not tmpWeightOutPath is None:
            print('Weights will be saved at %s' %tmpWeightOutPath)
    moduleRead.saveBuiltModel(model, tmpModelOutPath, weightPath=tmpWeightOutPath)

if verbose:
    tmpFigSavePath = None
    if showFig:
        print('Drawing figure recording loss...')    
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'epoch_loss.pdf'
        print('Saving figure recording loss at %s' %tmpFigSavePath)
    history.loss_plot('epoch',showFig=showFig,savePath=tmpFigSavePath)


predicted_Probability = model.predict(testDataMats)
if not 'predict_classes' in dir(model):
    prediction = np.rint(predicted_Probability)
else:
    prediction = model.predict_classes(testDataMats)


if labelToMat:
    testLabelArr = dataProcess.matToLabel(testLabelArr, testArrLabelDict)
else:
    testLabelArr = testLabelArrs[0]
print('Showing the confusion matrix')
cm=confusion_matrix(testLabelArr,prediction)
print(cm)
print("ACC: %f "%accuracy_score(testLabelArr,prediction))
if not labelToMat:
    print("F1: %f "%f1_score(testLabelArr,prediction))
    print("Recall: %f "%recall_score(testLabelArr,prediction))
    print("Pre: %f "%precision_score(testLabelArr,prediction))
    print("MCC: %f "%matthews_corrcoef(testLabelArr,prediction))
    print("AUC: %f "%roc_auc_score(testLabelArr,prediction))

if savePrediction:
    tmpPredictSavePath = outSaveFolderPath + os.path.sep + 'predicts'
    if verbose:
        print('Saving predictions at %s' %tmpPredictSavePath)
    with open(tmpPredictSavePath, 'w') as FIDO:
        FIDO.write('Label\tPrediction\tPobability\n')
        for i in range(len(testLabelArr)):
            tmpLabel = testLabelArr[i]
            tmpPrediction = prediction[i]
            while len(tmpPrediction.shape) > 0:
                tmpPrediction = tmpPrediction[0]
            tmpProbability = predicted_Probability[i]
#            tmpStr = '%r\t%r\t%r\n' %(tmpLabel,tmpPrediction,tmpProbability)
            if len(tmpProbability.shape) == 0:
                tmpStr = '%r\t%r\t%f\n' %(tmpLabel,tmpPrediction,tmpProbability)
            else:
                if len(tmpProbability) == 1:
                    tmpStr = '%r\t%r\t%f\n' %(tmpLabel,tmpPrediction,tmpProbability[0])
                else:
                    tmpStr = '%r\t%r\t[ %s ]\n' %(tmpLabel,tmpPrediction,' , '.join(tmpProbability.astype(str)))
            FIDO.write(tmpStr)
    tmpCMPath = outSaveFolderPath + os.path.sep + 'performance'
    if verbose:
        print('Saving confusion matrix and predicting performance at %s' %tmpPredictSavePath)
    with open(tmpCMPath, 'w') as FIDO:
        FIDO.write('Confusion Matrix:\n')
        for i in range(cm.shape[0]):
            tmpStr = ''
            for j in range(cm.shape[1]):
                tmpStr += '%d\t' %cm[i,j]
            tmpStr += '\n'
            FIDO.write(tmpStr)
        FIDO.write('Predicting Performance:\n')
        FIDO.write("ACC: %f \n"%accuracy_score(testLabelArr,prediction))
        if not labelToMat:
            FIDO.write("F1: %f \n"%f1_score(testLabelArr,prediction))
            FIDO.write("Recall: %f \n"%recall_score(testLabelArr,prediction))
            FIDO.write("Pre: %f \n"%precision_score(testLabelArr,prediction))
            FIDO.write("MCC: %f \n"%matthews_corrcoef(testLabelArr,prediction))
            FIDO.write("AUC: %f \n"%roc_auc_score(testLabelArr,prediction))
if not labelToMat:
    tmpFigSavePath = None
    if showFig:  
        if verbose:
            print('Plotting the ROC and PR curves...')
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'roc.pdf'
        if verbose:
            print('Saving figure recording ROC curve at %s' %tmpFigSavePath)        
    analysisPlot.plotROC(testLabelArr,predicted_Probability,showFig=showFig,savePath=tmpFigSavePath)
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'pr.pdf'
        if verbose:
            print('Saving figure recording ROC curve at %s' %tmpFigSavePath)       
        analysisPlot.plotPR(testLabelArr,predicted_Probability,showFig=showFig,savePath=tmpFigSavePath)

if not paraSaveName is None:
    tmpParaSavePath = outSaveFolderPath + os.path.sep + paraSaveName
    if verbose:
        print('Saving parameters at %s' %tmpParaSavePath)       
#    paraParser.saveParameters(tmpParaSavePath,sys.argv[1:])
    paraParser.saveParameters(tmpParaSavePath,paraDict)
print('Finished')