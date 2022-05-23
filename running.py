# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:06:26 2019

@author: jingr

processing the modeling 
"""

'''
rest

new flag for checking the parameters (file name!)
###dataloader for different model index
###function for model merge
#for predicting
#Kmer 
###check the labels, should be the same
#label to mat
#the input shape of the first layer (or change the function of reshape)
#--reshapeSize
descriptions
checke the samples kept the features after shuffled

manual (the relation of the parameters and the reshape size, matrix in datatype)
jupyter notebook
'''

#%% init
import matplotlib
defaultBackEnd = matplotlib.get_backend()
matplotlib.use('Agg')

import os, sys, re
sys.path.append(os.path.curdir)
sys.path.append(sys.argv[0])

import paraParser
if '-h' in sys.argv or '--help' in sys.argv:
    paraParser.showHelpDoc()
    exit()

import moduleRead
import dataProcess
import analysisPlot
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
import tensorflow as tf
import keras
from utils import TextDecorate, evalStrList, mergeDict
td = TextDecorate()

paraDict = paraParser.parseParameters(sys.argv[1:])

paraFile = paraDict['paraFile']
if not paraFile is None:
    paraDict = paraParser.parseParametersFromFile(paraFile)


#parameters
dataTypeList = paraDict['dataType']
dataEncodingType = paraDict['dataEncodingType']

firstKernelSize = paraDict['firstKernelSize']
#tmp = ','.join(firstKernelSize)
#tmp = re.sub('\[\,+\[','[[',tmp)
#tmp = re.sub('\[\,+([^\]])','[\\1',tmp)
#tmp = re.sub('\],+\[','],[',tmp)
#tmp = re.sub('\]\,+\]',']]',tmp)
if len(firstKernelSize) > 0:
#    firstKernelSizes = eval(tmp)
    firstKernelSizes = evalStrList(firstKernelSize)
else:
    firstKernelSizes = []
    
reshapeSize = paraDict['reshapeSize']
if len(reshapeSize) > 0:
    reshapeSizes = evalStrList(reshapeSize)
else:
    reshapeSizes = None

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
if showFig:
    matplotlib.use(defaultBackEnd)

saveFig = paraDict['saveFig']
savePrediction = paraDict['savePrediction']
paraSaveName = paraDict['paraSaveName']

loss = paraDict['loss']
optimizer = paraDict['optimizer']
if not optimizer.startswith('optimizers.'):
    optimizer = 'optimizers.' + optimizer
if not optimizer.endswith(')'):
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

colorText = paraDict['colorText']
if colorText.lower() == 'auto':
    import platform
    if 'win' in platform.system().lower():
        td.disable()
elif not bool(eval(colorText)):
    td.disable()

verbose = paraDict['verbose']

#%%checking parameters
if verbose:
    td.printC('Parameters:','B')
    paraParser.printParameters(paraDict)
    td.printC('Generating dataset...','b')
    td.printC('Checking the number of train files, which should be larger than 1 (e.g. at least two labels)...','b')
assert len(dataTrainFilePaths) > 1
if verbose:
    td.printC('......OK','g')

if noGPU:
    if verbose:
        td.printC('As set by user, gpu will be disabled.','g')
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
else:
    if verbose:
        td.printC('Using gpu if possible...','g')
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
    td.printC('Checking the number of the train files and the labels, they should be the same','b')    
assert len(dataTrainFilePaths) == len(dataTrainLabel)
if verbose:
    td.printC('......OK','g')

if not len(dataTypeList) == len(modelLoadFile):
    if verbose:
        td.printC('Please provide enough data type(s) as the number of --modelLoadFile','r')
assert len(dataTypeList) == len(modelLoadFile)

if not len(spcLen) == len(modelLoadFile):
    if verbose:
        td.printC('The number of --spcLen should be the same as --modelLoadFile','r')
assert len(spcLen) == len(modelLoadFile)

#%% feature generator
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
    elif subDataType.lower() == 'smiles':
        if verbose:
            td.printC('Enconding Smiles data for model %d ...' %i,'b')
        featureGenerator = dataProcess.SmilesFeatureGenerator(dataEncodingType[i], useKMer=useKMerList[i], KMerNum=KMerNumList[i])
    elif subDataType.lower() == 'other':
        if verbose:
            td.printC('Reading CSV-like data for model %d ...' %i,'b')
        featureGenerator = dataProcess.OtherFeatureGenerator()
    else:
        td.printC('Unknow dataType %r, please use \'protein\', \'dna\' ,\'rna\' or \'other\'' %subDataType, 'r')
    featureGenerators.append(featureGenerator)
    assert subDataType.lower() in ['protein','dna','rna','smiles','other']

#%% dataset generating
if len(dataTestFilePaths) > 0:
    if verbose:
        td.printC('test datafiles provided, the test dataset will be generated from the test datafiles...', 'b')
    if verbose:
        td.printC('Checking the number of the test files and the labels, they should be the same', 'b')  
    assert len(dataTestFilePaths) == len(dataTestLabel)
    
    if not len(dataTrainModelInd) == len(dataTrainFilePaths):
        td.printC('The length of --dataTrainModelInd and --dataTrainFilePaths are not the same','r')
        assert len(dataTrainModelInd) == len(dataTrainFilePaths)
    
    if not len(dataTestModelInd) == len(dataTestFilePaths):
        td.printC('The length of --dataTestModelInd and --dataTestFilePaths are not the same','r')
        assert len(dataTrainModelInd) == len(dataTrainFilePaths)
    
    if verbose:
        td.printC('......OK','g')
        

        
    if verbose:
        td.printC('Begin to generate training datasets...','b')
    
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
        
    nameTemp = trainNameLists[0].copy()
#    print(trainNameLists)
    trainDataMats, trainLabelArrs, sortedIndexes = dataProcess.matAlignByName(trainDataMats,nameTemp,trainLabelArrs,trainNameLists)
    trainNameLists = [nameTemp] * len(trainNameLists)
      
    
    if verbose:
        td.printC('Training datasets generated.','g')
        td.printC('Begin to generate test datasets...','b')
    
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
        
    nameTemp = testNameLists[0].copy()
    testDataMats, testLabelArrs, sortedIndexes = dataProcess.matAlignByName(testDataMats,nameTemp,testLabelArrs,testNameLists)
    testNameLists = [nameTemp] * len(testNameLists)
    
    if verbose:
        td.printC('Test datasets generated.','g')
else:
    if verbose:
        td.printC('No test datafiles provided, the test dataset will be generated by spliting the train datafiles...','b')
        td.printC('Checking if the scale for spliting (--dataSplitScale) provided...','b')
        assert not dataSplitScale is None
        td.printC('......OK','g')
    if not len(dataTrainModelInd) == len(dataTrainFilePaths):
        td.printC('The length of --dataTrainModelInd and --dataTrainFilePaths are not the same','r')
        assert len(dataTrainModelInd) == len(dataTrainFilePaths)
        td.printC('Generating the train and test datasets','b')
    
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
    #the split of the name becomes complex since different dataloader could get different sequence of the sample
    #thus the matrix should be alignmented once before split
    nameTemp = trainNameLists[0].copy()
    np.random.shuffle(nameTemp) #shuffle the dataset at first
    trainDataMats, trainLabelArrs, sortedIndexes = dataProcess.matAlignByName(trainDataMats,nameTemp,trainLabelArrs,trainNameLists)
    trainNameLists = [nameTemp] * len(trainNameLists)
    
#    tmpTempLabel = trainLabelArrs[0]
#    for tmpLabel in trainLabelArrs:
#        assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0
    
    trainDataMatsNew = []
    trainLabelArrsNew = []
    trainNameListsNew = []
    testDataMatsNew = []
    testLabelArrsNew = []
    testNameListsNew = []
    indexArr = np.arange(len(trainLabelArrs[0]),dtype=int)
    np.random.shuffle(indexArr)
    for modelIndex in range(len(modelLoadFile)):
        matIn = trainDataMats[modelIndex]
        label = trainLabelArrs[modelIndex]
        nameList = trainNameLists[modelIndex]
        trainDataMat, testDataMat, trainLabel, testLabel, namesTrain, namesTest = dataProcess.splitMatByScaleAndIndex(dataSplitScale, matIn, label, indexArr, nameList = nameList)    
        trainDataMatsNew.append(trainDataMat)
        trainLabelArrsNew.append(trainLabel)
        trainNameListsNew.append(namesTrain)
        testDataMatsNew.append(testDataMat)
        testLabelArrsNew.append(testLabel)
        testNameListsNew.append(namesTest)
    trainDataMats = trainDataMatsNew
    trainLabelArrs = trainLabelArrsNew
    trainNameLists = trainNameListsNew
    testDataMats = testDataMatsNew
    testLabelArrs = testLabelArrsNew
    testNameLists = testNameListsNew
        
    '''    
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
    '''
    if verbose:
        td.printC('Training and test datasets generated.','g')
    
#nameTemp = trainNameLists[0]
if shuffleDataTrain:
    if verbose:
        td.printC('Shuffling training datasets.','b')
#    print(nameTemp)
    nameTemp = trainNameLists[0].copy()
    np.random.seed = seed
    np.random.shuffle(nameTemp)
    trainDataMats, trainLabelArrs, sortedIndexes = dataProcess.matAlignByName(trainDataMats,nameTemp,trainLabelArrs,trainNameLists)
    trainNameLists = [nameTemp] * len(trainNameLists)
#    trainDataMat, trainLabelArr = dataProcess.matSuffleByRow(trainDataMat, trainLabelArr)

#import pickle
#with open('../tmpdata2Train.bin','wb') as FIDO:
#    pickle.dump(trainDataMats,FIDO)
#with open('../tmpdata2Test.bin','wb') as FIDO:
#    pickle.dump(testDataMats,FIDO)    
#assert False
#nameTemp = testNameLists[0]    
if shuffleDataTest:
    if verbose:
        td.printC('Shuffling test datasets.','b')
    nameTemp = testNameLists[0].copy()    
    np.random.seed = seed
    np.random.shuffle(nameTemp)
    testDataMats, testLabelArrs, sortedIndexes = dataProcess.matAlignByName(testDataMats,nameTemp,testLabelArrs,testNameLists)
    testNameLists = [nameTemp] * len(testNameLists)
#    testDataMat, testLabelArr = dataProcess.matSuffleByRow(testDataMat, testLabelArr)

tmpTempLabel = trainLabelArrs[0]
for tmpLabel in trainLabelArrs:
    assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0

tmpTempLabel = testLabelArrs[0]
for tmpLabel in testLabelArrs:
    assert np.sum(np.array(tmpTempLabel) - np.array(tmpLabel)) == 0
    
if labelToMat:
    if verbose:
        td.printC('Since labelToMat is set, the labels would be changed to onehot-like matrix','b')
    trainLabelArr,trainLabelArrDict,trainArrLabelDict = dataProcess.labelToMat(trainLabelArrs[0])
    testLabelArr,testLabelArrDict,testArrLabelDict = dataProcess.labelToMat(testLabelArrs[0])
#    print(testLabelArr)
else:
    trainLabelArr = trainLabelArrs[0]
    testLabelArr = testLabelArrs[0]
    

    
if verbose:
#    print('Datasets generated')
    for i,trainDataMat in enumerate(trainDataMats):
        testDataMat = testDataMats[i]
        td.printC('The %dth scales are:\n\ttraining: %d x %d\n\ttest: %d x %d' %(i,trainDataMat.shape[0],trainDataMat.shape[1],testDataMat.shape[0],testDataMat.shape[1]), 'b')    
    td.printC('Begin to prepare model...','b')
    
#if not inputLength is None:
#    if inputLength == 0:
#        inputLength = trainDataMat.shape[1]

#%% model config

if verbose:
    td.printC('Checking the number of model files','b')
if len(modelLoadFile) < 1:
    if verbose:
        td.printC('please provide a model file in a python script or a json file. You can find some examples in the \'model\' folder', 'r')
assert not len(modelLoadFile) < 1

models = []
custom_objects_list = []
if len(weightLoadFile) > 0:
    if verbose:
        td.printC('Weight file provided by users, checking the number which should be the same as the model files.','b')
    assert len(weightLoadFile) == len(modelLoadFile)
    if verbose:
        td.printC('OK','g')
for i,subModelFile in enumerate(modelLoadFile):
    weightFile = None
    if len(weightLoadFile) > 0:
        weightFile = weightLoadFile[i]
    if subModelFile.endswith('.py'):
        model = moduleRead.readModelFromPyFileDirectly(subModelFile,weightFile=weightFile)
        tmpCustom = moduleRead.getCustomObjects(subModelFile)
        if not tmpCustom is None:
            custom_objects_list.append(tmpCustom)
    else:
        model = moduleRead.readModelFromJsonFileDirectly(subModelFile,weightFile=weightFile)
    for tmpmode in models:
        #some time the model will be the same in MEMORY! So I have to change it.
        if model is tmpmode:
            model = keras.models.clone_model(tmpmode)
            break
    models.append(model)
#print(custom_objects_list)
if len(custom_objects_list) > 0:
    custom_objects = mergeDict(custom_objects_list)
else:
    custom_objects = None



#input_length
if len(inputLength) < 1:
    inputLength = []
    for i,trainDataMat in enumerate( trainDataMats ):
        inputLength.append(trainDataMat.shape[1])    
#print(inputLength)
moduleRead.modifyInputLengths(models,inputLength,verbose=verbose,td=td,custom_objects=custom_objects)



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
    if len(models) == 1:
        firstKernelSize = tuple(np.array(firstKernelSizes).flatten())
        if verbose:
            td.printC('--firstKernelSize %s is provided, program will use it to change the layer if possible' %(str(firstKernelSize)),'b')
            td.printC('Note that --firstKernelSize will be abandoned in the next version, please modify the model file directly or using --reshapSizes instead','p')
        moduleRead.modifyFirstKenelSizeDirectly(models[0], firstKernelSize)
    else:
        for i,model in enumerate(models):
            firstKernelSize = tuple(firstKernelSizes[i])
            if verbose:
                td.printC('--firstKernelSize %s is provided, program will use it to change the layer if possible' %(str(firstKernelSize)),'b')
                td.printC('Note that --firstKernelSize will be abandoned in the next version, please modify the model file directly or using --reshapSizes instead','p')
            moduleRead.modifyFirstKenelSizeDirectly(model, firstKernelSize)

#merge model
if len(modelLoadFile) > 1:
    try:
        if verbose:
            td.printC('Multiple models detected, trying to merge them directly... ','b')
        model = moduleRead.modelMerge(models)
        if verbose:
            td.printC('Merging finished. ','g')
    except:
        if verbose:
            td.printC('Merging failed, trying adding reshape layer for the models... ','b')
        model = moduleRead.modelMergeByAddReshapLayer(models, dataMats=trainDataMats,label=trainLabelArr, reshapeSizes=reshapeSizes, verbose=verbose, td=td, custom_objects=custom_objects)
        if verbose:
            td.printC('Merging finished. ','g')
else:
    if verbose:
        td.printC('Only one model detected, will use it for training...','b')
    model = models[0]
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
            assert shapeProdNum == trainDataMats[0].shape[1]        
    except:        
#        if verbose:
#            td.printC('The input_shape %s of the first layer is not consistent with the datashape %s, a reshape layer will be added.' %(str(model.input.shape),str(trainDataMats[0].shape[1:])),'b')
        if reshapeSizes is None:
            currReshapeSize = None
        else:
            currReshapeSize = np.array(reshapeSizes).flatten()
        model = moduleRead.reshapeSingleModelLayer(model,trainDataMats[0],reshapeSize=currReshapeSize,verbose=verbose,td=td)
#        print(model.input)
moduleRead.modelCompile(model,loss = loss,optimizer = optimizer,metrics = metrics)

#%% training and predicting
if verbose:
    td.printC('Start training...','b')
#print(len(trainDataMats))
history = analysisPlot.LossHistory()
if len(trainDataMats) == 1:
    model.fit(trainDataMats[0], trainLabelArr,batch_size = batch_size,epochs = epochs,validation_split = 0.1,callbacks = [history])
else:
    model.fit(trainDataMats, trainLabelArr,batch_size = batch_size,epochs = epochs,validation_split = 0.1,callbacks = [history])
if verbose:
    td.printC('Training finished, generating the summary of the model','b')
    model.summary()

if not outSaveFolderPath is None:
    if not os.path.exists(outSaveFolderPath):        
        os.makedirs(outSaveFolderPath, exist_ok=True)
    else:
        if verbose:
            td.printC('outpath %s is exists, the outputs might be overwirten' %outSaveFolderPath,'p')
if not modelSaveName is None:
    tmpModelOutPath = outSaveFolderPath + os.path.sep + modelSaveName
    tmpWeightOutPath = None
    if not weightSaveName is None:
        tmpWeightOutPath = outSaveFolderPath + os.path.sep + weightSaveName
    if verbose:
        td.printC('\'modelSaveName\' provided, module will be saved at %s' %tmpModelOutPath,'g')
        if not tmpWeightOutPath is None:
            td.printC('Weights will be saved at %s' %tmpWeightOutPath,'g')
    moduleRead.saveBuiltModel(model, tmpModelOutPath, weightPath=tmpWeightOutPath)

if verbose:
    tmpFigSavePath = None
    if showFig:
        td.printC('Drawing figure recording loss...','b')    
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'epoch_loss.pdf'
        td.printC('Saving figure recording loss at %s' %tmpFigSavePath,'g')
    history.loss_plot('epoch',showFig=showFig,savePath=tmpFigSavePath)


predicted_Probability = model.predict(testDataMats)
if not 'predict_classes' in dir(model):
    prediction = np.rint(predicted_Probability)
    if labelToMat:
        prediction = dataProcess.matToLabel(np.array(prediction,dtype=int), testArrLabelDict,td=td)
else:
    prediction = model.predict_classes(testDataMats)
if labelToMat:
    testLabelArr = dataProcess.matToLabel(testLabelArr, testArrLabelDict)
else:
    testLabelArr = testLabelArrs[0]


#%% output analysis
td.printC('Showing the confusion matrix','b')
#print(testLabelArr)
#print(prediction)
cm=confusion_matrix(testLabelArr,prediction)
td.printC(str(cm),'B')
td.printC("ACC: %f "%accuracy_score(testLabelArr,prediction),'B')
if not labelToMat:
    td.printC("F1: %f "%f1_score(testLabelArr,prediction),'B')
    td.printC("Recall: %f "%recall_score(testLabelArr,prediction),'B')
    td.printC("Pre: %f "%precision_score(testLabelArr,prediction),'B')
    td.printC("MCC: %f "%matthews_corrcoef(testLabelArr,prediction),'B')
#    td.printC("AUC: %f "%roc_auc_score(testLabelArr,prediction),'B')

if savePrediction:
    tmpPredictSavePath = outSaveFolderPath + os.path.sep + 'predicts'
    if verbose:
        td.printC('Saving predictions at %s' %tmpPredictSavePath,'g')
    nameTemp = testNameLists[0]
    with open(tmpPredictSavePath, 'w') as FIDO:
        FIDO.write('#Name\tLabel\tPrediction\tProbability\n')
        for i in range(len(testLabelArr)):
            tmpLabel = testLabelArr[i]
            tmpPrediction = np.array(prediction[i]).flatten()
            if len(tmpPrediction) == 1:
                tmpPrediction = tmpPrediction[0]
            else:
                td.printC('irregular prediction detected %s' %(str(tmpPrediction)))
                tmpPrediction = tmpPrediction[0]
#            while len(tmpPrediction.shape) > 0:
#                tmpPrediction = tmpPrediction[0]
            tmpProbability = predicted_Probability[i]
#            tmpStr = '%r\t%r\t%r\n' %(tmpLabel,tmpPrediction,tmpProbability)
            if len(tmpProbability.shape) == 0:
                tmpStr = '%s\t%r\t%r\t%f\n' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability)
            else:
                if len(tmpProbability) == 1:
                    tmpStr = '%s\t%r\t%r\t%f\n' %(nameTemp[i],tmpLabel,tmpPrediction,tmpProbability[0])
                else:
                    tmpStr = '%s\t%r\t%r\t[ %s ]\n' %(nameTemp[i],tmpLabel,tmpPrediction,' , '.join(tmpProbability.astype(str)))
            FIDO.write(tmpStr)
    tmpCMPath = outSaveFolderPath + os.path.sep + 'performance'
    if verbose:
        td.printC('Saving confusion matrix and predicting performance at %s' %tmpCMPath,'g')
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
#            FIDO.write("AUC: %f \n"%roc_auc_score(testLabelArr,prediction))
if not labelToMat:
    tmpFigSavePath = None
    if showFig:  
        if verbose:
            td.printC('Plotting the ROC and PR curves...','b')
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'roc.pdf'
        if verbose:
            td.printC('Saving figure recording ROC curve at %s' %tmpFigSavePath,'g')        
    analysisPlot.plotROC(testLabelArr,predicted_Probability,showFig=showFig,savePath=tmpFigSavePath)
    if saveFig:
        tmpFigSavePath = outSaveFolderPath + os.path.sep + 'pr.pdf'
        if verbose:
            td.printC('Saving figure recording PR curve at %s' %tmpFigSavePath,'g')       
        analysisPlot.plotPR(testLabelArr,predicted_Probability,showFig=showFig,savePath=tmpFigSavePath)

if not paraSaveName is None:
    tmpParaSavePath = outSaveFolderPath + os.path.sep + paraSaveName
    if verbose:
        td.printC('Saving parameters at %s' %tmpParaSavePath, 'g')       
#    paraParser.saveParameters(tmpParaSavePath,sys.argv[1:])
    paraParser.saveParameters(tmpParaSavePath,paraDict)
td.printC('Finished','g')
