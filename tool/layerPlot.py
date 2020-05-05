# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:09:28 2020

@author: jingr

Model plotting


need to do:
    flags (grid and single value, outFigFolder)
    test for single and multi models 
    help
    manual
    jupyter notebook
"""
import os, sys
sys.path.append('./libs')
sys.path.append('./tool/libs')
sys.path.append('../')



import os, sys, re
sys.path.append(os.path.curdir)
sys.path.append(sys.argv[0])

helpDoc = '''
The predicting script for using built model. 

Usage python predicting.py --dataTestFilePaths File1 File2 ... --paraFile filepath --predictionSavePath name [--verbose 0/1]

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
from utils import TextDecorate, evalStrList


import umap
import umap.plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
td = TextDecorate()


defaultParaDict, numSet, intSet, boolSet, objSet = paraParser.getDefaultParameters()
defaultParaDict['grid_n_neighbors'] = []
defaultParaDict['grid_min_dist'] = []
defaultParaDict['n_neighbors'] = 15
defaultParaDict['min_dist'] = 0.1
defaultParaDict['layerIndex'] = -2
defaultParaDict['figWidth'] = 800
defaultParaDict['figHeight'] = 800
defaultParaDict['outFigFolder'] = None
#numSet.add('grid_min_dist')
#intSet.add('grid_n_neighbors')
numSet.add('min_dist')
intSet.add('n_neighbors')
intSet.add('layerIndex')
intSet.add('figWidth')
intSet.add('figHeight')

paraDictCMD = paraParser.parseParameters(sys.argv[1:],defaultParaTuple=(defaultParaDict, numSet, intSet, boolSet, objSet))
paraFile = paraDictCMD['paraFile']
paraDict = paraParser.parseParametersFromFile(paraFile,defaultParaTuple=(paraDictCMD, numSet, intSet, boolSet, objSet))

#paraFile = paraDictCMD['paraFile']
#if not paraFile is None:
#    paraDict = paraParser.parseParametersFromFile(paraFile)
#    paraDict['dataTestFilePaths'] = paraDictCMD['dataTestFilePaths']
#    paraDict['dataTestModelInd'] = paraDictCMD['dataTestModelInd']
#else:
#    paraDict = paraDictCMD.copy()


#paraFile = 'D:/workspace/autoBioSeqpy/tmpOut/parameters.txt'
#paraDict = paraParser.parseParametersFromFile(paraFile)
#paraDict['dataTestFilePaths'] = paraDictCMD['dataTestFilePaths']
#paraDict['dataTestModelInd'] = paraDictCMD['dataTestModelInd']
paraDict['dataTestFilePaths'] = paraDict['dataTrainFilePaths']
paraDict['dataTestModelInd'] = paraDict['dataTrainModelInd']
paraDict['dataTestLabel'] = paraDict['dataTrainLabel']
#os.chdir('../')


#parameters

grid_n_neighbors = paraDict['grid_n_neighbors']
if len(grid_n_neighbors) > 0:
    grid_n_neighbors = evalStrList(grid_n_neighbors)
grid_min_dist = paraDict['grid_min_dist']
if len(grid_min_dist) > 0:
    grid_min_dist = evalStrList(grid_min_dist)
layerIndex = paraDict['layerIndex']
figWidth = paraDict['figWidth']
figHeight = paraDict['figHeight']
outFigPath = paraDict['outFigFolder']
if outFigPath is None:
    outFigPath = paraDict['outSaveFolderPath']

#dataType = paraDict['dataType']
dataTypeList = paraDict['dataType']
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

#useKMer = paraDict['useKMer']
#KMerNum = paraDict['KMerNum']
inputLength = paraDict['inputLength']
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
#modelPredictFile = 'D:/workspace/autoBioSeqpy/tmpOut/tmpMod.json'
#weightLoadFile = 'D:/workspace/autoBioSeqpy/tmpOut/tmpWeight.bin'

model = moduleRead.readModelFromJsonFileDirectly(modelPredictFile,weightLoadFile)
from keras.models import Model
from keras.models import Sequential
    
#modelPredictFile = outSaveFolderPath + os.path.sep + modelSaveName
#
#
#
#weightLoadFile = outSaveFolderPath + os.path.sep + weightSaveName


def unBoundLayers(modelIn,layers = []):
    for layer in modelIn.layers:
        if not 'sequential' in layer.name.lower():
            layers.append(layer)
        else:
            unBoundLayers(layer,layers)
    return layers

def generateNewModelFromLayers(layers):
    newModel = Sequential()
    for layer in layers:
        newModel.add(layer)
    return newModel

def genrerateNewModelFromModel(oriModel, selectedLayerIndex = -2, td = td):
    isConcatenate, hasSequential, avaiLayerIndex = analAvaiLayers(oriModel)
    
    layerIndexFix = selectedLayerIndex
    if layerIndexFix < 0:
        layerIndexFix = layerIndexFix + len(analAvaiLayers)
    if layerIndexFix < 0 or layerIndexFix > len(analAvaiLayers) - 1:
        td.printC('Only %d layers could be used to generating UMAP, but the index %d is out of the range.' %(len(analAvaiLayers),selectedLayerIndex),'r')
    if isConcatenate:
        newModel = Model(inputs=oriModel.input,outputs=oriModel.layers[avaiLayerIndex[selectedLayerIndex]].output)
    else:
        if hasSequential:
            upackedLayers = unBoundLayers(oriModel)
            newModel = generateNewModelFromLayers(upackedLayers[:selectedLayerIndex])
        else:
            newModel=Model(inputs=oriModel.input,outputs=oriModel.layers[avaiLayerIndex[selectedLayerIndex]].output)
    return newModel

def analAvaiLayers(modelIn):
    isConcatenate = False
    hasSequential = False
    lastConcatenateLayerNum = None
    for i,layer in enumerate(modelIn.layers):
        if 'concatenate' in layer.name.lower():
            isConcatenate = True
            lastConcatenateLayerNum = i
        if 'sequential' in layer.name.lower():
            hasSequential = True
    avaiLayerIndex = []
    if isConcatenate:
        avaiLayerIndex = range(lastConcatenateLayerNum,len(modelIn.layers))
    else:
        avaiLayerIndex = range(len(modelIn.layers))
    return isConcatenate, hasSequential, list(avaiLayerIndex)

def plotOneUMAP(outName, featureDict = None, pdf=None, td=td):
    if featureDict is None:
        mapper = umap.UMAP().fit(predicted_Probability)
    else:
        mapper = umap.UMAP(**featureDict).fit(predicted_Probability)
#    fig = plt.figure()
    plotObj = umap.plot.points(mapper, labels=testLabelArr)
#    plt.savefig('tmpOut/tmp.jpg')
    
#    print('Saving %s' %(subTitle))
    if pdf is None:        
        plt.savefig(outName)
        td.printC('%s saved.' %(outName), 'g')
    else:
        subTitle = re.findall('(NNeighbor.+)\.pdf',outName)[0]
        plt.title(subTitle)
#        pdf.savefig(fig)
        pdf.savefig()
        td.printC('%s plotted.' %(subTitle), 'b')

    plt.clf()
    plt.close('all')
    
    
    
    
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
    else:
        td.printC('Unknow dataType %r, please use \'protein\', \'dna\' ,\'rna\' or \'other\'' %subDataType, 'r')
    featureGenerators.append(featureGenerator)
    assert subDataType.lower() in ['protein','dna','rna','other']

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
    dataLoader = dataProcess.DataLoader(label = dataTestLabel[i], featureGenerator=featureGenerator)
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
model = moduleRead.readModelFromJsonFileDirectly(modelPredictFile,weightLoadFile)
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
'''
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
'''
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
if not outFigPath is None:
    if not os.path.exists(outFigPath):        
        os.makedirs(outFigPath, exist_ok=True)
#    else:
#        if verbose:
#            td.printC('outpath %s is exists, the outputs might be overwirten' %outSaveFolderPath,'p')    

#exclude reshape
#subModel = model.layers[1].layers[3]
#subModel.output

#tmpLayer = model.get_input_at(0)
#subModel=Model(inputs=model.input,outputs=model.layers[-2].output)
#subModel1 = Model(inputs=model.layers[1].input,outputs = subModel.layers[1].output)

#tmpModel =  Sequential()
#tmpModel.add(model.layers[0])
#tmpModel.add(model.layers[1].layers[0])
#tmpModel.add(model.layers[1].layers[1])
#tmpModel.add(model.layers[1].layers[2])
#tmpModel.add(model.layers[1].layers[3])
#tmpModel.add(model.layers[1].layers[4])
#tmpModel.add(model.layers[1].layers[5])
#tmpModel.add(model.layers[1].layers[6])
    

#upackedLayers = unBoundLayers(model)
#upackedModel = generateNewModelFromLayers(upackedLayers[:-4])
#upackedModel.summary()
oriPrediction = model.predict(testDataMats)
newModel = genrerateNewModelFromModel(model,selectedLayerIndex=layerIndex, td=td)
predicted_Probability = newModel.predict(testDataMats)

if len(grid_min_dist) > 0 or len(grid_n_neighbors) > 0:
    pdf = PdfPages('%s/UMAP.pdf' %outFigPath)
    
featureDict={
        'n_neighbors' : None,
        'min_dist' : None,
        }
#print(grid_n_neighbors)
if verbose:
    td.printC('Started to generating UMAP...', 'b')
if len(grid_min_dist) > 0 and len(grid_n_neighbors) > 0:
    for min_dist in np.arange(*grid_min_dist):
        featureDict['min_dist'] = min_dist
        for n_neighbors in np.arange(*grid_n_neighbors).astype(int):
            featureDict['n_neighbors'] = n_neighbors
            outName = '%s/UMAP_NNeighbor_%s_MDist_%s.pdf' %(outFigPath,str(n_neighbors),str(min_dist))
#            print(outName)
            plotOneUMAP(outName, featureDict = featureDict, pdf=pdf, td=td)
elif len(grid_min_dist) > 0:
    n_neighbors = paraDict['n_neighbors']
    featureDict['n_neighbors'] = n_neighbors
    for min_dist in np.arange(*grid_min_dist):
        featureDict['min_dist'] = min_dist
        outName = '%s/UMAP_NNeighbor_%s_MDist_%s.pdf' %(outFigPath,str(n_neighbors),str(min_dist))
        plotOneUMAP(outName, featureDict = featureDict, pdf=pdf, td=td)
elif len(grid_n_neighbors) > 0:
    min_dist = paraDict['min_dist']
    featureDict['min_dist'] = min_dist
    for n_neighbors in np.arange(*grid_n_neighbors).astype(int):
        featureDict['n_neighbors'] = n_neighbors
        outName = '%s/UMAP_NNeighbor_%s_MDist_%s.pdf' %(outFigPath,str(n_neighbors),str(min_dist))
        plotOneUMAP(outName, featureDict = featureDict, pdf=pdf, td=td)
else:
    n_neighbors = paraDict['n_neighbors']
    featureDict['n_neighbors'] = n_neighbors
    min_dist = paraDict['min_dist']
    featureDict['min_dist'] = min_dist
    outName = '%s/UMAP.pdf' %(outFigPath)
    plotOneUMAP(outName, featureDict = featureDict, td=td)
#mapper = umap.UMAP(**featureDict).fit(predicted_Probability)
#plt.figure()
#plotObj = umap.plot.points(mapper, labels=testLabelArr)
#plt.savefig('tmpOut/tmp.jpg')
#plotOneUMAP(featureDict = featureDict)


if len(grid_min_dist) > 0 or len(grid_n_neighbors) > 0:
    pdf.close()
    if verbose:
        td.printC('%s/UMAP.pdf saved.' %outFigPath, 'g')
