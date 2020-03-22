# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:48:57 2019

@author: jingr

Parse parameters
"""

'''
dataEncodingType = 'dict'
spcLen = 100
dataTrainFilePaths = ['D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\train\\train_pos.txt', 
                 'D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\train\\train_neg.txt']
dataTrainLabel = [1,0]
dataTestFilePaths = ['D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\test\\test_pos.txt', 
                 'D:\\workspace\\proteinPredictionUsingDeepLearning\\original\\data\\test\\test_neg.txt']
dataTestLabel = [1,0]
modelFile = 'D:\\workspace\\proteinPredictionUsingDeepLearning\\models\\LSTM.py'
shuffleDataTrain = True
shuffleDataTest = True

batch_size = 40
epochs = 250

useKMer = False
KMerNum = 3

#inputLength = -1
inputLength = None

modelSavePath = None
'''
import re
import warnings

helpText = '''
Usage python running.py <--dataTrainFilePaths PATH1 PATH2 ...> <--dataTrainLabel label1 label2 ...> <--dataType protein/dna/rna> <--modelLoadFile modelPath> [options]
options:
    --dataType              {protein, dna, rna} 
                            No default value, should be provided by user
                            The type of the data, should be protein, dna or rna (upper case is supported either)
                            
    --dataEncodingType      list of {onehot, dict} 
                            Default: Dict
                            the type for encoding the data, if dict choosed, a character (e.g. A/G/C/T for DNA) is represented as a number (such as A:1 T:2 C:3 T:4), and if onehot choosed, a character will be represented as an array (such as A:[1,0,0,0] G:[0,1,0,0] C:[0,0,1,0] T[0,0,0,1])         
                            
    --spcLen                int 
                            Default: 100
                            The length of the input sequence which will be used for enconding. If the length of an input sequence is larger than the 'spcLen', the exceed part will be ignored, and if the length is less than 'spcLen', zeros (or zero arrays) will be added to make the length to 100.
                            
    --dataTrainFilePaths    list of paths
                            No default value, should be provided by user
                            The inputs are separated by space. FASTA data should be provided in separated files according to the labels, if two labels provided, there should be at least two FASTA files. For example, there are two files containing positive and negative samples separately, the inputs are:
                                --dataTrainFilePaths the/path/of/the/positive/file1.fasta the/path/of/the/negative/file2.fasta
    
    --dataTrainLabel        list of int
                            No default value, should be provided by user
                            The label of each file, and the length should be the same as --dataTrainFilePaths. As the example above, two FASTA file provided, so the label could be:
                                --dataTrainLabel 1 0
                        
    --dataTrainModelInd     list of int
                            No default value, should be provided by user
                            The index for the model of each file, and the length should be the same as --dataTrainFilePaths, and the values should be not larger than --modelLoadFile. As the example, if three FASTA files and two models (model_0, model_1 for example) provided, so the index could be:
                                --dataTrainModelInd 1 1 0
                            Here the '1 1 0' means the first two data will be train by model_1 and the 3rd model will be trained by model_0
                            
    --dataTestFilePaths     list of paths
                            No default value
                            Conflicting: --dataSplitScale
                            The data for independent test. The format and usage are the same as --dataTrainFilePaths.
                            NOTE: if no independent data provided, this parameter could be ignored, the dataset for testing will be generated from the training data by spliting it according to '--dataSplitScale'
                            
    --dataTestLabel         list of int
                            No default value
                            Conflicting: --dataSplitScale
                            The format is the same as --dataTrainLabel but for the test data. The length should be the same as --dataTestFilePaths
    
    --dataTestModelInd      list of int
                            No default value, should be provided by user
                            The index for the model of each test file if provided. The other explainations are the same with --dataTrainModelInd
                        
    --outSaveFolderPath     string
                            No default value
                            A folder path for saving the outputs, if not provide, only STDOUT will be generated.
                            
    --showFig               bool
                            Default: 1(True)
                            Switch to show the figures
                            
    --saveFig               bool
                            Default: 1(True)
                            Switch to save the figures to '--outSaveFolderPath'
                            
    --figDPI                int
                            Default:300
                            The dpi of the figure
                            
    --savePrediction        bool
                            Default: 1(True)
                            Switch to save the predictions to '--outSaveFolderPath'
                            
    --dataSplitScale        float (larger than 0 and less than 1)
                            No default value
                            Conflicting: --dataTestFilePaths, --dataTestLabel
                            A scale for spliting the training data into two piece, one is for training and the other for independent test.
                            For example, if the '--dataTestLabel' is 0.8, then the training data-set is 80% and the test data-set is 20% from the provided data.
                            
    --modelLoadFile         list of paths
                            No default value
                            Load the Keras model for modeling. Both user made model (in .py file) and keras model (in .json file) are supported. Few templates in python script (e.g. .py file) are provided in folder 'models'.
                            
    --weightLoadFile        string of path
                            No default value
                            Relating: --modelLoadFile
                            A built Keras model could save weight file as well, thus the weight file could be loaded when loading the model
                            
    --shuffleDataTrain       bool
                            Default: True
                            shuffle the sequence of training data
                            
    --shuffleDataTest        bool
                            Default: False
                            shuffle the sequence of test dataset. The default is False because the sequence will not change the modeling performance.
                            
    --batch_size            int
                            Default: 40
                            The parameter for keras to decide the size of batch (e.g. the number of used data) when training
                            
    --epochs                int
                            Default: 100
                            The parameter for keras to decide the number of iteration of training
                            
    --useKMer               bool
                            False
                            To considering the environment of a residue. For example, if a sequence is ATTACT, and '--KMerNum' is 3, then the first A will be considered as 'ATT' and the shape of dataset will be expanded accordingly (see the manual for more details).
                            
    --KMerNum               int
                            Default: 3
                            Relating: --userKMer
                            The length of the sequence which will be taken as environment, please see the details of '--UseKMer'
                            
    --inputLength           int
                            No default value
                            A parameter for 2D layer. This parameter is added to modify the size of the built model before compiling. The "batch_input_shape" and "input_length" will be changed according to this parameter. If not provided, program will change the size to the current shape automaticly if a 2D convolution layer is used as the first layer.
                            
    --firstKernelSize       int
                            No default value
                            A parameter for changing the kernel size of the first layer. Since the shape of input dataset might be not fit for the first layer, this parameter is added to modify the size of the built model before compiling. The "kernel_size" will be changed according to this parameter. If not provided, program will change the size to the current shape automaticly.
                            
    --loss                  string of loss function
                            Default: 'binary_crossentropy'
                            Keras parameter, available candidates are 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'
                            (reference https://keras.io/losses/)
                            
    --optimizer             string of optimizer
                            default: Adam
                            Keras parameters. Available candidates are SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, 
                            However, the optimizer have different parameter, if user want to modify it, please using the template instead of this script.
                            (reference https://keras.io/optimizers/)
                            
    --metrics               list of the metrics
                            Default: ['acc']
                            Keras parameters. Available candidates are 'acc', 'mae', 'binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy', 'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy'.
                            Note: The loss function is available here.
                            (reference https://keras.io/metrics/)
                            
    --modelSaveName         String of name
                            No default value
                            Save the built model in json format.
                            
    --weightSaveName        String of name
                            No default value
                            Save the weights of built model in binary format.
                            
    --noGPU                 bool
                            Only using CPU for modeling, sometimes is useful for debugging          
                            
    --paraFile              String of path
                            Sometimes using command line is not easy for use, write the parameters into file is better for modification.
                            The parameters in the paraFile is the same as writen in command line, such as '--noGPU 1 --figDPI 600 ...'
                            
    --paraSaveName          String of path
                            Save used parameters into file. Sometimes saving the parameters into a file will make the model easier for prediction.
                            
    --labelToMat            bool
                            Default: False
                            Change the label into matrix as follows: 
                                [0,1,2,1,1] =>  [1,0,0]
                                                [0,1,0]
                                                [0,0,1]
                                                [0,1,0]
                                                [0,1,0]
                            The change of the label could be useful for some kind of CNN with multilabel training.
    
    --seed                  int
                            Default: 1
                            The random seed of numpy.
                        
    --verbose               bool
                            Default: False
                            See a detailed output when the script running.
    
    --help                  See this document
'''

def showHelpDoc():
    print(helpText)
    
def getDefaultParameters():
    paraDict = {'dataType' : [],
                'dataEncodingType' : [],
                'spcLen' : [],
                'firstKernelSize' : [],
                'dataTrainFilePaths' : [],
                'dataTrainLabel' : [],
                'dataTestFilePaths' : [],
                'dataTestLabel' : [],
                'dataTrainModelInd' : [],
                'dataTestModelInd' : [],
                'outSaveFolderPath' : None,
                'showFig' : True,
                'saveFig' : True,
                'figDPI' : 300,
                'savePrediction' : True,
                'dataSplitScale' : None,
                'modelLoadFile' : [],
                'weightLoadFile' : [],
                'shuffleDataTrain' : True,
                'shuffleDataTest' : False,
                'batch_size' : 40,
                'epochs' : 100,
                'useKMer' : [],
                'KMerNum' : [],
                'inputLength' : [],
                'loss' : 'binary_crossentropy',
                'optimizer' : 'optimizers.Adam()',
                'metrics' : ['acc'],
                'modelSaveName' : None,
                'weightSaveName' : None,
                'noGPU' : False,
                'paraFile' : None,
                'paraSaveName' : None,
                'seed' : 1,
                'labelToMat' : False,
                'verbose' : True,
                }
    #'firstKernelSize' were exclued from intSet
    intSet = set(['seed','spcLen','batch_size','epochs','KMerNum','dataTrainLabel','dataTestLabel','dataTrainModelInd','dataTestModelInd','figDPI'])
    numSet = set(['dataSplitScale'])
    boolSet = set(['shuffleDataTrain','shuffleDataTest','useKMer','verbose','showFig','saveFig','savePrediction','noGPU','labelToMat'])
    objSet = set(['inputLength'])
    return paraDict, numSet, intSet, boolSet, objSet

    
def addSingleParameter(paraDict, numSet, intSet, boolSet, objSet, tmpName, tmpVal, modifiedSet):    
    if not tmpName in paraDict:
        return
    if re.sub('\s','',tmpVal) == 'None':
        tmpVal = None
    elif tmpName in intSet:
        tmpVal = int(tmpVal)
    elif tmpName in numSet:
        tmpVal = float(tmpVal)
    elif tmpName in boolSet:
        try:
            tmpVal = bool(int(tmpVal))
        except:
            tmpVal = eval(tmpVal)
    elif tmpName in objSet:
        tmpVal = eval(tmpVal)
    
    if tmpVal == 'None':
        tmpVal = None
    
    if isinstance(paraDict[tmpName],list) :        
        paraDict[tmpName].append( tmpVal)
        modifiedSet.add(tmpName)
    else:
        if tmpName in modifiedSet:
            warnings.warn('Parameter \'%s\' which is defined more than once and the new value \'%r\' will be ignored, please check your input.\n' %(tmpName,tmpVal), DeprecationWarning)
            return
        paraDict[tmpName] = tmpVal
        modifiedSet.add(tmpName)
    
    
def parseParameters(listIn):
    modifiedSet = set()
    paraDict, numSet, intSet, boolSet, objSet = getDefaultParameters()
    tmpName = None
    for i,subPara in enumerate(listIn):
        if subPara.startswith('-'):
            tmpName = subPara.replace('-','')
            if not tmpName in paraDict:
                warnings.warn('Unknow parameter \'%s\' which will not be used this time, please check your input.\n' %tmpName, DeprecationWarning)
        else:
            if not tmpName in paraDict:
                continue
            if not tmpName is None:
                if isinstance(paraDict[tmpName],list) :
                    tmpVals = subPara.split(',')
                    for tmpVal in tmpVals:
                        addSingleParameter(paraDict, numSet, intSet, boolSet, objSet, tmpName, tmpVal, modifiedSet)
                else:
                    tmpVal= subPara       
                    addSingleParameter(paraDict, numSet, intSet, boolSet, objSet, tmpName, tmpVal, modifiedSet)
        
    return paraDict
        
def printParameters(dictIn):
    print('parameters')
    defaultPara = getDefaultParameters()[0]
    for k in dictIn:
        v = dictIn[k]
        vDefault = defaultPara[k]
        tmpStr = '%s : %r' %(k,v)
        if v is vDefault:
            tmpStr += '  (as default or not used)'
        print(tmpStr)
        
def saveParameters(savePath,dictIn,sep=' '):
    with open(savePath,'w') as FIDO:
        for k in dictIn:
#            if dictIn[k] is None:
#                continue
            tmpStr = '--' + k + sep
            if isinstance(dictIn[k],list):
                tmpStr += str.join(sep,map(str,dictIn[k]))
            else:
                tmpStr += str(dictIn[k])
            FIDO.write(tmpStr+'\n')
        
def parseParametersFromFile(fileIn,sep=' '):
    paraIn = []
    with open(fileIn) as FID:
        for line in FID:
            if line.startswith('#'):
                continue
            paraIn += line.strip().split(sep)
    return parseParameters(paraIn)
                               