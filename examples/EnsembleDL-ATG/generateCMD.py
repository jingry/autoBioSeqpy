# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:35:51 2022

@author: tmp
"""

import os, sys
from itertools import combinations
positiveSeqFile = './examples/EnsembleDL-ATG/data/ATGtrpo393.txt'
negativeSeqFile = './examples/EnsembleDL-ATG/data/ATGtrne393.txt'

# positiveFeaFiles = []
# negativeFeaFiles = []
# dataPath = './data'
# for f in os.listdir(dataPath):
#     if f.startswith('po'):
#         positiveFeaFiles.append('%s/%s' %(dataPath,f))
#     elif f.startswith('ne'):
#         negativeFeaFiles.append('%s/%s' %(dataPath,f))

    
positiveFeaFiles = ['./examples/EnsembleDL-ATG/data/po-aac_pssm.txt', './examples/EnsembleDL-ATG/data/po-DFMCA_PSSM.txt',
                    './examples/EnsembleDL-ATG/data/po-dpc_pssm.txt', './examples/EnsembleDL-ATG/data/po-DP_PSSM.txt', 
                    './examples/EnsembleDL-ATG/data/po-pse_pssm.txt', './examples/EnsembleDL-ATG/data/po-pssm400.txt', 
                    './examples/EnsembleDL-ATG/data/po-pssm_ac.txt', './examples/EnsembleDL-ATG/data/po-single_Average.txt', 
                    './examples/EnsembleDL-ATG/data/po-SVD_PSSM.txt']

negativeFeaFiles = ['./examples/EnsembleDL-ATG/data/ne-aac_pssm.txt', './examples/EnsembleDL-ATG/data/ne-DFMCA_PSSM.txt',
                    './examples/EnsembleDL-ATG/data/ne-dpc_pssm.txt', './examples/EnsembleDL-ATG/data/ne-DP_PSSM.txt', 
                    './examples/EnsembleDL-ATG/data/ne-pse_pssm.txt', './examples/EnsembleDL-ATG/data/ne-pssm400.txt', 
                    './examples/EnsembleDL-ATG/data/ne-pssm_ac.txt', './examples/EnsembleDL-ATG/data/ne-single_Average.txt', 
                    './examples/EnsembleDL-ATG/data/ne-SVD_PSSM.txt']

# spcLenList = []

feaDict = {}
for i in range(len(positiveFeaFiles)):
    posFile = positiveFeaFiles[i]
    negFile = negativeFeaFiles[i]
    feaName = os.path.split(posFile)[-1].split('-')[-1].split('.')[0]
    modelName = './examples/EnsembleDL-ATG/model/%s.py' %feaName
    # assert os.path.exists(modelName)
    feaDict[feaName] = (posFile,negFile,modelName,200)

repeatTime = 5

cmdTemp = 'python running.py --dataType protein %s --dataEncodingType dict %s --dataTrainFilePaths %s --dataTrainLabel 1 0%s --dataSplitScale 0.8 --modelLoadFile examples/EnsembleDL-ATG/model/CNN.py %s --verbose 1 --showFig 0 --outSaveFolderPath %s --savePrediction 1 --saveFig 1 --batch_size 128 --epochs 20 --shuffleDataTrain 1 --spcLen 2000 %s --modelSaveName tmpMod.json --weightSaveName tmpWeight.bin --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.001,amsgrad=False,decay=False) --dataTrainModelInd 0 0%s'
errCMDList = []
feaNames = list(feaDict.keys())

for repeatNum in range(repeatTime):
    for combNum in range(9):
        combIterObj = combinations(feaNames, combNum + 1)
        
        for combIter in combIterObj:
            dataType = ''
            dataEncodingType = ''
            dataTrainFilePaths = positiveSeqFile+' '+negativeSeqFile #oriFile needed
            dataTrainLabel = ''
            modelLoadFile = ''
            outSaveFolderPath = 'outs%d/' %repeatNum
            spcLen = ''
            dataTrainModelInd = ''
            modelCount = 1
            
            for feaName in combIter:
                _posFile,_negFile,_modelName,_spcLen = feaDict[feaName]
                
                dataType += ' other'
                dataEncodingType += ' other'
                dataTrainFilePaths += ' ' + _posFile + ' ' + _negFile
                dataTrainLabel += ' 1 0'
                modelLoadFile += ' ' + _modelName
                outSaveFolderPath += feaName + '__'
                spcLen += ' %d' %_spcLen
                dataTrainModelInd += ' %d %d' %(modelCount,modelCount)
                modelCount += 1
            outSaveFolderPath = outSaveFolderPath[:-2]
            cmd = cmdTemp %(dataType, dataEncodingType, dataTrainFilePaths, dataTrainLabel, modelLoadFile, outSaveFolderPath, spcLen, dataTrainModelInd)
            print('#' * 10)
            print(cmd)
            isErr = os.system(cmd)
            if isErr:
                errCMDList.append('%d::%s' %(repeatNum,cmd))
print('#'*50)
print('err:')
for cmd in errCMDList:
    print('*'*10)
    print(cmd)
