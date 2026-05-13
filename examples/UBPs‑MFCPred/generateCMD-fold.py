# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:35:51 2022

@author: tmp
"""

import os, sys
from itertools import combinations
positiveSeqTrFile = './examples/UBPs‑MFCPred/data/CV/fold0/potr.txt'
negativeSeqTrFile = './examples/UBPs‑MFCPred/data/CV/fold0/potr.txt'

positiveSeqTeFile = './examples/UBPs‑MFCPred/data/CV/fold0/pote.txt'
negativeSeqTeFile = './examples/UBPs‑MFCPred/data/CV/fold0/pote.txt'

# positiveFeaFiles = []
# negativeFeaFiles = []
# dataPath = './data'
# for f in os.listdir(dataPath):
#     if f.startswith('po'):
#         positiveFeaFiles.append('%s/%s' %(dataPath,f))
#     elif f.startswith('ne'):
#         negativeFeaFiles.append('%s/%s' %(dataPath,f))

    
positiveFeaTrFiles = ['./examples/UBPs‑MFCPred/data/CV/fold0/potr_CTriad.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_DC.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_Ankh.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_ESM1b.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_ESM2.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_PAAComp.txt','./examples/UBPs‑MFCPred/data/CV/fold0/potr_ProtT5.txt']

negativeFeaTrFiles = ['./examples/UBPs‑MFCPred/data/CV/fold0/netr_CTriad.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_DC.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_Ankh.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_ESM1b.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_ESM2.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_PAAComp.txt','./examples/UBPs‑MFCPred/data/CV/fold0/netr_ProtT5.txt']

positiveFeaTeFiles = ['./examples/UBPs‑MFCPred/data/CV/fold0/pote_CTriad.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_DC.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_Ankh.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_ESM1b.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_ESM2.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_PAAComp.txt','./examples/UBPs‑MFCPred/data/CV/fold0/pote_ProtT5.txt']

negativeFeaTeFiles = ['./examples/UBPs‑MFCPred/data/CV/fold0/nete_CTriad.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_DC.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_Ankh.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_ESM1b.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_ESM2.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_PAAComp.txt','./examples/UBPs‑MFCPred/data/CV/fold0/nete_ProtT5.txt']

# spcLenList = []

feaDict = {}
for i in range(len(positiveFeaTrFiles)):
    posTrFile = positiveFeaTrFiles[i]
    negTrFile = negativeFeaTrFiles[i]
    posTeFile = positiveFeaTeFiles[i]
    negTeFile = negativeFeaTeFiles[i]
    feaName = os.path.split(posTrFile)[-1].split('_')[-1].split('.')[0]
    modelName = './examples/UBPs‑MFCPred/model/%s.py' %feaName
    # assert os.path.exists(modelName)
    feaDict[feaName] = (posTrFile,negTrFile,posTeFile,negTeFile,modelName,200)

repeatTime = 3

cmdTemp = 'python running.py --dataType protein %s --dataEncodingType onehot %s --dataTrainFilePaths %s --dataTrainLabel 1 0%s --dataTestFilePaths %s --dataTestLabel 1 0%s --modelLoadFile examples/UBPs‑MFCPred/model/CNN.py %s --verbose 1 --showFig 0 --outSaveFolderPath %s --savePrediction 1 --saveFig 1 --batch_size 32 --epochs 20 --shuffleDataTrain 1 --spcLen 50 %s --noGPU 0 --paraSaveName parameters.txt --optimizer optimizers.Adam(lr=0.0001,amsgrad=False,decay=False) --dataTrainModelInd 0 0%s --dataTestModelInd 0 0%s'
errCMDList = []
feaNames = list(feaDict.keys())

for repeatNum in range(repeatTime):
    #need to change this number manually, should not larger than the number of the features
    for combNum in range(11):
        combIterObj = combinations(feaNames, combNum + 1)
        
        for combIter in combIterObj:
            dataType = ''
            dataEncodingType = ''
            
            dataTrainFilePaths = positiveSeqTrFile+' '+negativeSeqTrFile #oriFile needed
            dataTrainLabel = ''

            dataTestFilePaths = positiveSeqTeFile+' '+negativeSeqTeFile #oriFile needed
            dataTestLabel = ''
            
            modelLoadFile = ''
            outSaveFolderPath = 'tmpOut%d/' %repeatNum
            spcLen = ''
            dataTrainModelInd = ''
            dataTestModelInd = ''
            modelCount = 1
            
            for feaName in combIter:
                _posTrFile,_negTrFile,_posTeFile,_negTeFile,_modelName,_spcLen = feaDict[feaName]
                
                dataType += ' other'
                dataEncodingType += ' other'
                
                dataTrainFilePaths += ' ' + _posTrFile + ' ' + _negTrFile
                dataTrainLabel += ' 1 0'

                dataTestFilePaths += ' ' + _posTeFile + ' ' + _negTeFile
                dataTestLabel += ' 1 0'
                
                modelLoadFile += ' ' + _modelName
                outSaveFolderPath += feaName + '__'
                spcLen += ' %d' %_spcLen
                
                dataTrainModelInd += ' %d %d' %(modelCount,modelCount)
                dataTestModelInd += ' %d %d' %(modelCount,modelCount)
                
                modelCount += 1
            outSaveFolderPath = outSaveFolderPath[:-2]
            cmd = cmdTemp %(dataType, dataEncodingType, dataTrainFilePaths, dataTrainLabel, dataTestFilePaths, dataTestLabel, modelLoadFile, outSaveFolderPath, spcLen, dataTrainModelInd, dataTestModelInd)
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
