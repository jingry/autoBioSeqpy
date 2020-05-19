# -*- coding: utf-8 -*-
"""
Created on Mon May 11 20:55:14 2020

@author: jingr
"""

import os, sys
sys.path.append('./libs')
sys.path.append('./tool/libs')
sys.path.append('../')



import os, sys, re
sys.path.append(os.path.curdir)
sys.path.append(sys.argv[0])
import numpy as np
import analysisPlot

#Name1 -- Name2\tIndex1 -- Index2\tPredction1 -- Prediction2\tprobDIff
fileIn = '../tmpOut/diffMat.txt'
subSep = ' -- '

oriIndex = 0

proteinStr = 'GAVLIPFYWSTCMNQDEKRHXBJOUZ'
dnaStr = 'AGCT'
rnaStr = 'AGCU'

#names = []
#indexes = []
#predicts = []
#probDiffs = []
#with open(fileIn) as FID:
#    rowList = []
#    rowOldInd = None
#    for line in FID:
#        if line.startswith('#'):
#            continue
#        eles = line.strip().split('\t')
#        name1, name2 = eles[0].split(subSep)
#        index1, index2 = eles[1].split(subSep)
#        index1 = int(index1)
#        index2 = int(index2)
#        out1, out2 = eles[2].split(subSep)
#        probDiff = float(eles[3])
#        names.append((name1,name2))
#        indexes.append((index1,index2))
#        predicts.append((out1,out2))
#        probDiffs.append(probDiff)
#maxInd = np.max(indexes) + 1
#outmat = np.zeros([maxInd,maxInd])
#for i in range(len(names)):
#    outmat[indexes[i][0],indexes[i][1]] = probDiffs[i]

mutationDict = {} # position:ori:mutation
keySet = set()
with open(fileIn) as FID:     
    rowList = []
    rowOldInd = None
    for line in FID:
        if line.startswith('#'):
            continue
        eles = line.strip().split('\t')
        name1, name2 = eles[0].split(subSep)
        index1, index2 = eles[1].split(subSep)
        index1 = int(index1)
        index2 = int(index2)
        out1, out2 = eles[2].split(subSep)
        probDiff = float(eles[3])
        if index1 == oriIndex:
            position, oriRes, mutatedRes = re.findall('(\d+)_(\w)-(\w)',name2.split('###')[1])[0]
            position = int(position)
            if not position in mutationDict:
                mutationDict[position] = {}
            if not oriRes in mutationDict[position]:
                mutationDict[position][oriRes] = {}
            mutationDict[position][oriRes][mutatedRes] = probDiff
            keySet.add(oriRes.upper())
            keySet.add(mutatedRes.upper())

if 'X' in keySet:
    keySet.remove('X')
if len(keySet) == 4:
    if 'U' in keySet:
        tempStr = rnaStr
    else:
        tempStr = dnaStr
else:
    tempStr = proteinStr

outMat = np.zeros([len(tempStr),len(list(mutationDict.keys()))])        
outStrDict = {'ori':'Seq:\t'}
for res in tempStr:
    outStrDict[res] = '%s:\t' %res
for position in np.sort(list(mutationDict.keys())):
    oriRes = list(mutationDict[position].keys())[0]
    outStrDict['ori'] += '%s\t' %oriRes
    for i,mutatedRes in enumerate(tempStr):
        if mutatedRes in mutationDict[position][oriRes]:
            outStrDict[mutatedRes] += '%.6e\t' %(mutationDict[position][oriRes][mutatedRes])
            outMat[i,position] = mutationDict[position][oriRes][mutatedRes]
        else:
            outStrDict[mutatedRes] += '0\t'
            outMat[i,position] = 0

#text output 
print(outStrDict['ori'])
for res in tempStr:
    print(outStrDict[res])
        
#figure output
analysisPlot.showMatWithVal(outMat)
        
        
        
        
        
        
        
        
        
        
        
        
        