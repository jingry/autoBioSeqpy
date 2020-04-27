# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:59:33 2020

@author: jingr


"""

import os, sys
import re
import numpy as np

helpStr = '''

Usage: python findDiff.py predictionFile

Where the predictionFile is the output from 'predcting.py'

Find the differeces of probability for predicted the mutation
The sequences with the same head name will be compared (i.e. find the difference), the name here is:
    headname###subname
That is, if there is a separation '###' in the name, the left side is the head name, otherwise the full name is the headname.
The output will be:
    Name1 -- Name2  Index1 -- Index2    Prediction1 -- Prediction2  absProbabilityDiff
'''
if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
    print(helpStr)

class outObj:
    def __init__(self,name,label,predict,probability):
        self.name = name
        self.label = label
        self.predict = predict
        self.probability = probability
    
    def diffProbability(self, otherObj):
        return np.abs(self.probability - otherObj.probability)

class outGroup:
    def __init__(self):
        self.headObjDict = {}
        
    def addObj(self,objIn,sep='###'):
        headEle = re.split(sep,objIn.name)
        if len(headEle) > 1:
            head = sep.join(headEle[:-1])
        else:
            head = headEle[0]
        if not head in self.headObjDict:
            self.headObjDict[head] = []
        self.headObjDict[head].append(objIn)
        
    def printGroupDiff(self):
        print('Name1 -- Name2\tIndex1 -- Index2\tPredction1 -- Prediction2\tabsProbDIff')
        for head in self.headObjDict:
            groupLen = len(self.headObjDict[head])
            if groupLen == 1:
                continue
            for i in range(groupLen - 1):
                obj1 = self.headObjDict[head][i]
                for j in range(i+1,groupLen):                    
                    obj2 = self.headObjDict[head][j]
                    diff = obj1.diffProbability(obj2)
                    tmpstr = '%s -- %s\t%s -- %s\t%s -- %s\t' %(obj1.name,obj2.name,obj1.label,obj2.label,obj1.predict,obj2.predict)
                    if len(diff) == 1:
                        tmpstr += str(float(diff))
                    else:
                        tmpstr += '[' + ','.join(diff.astype(str)) + ']'
                    print(tmpstr)


tmpGroups = outGroup()
fileIn = sys.argv[1]
with open(fileIn) as FID:
    for line in FID:
        if line.startswith('#'):
            continue
        eles = line.strip().split()
        name = eles[0]
        label = eles[1]
        predict = eles[2]
        probability = np.array(eles[3:],dtype=float)
        tmpObj = outObj(name,label,predict,probability)
        tmpGroups.addObj(tmpObj)
tmpGroups.printGroupDiff()
        