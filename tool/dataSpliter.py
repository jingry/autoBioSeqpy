#/usr/bin/python

import os, sys
import numpy as np


helpStr = '''This script is to shrink a dataset and then split it into folds for cross-validation. Currently, this script can split fasta and matrix data, if a FASTA and matrix file have the same id, this tool could split the dataset following the id. That is, two (or more) datasets can have the same split index. 
Usage:
Suppose that there are 4 files, two are FASTA and others are matrix, the following command can shrink the dataset into 60% and then make a 5-fold split. The first 2 files have the same IDs and so do the last 2 files:

python tool/dataSpliter.py --dataFilePath file1 file2 file3 file4 --dataGroupLabel 0 0 1 1 --foldNum 5 --shrinkScale 0.6 --outFolder /path/to/outputs --dataFileType fasta
other fasta other

    
Parameters:
    --dataFilePath      list of path
                        No default provided, should be provided by user.
                        The input files, the format should be FASTA or matrix (seperated by ',' and the first column should be the IDs)
                        
    --dataGroupLabel    list of int
                        No default provided, should be provided by user.
                        The label of the group. In a group, every file should contain the same IDs.
                        
    --dataFileType      list of 'fasta' or 'other'
                        No default provided, should be provided by user.
                        The type of the files, 'fasta' for FASTA format and 'other' for matrix.
                        
    --foldNum           int
                        No default provided, should be provided by user.
                        The number of fold, which means the number the dataset will be splited.
                        
    --shrinkScale       float
                        Default: None
                        The scale the dataset will be shrinked. If the value less than 1, the number will be regard as a ratio (e.g. 0.6 for 60% samples), otherwise it will be regard as the number of samples (e.g. 200 for shrinked to 200 samples)
                        
    --outFolder         string of path
                        No default provided, should be provided by user.                   
                        The path for outputs, since there will be multiple files, thus a folder is necessary, the output files will be with the format 'fileOriName_%fscale_%dfold_train/test'.
    
    --help              Print this document.
'''
def main():
    if '-h' in sys.argv or '--help' in sys.argv:
        print(helpStr)
        exit()
    paraDict = {
            'dataFilePath' : [],
            'dataGroupLabel' : [],
            'dataFileType' : [],
            'foldNum' : None,
            'shrinkScale' : None,
            'outFolder' : None
            }
    currPara = None
    for para in sys.argv:
        if para.startswith('--'):
            currPara = para[2:]
        else:
            if currPara is None:
                continue
            if isinstance(paraDict[currPara],list):
                paraDict[currPara].append(para)
            else:
                paraDict[currPara] = para
                
    dataFilePath = paraDict['dataFilePath']
    dataGroupLabel = paraDict['dataGroupLabel']
    foldNum = int(paraDict['foldNum'])
    try:
        shrinkScale = float(paraDict['shrinkScale'])
    except:
        shrinkScale = None
    outFolder = paraDict['outFolder']
    dataFileType = paraDict['dataFileType']
    
    uniqueIDs = []
    if len(dataGroupLabel) > 1:
        uniqueIDs = set(dataGroupLabel)
    if len(uniqueIDs) < 2:
        fileLists = [dataFilePath]
        fileTypes = [dataFileType]
    else:
        fileLists = []
        fileTypes = []
        for refLabel in uniqueIDs:
            currFile = []
            currType = []
            for i in range(len(dataFilePath)):
                tmpLabel = dataGroupLabel[i]
                tmpFile = dataFilePath[i]
                if len(dataFileType) > 0:
                    tmpType = dataFileType[i]
                if tmpLabel == refLabel:
                    currFile.append(tmpFile)
                    if len(dataFileType) > 0:
                        currType.append(tmpType)
            fileLists.append(currFile)
            fileTypes.append(currType)
    for i in range(len(fileLists)):
        fileList = fileLists[i]
        fileType = fileTypes[i]
        doOneLabel(fileList, outFolder, foldNum, shrinkScale,fileType)

def doOneLabel(fileList, outFolder, foldNum, shrinkScale,fileType):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder, exist_ok=True)
    KfoldInd = None
    for i,f in enumerate(fileList):
        fName = os.path.split(f)[1]
        dataType = None
        if len(fileType) == 0:
            if fName.endswith('.fasta'):
                dataType = 'fasta'
            else:
                dataType = 'other'
        else:
            dataType = fileType[i].lower()
        header = None
        if dataType == 'fasta':
            dataDict = fileDictFasta(f)
        else:
            dataDict,header = fileDictOther(f)
        if KfoldInd is None:
            if not shrinkScale is None:
                dataDict = dataShrink(dataDict,shrinkScale)
            KfoldInd = getKFoldIndex(foldNum,list(dataDict.keys()))
        for k,ttInd in enumerate(KfoldInd):
            if shrinkScale is None:
                ss = 1
            else:
                ss = shrinkScale
            outName = outFolder + os.sep + fName
            outName += '_scale%r' %ss
            outName += '_fold%d_train' %k
            generateDataByName(outName,dataDict,KfoldInd[k][0],header=header)
            outName = outFolder + os.sep + fName
            outName += '_scale%r' %ss
            outName += '_fold%d_test' %k
            generateDataByName(outName,dataDict,KfoldInd[k][1],header=header)
        
        
        
        
def fileDictFasta(fileIn):
    outDict = {}
    k = None
    v = ''
    with open(fileIn) as FID:
        for line in FID:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                if not k is None:
                    outDict[k] = v
                    v = ''
                k = line[1:].strip()
                v = line
            else:
                v += line
    outDict[k] = v
    return outDict

def fileDictOther(fileIn, sep=','):
    outDict = {}
    k = None
    v = ''
    header = ''
    with open(fileIn) as FID:
        for line in FID:
            if line.startswith('#'):
                header += line
                continue
            eles = line.strip().split(sep)
            k = eles[0]
#            v = np.array(eles[1:],dtype=float)
            v = line
            outDict[k] = v
    return outDict,header

def dataShrink(dataDict,shrinkScale):
    kList = list(dataDict.keys())
    ind = np.arange(len(kList))
    np.random.shuffle(ind)
    if shrinkScale <= 1:
        outLen = int(len(kList) * shrinkScale)
    else:
        outLen = int(shrinkScale)
    outDict = {}
    for i in range(outLen):
        tmpId = kList[ind[i]]
        outDict[tmpId] = dataDict[tmpId]
    return outDict
    
def getKFoldIndex(foldNum, nameList, toSort = True):
    tmpList = np.array(nameList,dtype=str)
    tmpSet = set(nameList)
    np.random.shuffle(tmpList)
    fullLength = len(tmpList)
    foldLength = int(fullLength / foldNum)
    outList = []
    for i in range(foldNum):
        if i < foldNum - 1:
            testIndex = tmpList[i * foldLength: (i+1)*foldLength]
        else:
            testIndex = tmpList[i * foldLength: ]
        trainIndex = tmpSet - set(testIndex)
        if toSort:
            outList.append((np.sort(np.array(list(trainIndex),dtype=str)), np.sort(testIndex)))
        else:
            outList.append((np.array(list(trainIndex),dtype=str),testIndex))
    return outList
    
def generateDataByName(outName, dataDict, nameList, header=None):
    with open(outName,'w') as FIDO:
        if not header is None:
            FIDO.write(header)
        for name in nameList:
#            outLine = dataDict[name] + '\n'
            FIDO.write(dataDict[name])
    
#tmpDict = fileDictFasta(r'D:\workspace\autoBioSeqpy\examples\T3T4\dataset\trainT3.txt')
#tmpDict1,tmpHeader = fileDictOther(r'D:\workspace\autoBioSeqpy\examples\T3T4\feature\trainT3_AAC_PSSM_DC.txt')
#tmpDict2 = dataShrink(tmpDict,shrinkScale)
#tmpKfoldInd = getKFoldIndex(5,list(tmpDict1.keys()))
#
#generateDataByName('tmp.txt',tmpDict,tmpKfoldInd[0][0])
#generateDataByName('tmp1.txt',tmpDict1,tmpKfoldInd[0][0],header=tmpHeader)

if __name__ == '__main__':
    main()






