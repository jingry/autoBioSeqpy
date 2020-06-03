# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:04:28 2020

@author: jingr

Modify the length of the sequences
"""

import os, sys, re
import numpy as np

def fastaFileRead(fileIn,dictIn):
    name = None
    with open(fileIn) as FID:
        for line in FID:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                name = line.strip()[1:]
                if name in dictIn:
                    print('duplicated fasta: %s' %name)
                else:
                    dictIn[name] = ''
            else:
                if name is None:
                    continue
                dictIn[name] += line.strip()
    
def fixOneSeq(seqIn,fixFrontScale,cutFrontScale,spcLen,paddingRes='X'):
    if len(seqIn) > spcLen:
        #cut
        exceedLen = len(seqIn) - spcLen
        frontLen = int(np.rint(float(exceedLen) * cutFrontScale))
        #lastLen = exceedLen - frontLen
        outSeq = seqIn[frontLen:frontLen+spcLen]
    elif len(seqIn) < spcLen:
#        print(len(seqIn) , spcLen)
        #add
        exceedLen = spcLen - len(seqIn)
        frontLen = int(np.rint(float(exceedLen) * fixFrontScale))
        lastLen = exceedLen - frontLen
        outSeq = ''
        outSeq += paddingRes * frontLen
        outSeq += seqIn
        outSeq += paddingRes * lastLen
#        print(outSeq)
    else:
        outSeq = seqIn
    return outSeq

def generateSlideWin(name,seq,winSize,dictIn,stride=1):
    count = 0
    currPos = 0
#    print(winSize,len(seq))
    if winSize >= len(seq):
        dictIn[name] = seq
        return
    while currPos + winSize <= len(seq):
        subSeq = seq[currPos:currPos + winSize]
        subName = name + '_%d' %count
        dictIn[subName] = subSeq
        count += 1
        currPos += stride
    #for the tail
    if currPos -  stride + winSize < len(seq):
        subSeq = seq[currPos:]
        subName = name + '_%d' %count
        dictIn[subName] = subSeq
        
def printOut(fileOut,dictIn,outLen = 80):
    with open(fileOut,'w') as FIDO:
        for k in dictIn:
            FIDO.write('>%s\n' %k)
            tmpStr = dictIn[k]
            while len(tmpStr) > 0:
                currStr = tmpStr[:outLen]
                tmpStr = tmpStr[outLen:]
                FIDO.write('%s\n' %currStr)
helpStr = '''
To modify the sequences to the same length. Usually there are two ways:
    1) using slide window
    2) cut/add character to make the length consistent
Here provide three commonly used actions:
    1) using slide window for a FASTA file:
            python tool/sequenceModify.py  --fileIn path/to/a/FASTA/file --fileOut path/to/a/text/file
        The command above will generate the slide window for all the fasta sequences in the input file according to the minimal length.
        
        If users want to specify the window size and the output length, the command becomes:
            python tool/sequenceModify.py  --fileIn path/to/a/FASTA/file --fileOut path/to/a/text/file --spcLen 60 --slideWinSize 60
        Then the sequences in the output file will have the length of 60, if the sequence doesn't have 60 residues/bases, the padding character 'X' will be added to the end of the such sequene to make the length to 60.
        
        Additionally, the position of the padding character could be set manually:
            python tool/sequenceModify.py  --fileIn path/to/a/FASTA/file --fileOut path/to/a/text/file --spcLen 60 --slideWinSize 60 --fixFrontScale 0.2
        The command above makes 20% of the padding character in front of the sequence and the rest behind.
        
    2) cut/add character to make the length consistent:
        This time the usage becomes a bit complex:
            python tool/sequenceModify.py  --fileIn path/to/a/FASTA/file --fileOut path/to/a/text/file --spcLen 60 --toSlide 0 --fixFrontScale 0.2 -cutFrontScale 0.3
        The command above makes the sequences either be cut to 60 residues/bases started at the position of 30% in the sequence, or add padding 20% character 'X' at the head and 80% and the end.
        
        
    The explaination of the parameters are listed below:
    --paddingRes            character
                            Default: 'X'
                            The padding charcter to fix the sequence.
                            
    --fixFrontScale         float ranges from 0 to 1
                            Default: 0
                            The scale to assign the padding charcter to the head of a sequence (the rest will to the tail). 
                            For example, set --fixFrontScale as 0.2, then 20% padding charcters (usually X) will be assign to the head of the sequence and the rest to the tail.
                            NOTE: Set --fixFrontScale to 0 means all the padding charcters to the tail (i.e. 0% : 100%), and set it to 1 means all the charcters to the head (100% : 0%).
                            
    --cutFrontScale         float ranges from 0 to 1
                            Default: 0
                            The scale to decide the start point to cut the sequence.
                            For example, assume the length of a sequence is 100 and larger than --spcLen which is set to 60, and --cutFrontScale is set as 0.3, then the start point will be the 13th (np.rint((100-60)*30%) + 1) residue/base, and thus 13th - 72th residues/bases will be conserved and others will be taken off.
                            NOTE: Set --cutFrontScale to 0 means start to the 1st residue/position and set to 1 means start from the last possible (i.e. seqLength - spclen + 1) residue/base.
                            
    --spcLen                int
                            Default: The minimal of the input sequences.
                            The specified length to be output.

    --toSlide               bool
                            Default: True
                            A switch to use the sliding window.

    --slideWinSize          int:
                            Default: The minimal of the input sequences.
                            The length of the sliding window size.
                            
     --stride               int:
                            Default: 1.
                            The length to stride when using the sliding window.                       

    --fileOut               Path of string
                            No default, should be provided by user
                            The file contain fasta sequences.

    --fileIn                Path of string
                            No default, should be provided by user
                            The file to record the outputs.           
'''

def main():

    
    
    if '-h' in sys.argv or '--help' in sys.argv:
        print(helpStr)
        exit()
    paraDict = {
            'paddingRes' : 'X',
            'fixFrontScale' : 0,
            'cutFrontScale' : 0,
            'spcLen' : None,
            'toSlide' : True,
            'slideWinSize' : None,
            'fileOut' : None,
            'fileIn' : None,
            'stride' : 1,
            }
    

    
    currPara = None
    for para in sys.argv:
        if para.startswith('--'):
            currPara = para[2:]
        else:
            if currPara is None:
                continue
            paraDict[currPara] = para
            
    paddingRes = paraDict['paddingRes']
    fixFrontScale = float(paraDict['fixFrontScale'])
    if fixFrontScale > 1:
        print('--fixFrontScale should be ranged in 0 to 1, but %f detected, will be set to 1.0')
    if fixFrontScale < 0:
        print('--fixFrontScale should be ranged in 0.0 to 1.0, but %f detected, will be set to 0.0')
    cutFrontScale = float(paraDict['cutFrontScale'])
    if cutFrontScale > 1:
        print('--cutFrontScale should be ranged in 0 to 1, but %f detected, will be set to 1.0')
    if cutFrontScale < 0:
        print('--cutFrontScale should be ranged in 0.0 to 1.0, but %f detected, will be set to 0.0')
    spcLen = paraDict['spcLen']
    if not spcLen is None:
        spcLen = int(spcLen)
    toSlide = paraDict['toSlide']
    if not toSlide is True:
        try:
            toSlide = eval(toSlide)
            assert toSlide is True or toSlide is False
        except:
            toSlide = bool(int(toSlide))
#            assert toSlide is True or toSlide is False
    slideWinSize = paraDict['slideWinSize']
    if not slideWinSize is None:
        slideWinSize = int(slideWinSize)
        
    fileIn = paraDict['fileIn']
    if fileIn is None:
        print('Please provide input files for parameter "--fileIn"')
        exit()
    stride = int(paraDict['stride'])
    
    fastaDict = {}
    fastaFileRead(fileIn,fastaDict)
    
    fileOut = paraDict['fileOut']
    if fileOut is None:
        print('Please provide input files for parameter "--fileOut"')
        exit()
        
    #for k in fastaDict:
    #    print(k,len(fastaDict[k]))
    if spcLen is None or slideWinSize is None:
        minLen = None
        for k in fastaDict:
            tmpLen = len(fastaDict[k])
#            print(k,tmpLen)
            if minLen is None:
                minLen = tmpLen
            else:
                minLen = np.min([tmpLen,minLen])
        if spcLen is None:
            print('Parameter "--spcLen" is not specificed, will find the minimal length (%d) of the secquence as the spcLen ' %minLen)
            spcLen = minLen
        if slideWinSize is None:
            print('Parameter "--slideWinSize" is not specificed, will find the minimal length (%d) of the secquence as the spcLen ' %minLen)
            slideWinSize = minLen
    
    
    if toSlide:
        slideDict = {}
        for k in fastaDict:
            generateSlideWin(k,fastaDict[k],slideWinSize,slideDict,stride=stride)
        outDict = {}
        for k in slideDict:
            tmpOut = fixOneSeq(slideDict[k],fixFrontScale,cutFrontScale,spcLen,paddingRes=paddingRes)
            outDict[k] = tmpOut
    else:
        outDict = {}
        for k in fastaDict:
            tmpOut = fixOneSeq(fastaDict[k],fixFrontScale,cutFrontScale,spcLen,paddingRes=paddingRes)
            outDict[k] = tmpOut
    printOut(fileOut,outDict)
#print(k,len(slideDict[k]),slideDict[k])
    
if __name__ == '__main__':
    main()