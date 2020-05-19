# -*- coding: utf-8 -*-
"""
Declaration:
    
This script used another open source module 'sequenceshuffles' from github https://github.com/dmitrip/SequenceShuffles for generating the shuffled seuqences.

"""

import os, sys
import numpy as np

sys.path.append('./libs')
sys.path.append('./tool/libs')
import sequenceshuffles as seqs
helpStr = '''
This script is to generate the shuffled sequences which keep the K-mer components.

Usage:
Considering that users in the root of autoBioSeqpy, then:
    python tool/generateShuffledSeqs.py --KMerNum 2 --outFile outSeqs.fasta --generateNum 10 --withOri 1 --oriName tmp --randomly 1 --oriSeq CATCGTGGCTAAACAGGTACTGCTGGGTAA
The command above is to generate 10 shuffled sequences from 'CATCGTGGCTAAACAGGTACTGCTGGGTAA', in the output file, the original sequence were kept and named tmp since --withOri and --oriName provided.
Also, for multiple sequences in a fasta file, users could using:
    python tool/generateShuffledSeqs.py --KMerNum 2 --outFile outSeqs.fasta --generateNum 5 --withOri 0 --randomly 1 --inFile inputFastaFile

Parameters:
    --KMerNum       int
                    Default: 2
                    The k-mer components the program will keep.
    --outFile       string
                    Default: None
                    The file to record the outputs.
    --inFile        string
                    Default: None
                    Conflicting: --oriSeq --oriName
                    The file which recorded multiple fasta sequences to be shuffled, if this value provided, --oriSeq and --oriName will not be used.
    --oriSeq        string
                    Default: None
                    Conflicting: --inFile
                    A fasta sequence to be shuffled, will be ignored if --infile is NOT None.
    --oriName       string
                    Default: tmpSeq
                    Conflicting: --inFile
                    Relating: --withOri --oriSeq
                    The name of --oriSeq when saving the original sequence, thus it will be used only if --withOri is True and --infile is None.     
    --withOri       bool
                    Default: False                    
                    To record the original sequence in the output file. If --oriSeq and --oriName used, the --oriName will be the name of the fasta, otherwise the name recorded in the fasta file will be used.
    --help          Print this document.
                
 
'''


def main():
    if '-h' in sys.argv or '--help' in sys.argv:
        print(helpStr)
        exit()
    paraDict = {
            'KMerNum' : 2,
            'outFile' : None,
            'oriSeq' : None,
            'oriName' : 'tmpSeq',
            'generateNum' : 2,
#            'withOri' : False,
            'randomly' : False,
            'bufferSize' : 2000,
            'dataType' : None,
            }
    
    proteinStr = 'GAVLIPFYWSTCMNQDEKRHXBJOUZ'
    dnaStr = 'AGCT'
    rnaStr = 'AGCU'
    
    currPara = None
    for para in sys.argv:
        if para.startswith('--'):
            currPara = para[2:]
        else:
            if currPara is None:
                continue
            paraDict[currPara] = para
    
    KMerNum = int(paraDict['KMerNum'])
    oriSeq = paraDict['oriSeq']
    oriSeq = oriSeq.upper()
    if oriSeq is None :
        print('Please provide --oriSeq ')
        exit()
    outFile = paraDict['outFile']
#    generateNum = paraDict['generateNum']
    try:
        withOri = bool(int(paraDict['withOri']))
    except:
        withOri = eval(paraDict['withOri'])
    
#    try:
#        randomly = bool(int(paraDict['randomly']))
#    except:
#        randomly = eval(paraDict['randomly'])
#    bufferSize = int(paraDict['bufferSize'])
    oriName = paraDict['oriName']
    
    if outFile is None:
        print('Please provide the --outFile')
    with open(outFile, 'w') as FIDO:
        FIDO.write('')
    
    dataType = paraDict['dataType']
    if dataType == 'protein':
        tempStr = proteinStr
    elif dataType == 'dna':
        tempStr = dnaStr
    elif dataType == 'rna':
        tempStr = rnaStr
    
    processOneSeq(oriSeq, withOri, outFile, oriName, KMerNum, tempStr)
#    else:
#        print('Since --inFile provided, --oriSeq and --oriName will not be used.')
#        with open(paraDict['inFile']) as FID:
#            oriName = None
#            for line in FID:
#                if line.startswith('#'):
#                    continue
#                if line.startswith('>'):
#                    if not oriName is None:
#                        shuffleOneSeq(oriSeq, outFile, generateNum, withOri, randomly, bufferSize, oriName, KMerNum)
#                    oriName = line.strip()[1:]
#                    print('Shuffling %s...' %oriName)
#                    oriSeq = ''
#                else:
#                    if oriName is None:
#                        continue
#                    oriSeq += line.strip()
#            #for last
#            shuffleOneSeq(oriSeq, outFile, generateNum, withOri, randomly, bufferSize, oriName, KMerNum)


def processOneSeq(oriSeq, withOri, outFile, oriName, KMerNum, tempStr):
#    try:
#        avaiNum = seqs.num_shuffles_from_string(oriSeq, k=KMerNum)
#        print('%.2E sequences available keeping %d-mer' %(avaiNum,KMerNum))
#    except:
#        try:
#            avaiNum = np.int(2**seqs.log_num_shuffles_from_string(oriSeq, k=KMerNum))
#            logAvaiNum = seqs.log_num_shuffles_from_string(oriSeq, k=KMerNum)
#            print('2^%d sequences available keeping %d-mer' %(logAvaiNum,KMerNum))
#        except:
#            avaiNum = sys.maxsize
#            print('more than %.2E sequences available keeping %d-mer' %(avaiNum,KMerNum))
#    if avaiNum == 1:
#        print('The sequence %s is unable to be shuffled in %d-mer' %(oriSeq,KMerNum))
#        return()
#    if generateNum == 'all':
#        generateNum = avaiNum
#        if generateNum > 500000:
#            print('The number of available sequences is larger than 500000, using "all" will be very slow, if you really want to generate a such number, please set a certain number instald of "all" for --generateNum.')
#            exit()
#    else:
#        generateNum = int(generateNum)
#        
#    generateNumOld = generateNum
#    
#    if generateNum > avaiNum:
#        print('%d sequences was ordered to be generated, but only %d are available.' %(generateNum,avaiNum))
#        generateNum = avaiNum
    
#    if bufferSize > avaiNum:
#        bufferSize = avaiNum
    with open(outFile, 'a') as FIDO:
        if withOri:
            FIDO.write('>%s\n' %oriName)
            FIDO.write('%s\n' %oriSeq)
        
#        numCount = 0
        for position, res in enumerate(oriSeq):
            for mutation in tempStr:
                if mutation == res:
                    continue
                newName = oriName + '###%d_%s-%s' %(position,res,mutation)
                newSeq = oriSeq[0:position] + mutation + oriSeq[position+1:]
                FIDO.write('>%s\n' %newName)
                FIDO.write('%s\n' %newSeq)
        

if __name__ == '__main__':
    main()