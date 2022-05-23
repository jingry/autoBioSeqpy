# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:40:27 2019

@author: jingr

data builder
"""

import numpy as np
import os

class FeatureGenerator:
    '''
    This class is a parent calss for deciding the way of FASTA sequence encoding type,
    to use this class please use 'ProteinFeatureGenerator', 'DNAFeatureGenerator' or
    'RNAFeatureGenerator' which inherit this class instead of using this parent class 
    directly.
    '''
    def __init__(self, encodingType, useKMer = False, KMerNum = 3):
        '''
        Function for init the class
        Parameter:
            encodingType:   'onehot' or 'dict', different encoding type could provide different tensor 
                            for modeling. 'dict' is providing a dictionary (hash table) for mapping the 
                            residues to numbers. 'onehot' is providing an array for a residue 
                            (e.g. [0,1,0,0] for T with enconding type [A,T,C,G]).
            useKMer:        bool (default: False), use kmer for encoding, which means use the environment 
                            (the left and right of a FASTA character) for representing a residue instead of 
                            a only character.
            KMerNum:        int (default: 3), the length of window when kmer used.
        '''
        self.encodingType = encodingType
        self.useKMer = useKMer
        self.KMerNum = KMerNum
        self.oneHotSeq = None
        self.oneHotIgnore = None
        self.oneHot1DDict = None
        self.word2int1DDict = None
        self.wordIndexDict = None

    
    def setIgnoreRes(self, seqIn):
        '''
        Setting the residue which will be ignored
        NOTE: this function is only for 'onehot' encoding
        
        Parameters:
            seqIn: string of FASTA, the FASTA characters are the residues to be ignored
        '''
        self.oneHotIgnore = seqIn
        self.regenerateWordIndexDict()
    
    def addIgnoreRes(self, strIn):
        '''
        Adding the residue which will be ignored
        NOTE: this function is only for 'onehot' encoding
        
        Parameters:
            strIn: string of FASTA, the FASTA characters are the residues to be ignored
        '''
        for tmpS in strIn:
            if not tmpS in self.oneHotIgnore:
                self.oneHotIgnore += tmpS
        self.regenerateWordIndexDict()
        
    def delIgnoreRes(self, strIn):
        '''
        Deleting the residue which will be ignored
        NOTE: this function is only for 'onehot' encoding
        
        Parameters:
            strIn: string of FASTA, the FASTA characters are the residues to be ignored
        '''
        for tmpS in strIn:
            if tmpS in self.oneHotIgnore:
                self.oneHotIgnore = self.oneHotIgnore.replace(tmpS,'')
        self.regenerateWordIndexDict()
        
    def regenerateWordIndexDict(self):
        '''
        Regenerating the sequence which will be used for enconding
        NOTE: this function is only for 'onehot' encoding
        '''
        self.wordIndexDict = {}
        i = 0
        for tmpRes in self.oneHotSeq:
            if tmpRes in self.oneHotIgnore:
                continue
            self.wordIndexDict[tmpRes] = i
            i += 1
    
    def dictEncoding1D(self,inpStr):
        '''
        Enconding the residues using 'dict'
        
        Parameters:
            inpStr: string, the sequence (usually FASTA)
        '''
        _res = []
        for inp in range(len(inpStr)):
            _res.append(self.word2int1DDict[inpStr[inp]])
        return _res
    
    def OnehotEncoding1D(self,inpStr):
        '''
        Enconding the residues using 'onehot'
        
        Parameters:
            inpStr: string, the sequence (usually FASTA)
        '''
        _res = []
        for base in inpStr:
            tmpOut = self.oneHot1DDict[base]
            _res.append(tmpOut)
        return _res
    
    def getKMerResPos(self, subSeq, baseLength, useIgnore = True):
        '''
        If KMer used, the array for representing the residue could be quite long.
        Therefore, this function is used for find the position of a residue piece.
        The rule is like the positional notation:
            If there are 20 residues will be uesd for representing, then the base
            is 20, and every residue will get a int number in [0,19]. If the length
            of the input sequence is 3, such as 'ATE', and related numbers are 3, 16 
            and 2, then the position is 3*20^2 + 16*20^1 + 2*20^0
        NOTE: This function is only for onehot encoding.
            
        Parameters:
            subSeq:         string of FASTA
            baseLength:     the length of base, usually the number of all the available residues
            useIgnore:      bool (default: True), ignore the special residue (self.oneHotIgnore) or not.
                            NOTE: if a residue included in the self.oneHotIgnore, the enconded array will 
                            be an array with zero.
        '''
        intPos = 0
        for ind,res in enumerate(subSeq):
            if useIgnore and (res in self.oneHotIgnore):
                return None
            intPos += self.wordIndexDict[res] * baseLength ** (self.KMerNum - 1 - ind)
        return intPos
        
    def dictEncodingMD(self,inpStr):
        '''
        Enconding the residues using 'dict' for KMer
        
        Parameters:
            inpStr: string, the sequence (usually FASTA)
        '''
        _res = []
        baseLength = len(self.oneHotSeq)
        itrLen = len(inpStr) - self.KMerNum + 1
        for resPos in range(itrLen):
            subSeq = inpStr[resPos:resPos+self.KMerNum]
            intPos = self.getKMerResPos(subSeq,baseLength,useIgnore = False)
            _res.append(intPos)
        return _res
    
    def OnehotEncodingMD(self,inpStr):
        '''
        Enconding the residues using 'onehot' for KMer
        
        Parameters:
            inpStr: string, the sequence (usually FASTA)
        '''
        baseLength = len(self.oneHotSeq) - len(self.oneHotIgnore)
        arrLength = baseLength ** self.KMerNum
        baseArr = np.zeros([1,arrLength])
        _res = []
        itrLen = len(inpStr) - self.KMerNum + 1
        for resPos in range(itrLen):
            subSeq = inpStr[resPos:resPos+self.KMerNum]
            intPos = self.getKMerResPos(subSeq,baseLength)
            if intPos is None:
                _res.append(baseArr)
            else:
                tmpArr = np.zeros_like(baseArr)
                tmpArr[0,intPos] += 1
                _res.append(tmpArr)
        return _res
    
    # def smilesStructureParser(self,seq):
    #     nodeSeq = []
    #     adjMat = None
        
    
    def seqEncoding(self,seq):
        '''
        Enconding the residues, the encoding type is decided by the inner parameters.
        
        Parameters:
            seq: string, the sequence (usually FASTA)
        '''
#        dataMat = []
        seqLen = len(seq)
        seqData = None
        feaLen = None
        if self.useKMer:
            #kmer
#            print('using kmer')
            if self.encodingType == "dict":
                feaLen = seqLen - self.KMerNum + 1
                seqData = np.array(self.dictEncodingMD(seq)).reshape([1,feaLen])
            else:
                feaLen = (seqLen - self.KMerNum + 1) * ((len(self.oneHotSeq) - len(self.oneHotIgnore)) ** self.KMerNum)
                seqData = np.array(self.OnehotEncodingMD(seq)).reshape([1,feaLen])
        else:
            if self.encodingType == "dict":
                feaLen = seqLen
                seqData = np.array(self.dictEncoding1D(seq)).reshape([1,feaLen])
            else:
                feaLen = seqLen * (len(self.oneHotSeq) - len(self.oneHotIgnore)) 
                seqData = np.array(self.OnehotEncoding1D(seq)).reshape([1,feaLen])
                
        
        
#        dataMat.append(seqData)
#        if encodingType == "dict":
#            return np.array(dataMat).reshape(len(self.names),seqLen),label
#        else:
#            return np.array(dataMat).reshape(len(self.names),num),label
        return seqData, feaLen

class ProteinFeatureGenerator(FeatureGenerator):
    '''
    This class is for deciding the way of protein sequence encoding type.    
    Most of the functions are provided in its parent class 'FeatureGenerator',
    thus here this class is only provide some necessary data.
    '''
    def __init__(self, encodingType, useKMer = False, KMerNum = 3):
        FeatureGenerator.__init__(self,encodingType=encodingType, useKMer = useKMer, KMerNum = KMerNum)
        self.oneHotSeq = 'GAVLIPFYWSTCMNQDEKRHXBJOUZ'
        self.oneHotIgnore = 'XBJOUZ'
        self.oneHot1DDict = {
                "G" : [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "A" : [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "V" : [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "L" : [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "I" : [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "P" : [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "F" : [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "Y" : [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                "W" : [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                "S" : [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                "T" : [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                "C" : [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                "M" : [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                "N" : [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                "Q" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                "D" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                "E" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                "K" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                "R" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                "H" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                "X" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "B" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "J" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "O" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "U" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "Z" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                }
#        self.word2int1DDict = {'G': 1, 'A': 2, 'V': 3, 'L': 4, 'I': 5, 'P': 6,
#                               'F': 7, 'Y': 8, 'W': 9, 'S': 10, 'T': 11, 'C': 12,
#                               'M': 13, 'N': 14, 'Q': 15, 'D': 16, 'E': 17,
#                               'K': 18, 'R': 19, 'H': 20, 'X': 21, 'B': 22,
#                               'J': 23, 'O': 24, 'U': 25, 'Z': 26}
        self.word2int1DDict = {'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5,
                               'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
                               'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16,
                               'K': 17, 'R': 18, 'H': 19, 'X': 20, 'B': 21,
                               'J': 22, 'O': 23, 'U': 24, 'Z': 25}
#        self.wordIndexDict = {'G': 1, 'A': 2, 'V': 3, 'L': 4, 'I': 5, 'P': 6,
#                               'F': 7, 'Y': 8, 'W': 9, 'S': 10, 'T': 11, 'C': 12,
#                               'M': 13, 'N': 14, 'Q': 15, 'D': 16, 'E': 17,
#                               'K': 18, 'R': 19, 'H': 20, 'X': 21, 'B': 22,
#                               'J': 23, 'O': 24, 'U': 25, 'Z': 26}
        self.wordIndexDict = {'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5,
                              'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
                              'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16,
                              'K': 17, 'R': 18, 'H': 19, 'X': 20, 'B': 21,
                              'J': 22, 'O': 23, 'U': 24, 'Z': 25}
        self.generatorType = 'Protein'
    
class DNAFeatureGenerator(FeatureGenerator):
    '''
    This class is for deciding the way of DNA sequence encoding type.
    Most of the functions are provided in its parent class 'FeatureGenerator',
    thus here this class is only provide some necessary data.
    '''
    def __init__(self, encodingType, useKMer = False, KMerNum = 3):
        FeatureGenerator.__init__(self, encodingType, useKMer = useKMer, KMerNum = KMerNum)
        self.oneHotSeq = 'AGCTX'
        self.oneHotIgnore = 'X'
        self.oneHot1DDict = {
                "A" : [1,0,0,0],
                "G" : [0,1,0,0],
                "C" : [0,0,1,0],
                "T" : [0,0,0,1],
                "X" : [0,0,0,0]
                }
        self.word2int1DDict = {'A':0,'G':1,'C':2,'T':3,'X':4}
        self.wordIndexDict = {'A':0,'G':1,'C':2,'T':3,'X':4}
        self.generatorType = 'DNA'

class RNAFeatureGenerator(FeatureGenerator):
    '''
    This class is for deciding the way of RNA sequence encoding type.
    Most of the functions are provided in its parent class 'FeatureGenerator',
    thus here this class is only provide some necessary data.    
    '''
    def __init__(self, encodingType, useKMer = False, KMerNum = 3):
        FeatureGenerator.__init__(self, encodingType, useKMer = useKMer, KMerNum = KMerNum)
        self.oneHotSeq = 'AGCUX'
        self.oneHotIgnore = 'X'
        self.oneHot1DDict = {
                "A" : [1,0,0,0],
                "G" : [0,1,0,0],
                "C" : [0,0,1,0],
                "U" : [0,0,0,1],
                "X" : [0,0,0,0]
                }
        self.word2int1DDict = {'A':0,'G':1,'C':2,'U':3,'X':4}
        self.wordIndexDict = {'A':0,'G':1,'C':2,'U':3,'X':4}
        self.generatorType = 'RNA'

class SmilesFeatureGenerator(FeatureGenerator):
    '''
    This class is for deciding the way of smiles sequence encoding type.
    As a new class, this class is designed for both basic NLP type and graphic type    
    '''
    def __init__(self, encodingType, useKMer = False, KMerNum = 3):
        FeatureGenerator.__init__(self, encodingType, useKMer = useKMer, KMerNum = KMerNum)
        self.oneHotSeq = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#)(+-/1325476890=@[]\.'
        self.oneHotIgnore = ''
        self.oneHot1DDict = {'a': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'd': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'e': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'f': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'g': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'h': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'i': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'j': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'n': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'o': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'p': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             's': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             't': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'u': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'v': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'w': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'x': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'A': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'B': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'C': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'D': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'E': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'F': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'G': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'H': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'J': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '#': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             ')': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '(': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '+': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '/': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '7': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '6': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             '8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                             '=': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                             '@': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             '[': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             ']': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                             '\\': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                             '.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
        self.word2int1DDict = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 
                               'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 
                               'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 
                               'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, '#': 53, ')': 54, '(': 55, '+': 56, '-': 57, '/': 58, '1': 59, '3': 60, '2': 61, 
                               '5': 62, '4': 63, '7': 64, '6': 65, '8': 66, '9': 67, '0': 68, '=': 69, '@': 70, '[': 71, ']': 72, '\\': 73, '.': 74}
        
        self.wordIndexDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 
                              'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'A': 26, 'B': 27, 'C': 28, 'D': 29, 'E': 30,
                              'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45,
                              'U': 46, 'V': 47, 'W': 48, 'X': 49, 'Y': 50, 'Z': 51, '#': 52, ')': 53, '(': 54, '+': 55, '-': 56, '/': 57, '1': 58, '3': 59, '2': 60, 
                              '5': 61, '4': 62, '7': 63, '6': 64, '8': 65, '9': 66, '0': 67, '=': 68, '@': 69, '[': 70, ']': 71, '\\': 72, '.': 73}
        
        self.generatorType = 'Smiles'
        
    def structureEncoding(self):
        'TODO: parsing the structure into graph'
        
        
class OtherFeatureGenerator():
    def __init__(self, encodingType=None, useKMer = None, KMerNum = None):
        """
        The 'useKMer' and 'KMerNum' will not be used, only provided for the format of feature genrator
        
        This generator will not used for encoding, just used to tell DataLoader not to encode the text
        """
        self.generatorType = 'Other'
        
class DataLoader:
    """
    For loading data    
    Example:
        positiveDL = DataLoader(label = 1)
        positiveDL.readFile(inputFilePath, spcLen = 100)
        positiveDataMat,positiveLabel = positiveDL.seqEncoding(encodingType = 'oneHot')
    """
    def __init__(self, label, featureGenerator):
        """
        parameters:
            label: int or str
            featureGenerator: the feature generator, currently only ProteinFeatureGenerator, DNAFeatureGenerator and RNAFeatureGenerator are available.
        """
        
        self.names = []
        self.seqs = {}
        self.annotation = {}
        self.label = label #0,1,2... or True/False
        self.featureGenerator = featureGenerator
        self.arrDict = {} #{name:dataArr}

        
    def readFastaFile(self,inpFile,spcLen):
        """
        Read a FASTA file and store the sequences
        
        Parameters:
                inpFile: string of the FASTA file
                spcLen: the length for array transfer, for instance, if set this value to 100, 
                        then at most first 100 residues will be used, if the length is less than 100, 
                        few 'X' will be added at the end.
        """
        fileabsName = os.path.split(inpFile)[-1]
        for line in open(inpFile):
            if line.startswith('#'):
                continue 
            if line.startswith('>'):
#               name = line.replace('>','').split()[0]
                name = fileabsName + '_' + line.replace('>','').split()[0]
                annotation = line.strip()
                self.names.append(name)
                self.seqs[name] = ''
                self.annotation[name] = annotation
            else:
                self.seqs[name] += line.replace('\n','')
    
        for name in self.names:
            if len(self.seqs[name]) <= spcLen:
                self.seqs[name] = self.seqs[name] + "X" * (spcLen - len(self.seqs[name]))
            else:
                self.seqs[name] = self.seqs[name][0:spcLen]
               
   
    def readMatFile(self,inpFile,sep=','):
        """
        Read a Matrix file and return a dict
        
        Parameters:
                inpFile: string of the file name
        """
        for line in open(inpFile):
            if line.startswith('#'):
               continue 
            eles = line.strip().split(sep)
            name = eles[0]
            dataArr = np.array(eles[1:],dtype=float)
            self.names.append(name)
            self.arrDict[name] = dataArr
            
    def readSmileFile(self,inpFile,spcLen):
        fileabsName = os.path.split(inpFile)[-1]
        for line in open(inpFile):
            if line.startswith('#'):
                continue 
            if line.startswith('>'):
#               name = line.replace('>','').split()[0]
                name = fileabsName + '_' + line.replace('>','').split()[0]
                annotation = line.strip()
                self.names.append(name)
                self.seqs[name] = ''
                self.annotation[name] = annotation
            else:
                self.seqs[name] += line.replace('\n','')
        self.spcLen = spcLen
        # for name in self.names:
        #     if len(self.seqs[name]) <= spcLen:
        #         self.seqs[name] = self.seqs[name] + "X" * (spcLen - len(self.seqs[name]))
        #     else:
        #         self.seqs[name] = self.seqs[name][0:spcLen]
        
        
    def readFile(self,inpFile,**kwargs):
        if self.featureGenerator.generatorType == 'Other':
            del kwargs['spcLen']
            self.readMatFile(inpFile,**kwargs)
        else:
            self.readFastaFile(inpFile,**kwargs)
            
    def shuffle(self,seed=1):
        """
        shuffle the sort of different sequences.
        
        Parameters:
            seed: the seed for random number generator. In this function, np.random is used as the generator.
        """
        if not np.random.seed == seed:
            np.random.seed = seed
        np.random.shuffle(self.names)
        
    def getSeqList(self):
        '''
        get the list of sequence name
        '''
        outSeq = []
        for name in self.names:
            outSeq.append(self.seqs[name])
        return outSeq
    
    def getOutDict(self):
        """
        get a dictionary which contain the FASTA sequences
        """
        return self.seqs
    
    def seqEncoding(self):
        """
        Encoding the FASTA sequence into array
        """
        label = np.array([self.label]*len(self.names))
        dataMat = []
        for name in self.names:    
            seq = self.seqs[name]
            seqData, feaLen = self.featureGenerator.seqEncoding(seq )
            dataMat.append(seqData)
        return np.array(dataMat).reshape(len(self.names),feaLen),label    
          
    def matGenerate(self):
        label = np.array([self.label]*len(self.names))
        dataMat = []
        for name in self.names:
            dataArr = self.arrDict[name]
            dataMat.append(dataArr)
#        dataMat = None
#        for name in self.names:
#            dataArr = self.arrDict[name]
#            if dataMat is None:
#                dataMat = dataArr.reshape([1,len(dataArr)])
#            else:
#                dataMat = np.concatenate([dataMat,dataArr.reshape([1,len(dataArr)])],axis=0)
        return np.array(dataMat),label 
#        return dataMat,label 
    
    def returnDataMat(self):
        if not self.featureGenerator.generatorType == 'Other':
            return self.seqEncoding()
        else:
            return self.matGenerate()
        
class DataSetCreator():
    """
    Create dataset by using multiple dataLoader
    Example:
        positiveDL = DataLoader(label = 1)
        positiveDL.readFile(inputFilePath1, spcLen = 100)
        negativeDL = DataLoader(label = 0)
        negativeDL.readFile(inputFilePath0, spcLen = 100)
        dataSC = DataSetCreator((positiveDL, negativeDL))
        trainDataMat, testDataMat, trainLabel, testLabel = dataSC.getTrainTestSet(trainScale=0.8, encodingType = 'oneHot')
    """
    def __init__(self,dataLoaders):
        """
        The dataLoaders here are the objects from the class DataLoader above.
        """
        self.dataLoaders = dataLoaders
        
    def shuffle(self,seed=1):
        """
        Call the shuffle in class DataLoador
        """
        for DL in self.dataLoaders:
            DL.shuffle(seed=seed)
    
    def getDataSet(self, toShuffle=True, seed=1, withNameList = False):
        """
        Get the datasets without spliting, outputs are the datamat (seqs x residues) and the labels respectively.
        
        Parameters:
            toShuffle: bool, shuffle the rows (seqs) of the datamat
            seed: int, the seed of the random generator.
            withNameList: bool, return the list of names. This parameter is for aligning the samples from different DataSetCreator
        """
        dataMat = None
        label = None
        names = []
        if toShuffle:
            self.shuffle(seed=seed)
        for DL in self.dataLoaders:
            if withNameList:
                names += DL.names
#           tmpDataMat, tmpLabel = DL.seqEncoding()
            tmpDataMat, tmpLabel = DL.returnDataMat()
            if dataMat is None:
                dataMat = tmpDataMat
                label = tmpLabel
            else:
                dataMat = np.r_[dataMat,tmpDataMat]
                label = np.concatenate((label,tmpLabel),axis=0).flatten()

        if withNameList:
            return dataMat, label, names
        else:
            return dataMat, label
    
    def getNameList(self):
        '''
        Abandon
        '''
        names = []
        for DL in self.dataLoaders:
            names += DL.names
        return names
        
    def getTrainTestSet(self, trainScale, toShuffle=True, seed=1, withNameList=False):
        """
        Get the datasets with spliting, outputs are the datamat (seqs x residues) and the labels respectively.
        This function is used for generating the training and testing dataset when user didn't provide previously.
        
        Parameters:
            trainScale: float in (0,1), a scale for spliting, for example, 
                        if set this value to 0.8, then 80% of the sequence will 
                        be used as training dataset, the rest 20% for test dataset.
            toShuffle: bool, shuffle the rows (seqs) of the datamat
            seed: the seed of the random generator.
        """
        trainDataMat = None
        testDataMat = None
        trainLabel = None
        testLabel = None
        namesTrain = []
        namesTest = []
        if toShuffle:
            self.shuffle(seed=seed)
        for DL in self.dataLoaders:
#            tmpDataMat, tmpLabel = DL.seqEncoding()
            tmpDataMat, tmpLabel = DL.returnDataMat()
            sampleNum = tmpDataMat.shape[0]
            tmpTrain = tmpDataMat[:int(sampleNum*trainScale),:]
            tmpTest = tmpDataMat[int(sampleNum*trainScale):,:]
            tmpTrainL = tmpLabel[:int(sampleNum*trainScale)]
            tmpTestL = tmpLabel[int(sampleNum*trainScale):]
            if trainDataMat is None:
                trainDataMat = tmpTrain
                testDataMat = tmpTest
                trainLabel = tmpTrainL
                testLabel = tmpTestL
            else:
                trainDataMat = np.r_[trainDataMat,tmpTrain]
                testDataMat = np.r_[testDataMat,tmpTest]
                trainLabel = np.concatenate((trainLabel,tmpTrainL),axis=0).flatten()
                testLabel = np.concatenate((testLabel,tmpTestL),axis=0).flatten()
            if withNameList:
                namesTrain += DL.names[:int(sampleNum*trainScale)]
                namesTest += DL.names[int(sampleNum*trainScale):]
        if withNameList:
            return trainDataMat, testDataMat, trainLabel, testLabel, namesTrain, namesTest
        return trainDataMat, testDataMat, trainLabel, testLabel

def matSuffleByRow(matIn, label = None, seed = 1):
    """
    Suffle matrix 
        [0,1,2,1,1] =>  [1,0,0]
                        [0,1,0]
                        [0,0,1]
                        [0,1,0]
                        [0,1,0]
    Parameters:     
        labelIn: List, the list of the labels
    """
    indexArr = np.arange(len(matIn))
    if not np.random.seed == seed:
        np.random.seed = seed
    np.random.shuffle(indexArr)
    if label is None:
        return matIn[indexArr,:]
    else:
        return matIn[indexArr,:],label[indexArr]
    
    
def labelToMat(labelIn):
    """
    Change the label to matrix as the following:
        [0,1,2,1,1] =>  [1,0,0]
                        [0,1,0]
                        [0,0,1]
                        [0,1,0]
                        [0,1,0]
    Parameters:     
        labelIn: List, the list of the labels
    """
    labelSet = set(labelIn)
    arrLength = len(labelSet)
    baseArr = [0] * arrLength
    labelArrDict = {}
    arrLabelDict = {}
    for i, label in enumerate(np.sort(list(labelSet))):
        tmpArr = baseArr.copy()
        tmpArr[i] += 1
        labelArrDict[label] = tmpArr
        arrLabelDict[tuple(tmpArr)] = label
    labelOut = []
    for label in labelIn:
        labelOut.append(labelArrDict[label])
    return np.array(labelOut),labelArrDict,arrLabelDict

def matToLabel(labelIn,arrLabelDict,td=None):
    """
    Change the label back to list as the following:
        [1,0,0] =>  [0,1,2,1,1]
        [0,1,0]
        [0,0,1]
        [0,1,0]
        [0,1,0]
    Parameters:     
        labelIn: List of tuple, generated by function labelToMat
    """
    labelOut = []
    for arr in labelIn:
        k = tuple(arr)
        if k in arrLabelDict:
            labelOut.append(arrLabelDict[tuple(arr)])
        else:
            if td:
                td.printC('Irregular label detected and will be changed to \'0\', which usually means the training is not enough','p')
            labelOut.append(0)
    return labelOut

def matAlignByName(mats,nameTemp,labels,names,checkNameLength = True):

#    nameTemp = names[0]
    nameArgArgIndex = np.argsort(np.argsort(nameTemp))
    matsOut = []
    labelsOut = []
#    matsOut.append(mats[0])
#    labelsOut.append(np.array(labels[0]))
    indexes = []
#    indexes.append(np.array(range(len(nameTemp))))
    
    if checkNameLength:
        nameTempSet = set(nameTemp)
        for tmpName in names:
            tmpSet = set(tmpName)
            assert nameTempSet == tmpSet
    for i in range(0,len(names)):
        tmpName = names[i]
        tmpArgIndex = np.argsort(tmpName)
        tmpIndex = tmpArgIndex[nameArgArgIndex]
        matsOut.append(mats[i][tmpIndex])
        labelsOut.append(np.array(labels[i])[tmpIndex])
        indexes.append(tmpIndex)
    return matsOut, labelsOut, indexes
        

def splitMatByScaleAndIndex(scale, matIn, label, indexArr, nameList = None, seed=1):
#    if toShuffle:
#        if not np.random.seed == seed:
#            np.random.seed = seed
#    indexArr = np.arange(len(label),dtype=int)
#    if toShuffle:
#        np.random.shuffle(indexArr)
    sampleNum = len(label)

    matOut = matIn.copy()[indexArr,:]
    trainDataMat = matOut[:int(sampleNum*scale),:]
    testDataMat = matOut[int(sampleNum*scale):,:]

    labelOut = np.array(label)[indexArr]
    trainLabel = labelOut[:int(sampleNum*scale)]
    testLabel = labelOut[int(sampleNum*scale):]
    if not nameList is None:
        nameListOut = np.array(nameList)[indexArr]
        namesTrain = nameListOut[:int(sampleNum*scale)]
        namesTest = nameListOut[int(sampleNum*scale):]
        return trainDataMat, testDataMat, trainLabel, testLabel, namesTrain, namesTest
    return trainDataMat, testDataMat, trainLabel, testLabel
    


  
#for debug
#tmpName1 = ['a','b','c']    
#tmpName2 = ['c','a','b']    
#tmpMat1 = np.array([[1,1],[2,2],[3,3]])
#tmpMat2 = np.array([[6,6],[4,4],[5,5]])
#tmpLabel1 = [1,1,0]
#tmpLabel2 = [1,1,0]
#matsOut, labelsOut, indexes, nameTemp = matAlignByName([tmpMat1,tmpMat2],[tmpLabel1,tmpLabel2],[tmpName1,tmpName2])
     


     
        
        

     
     
