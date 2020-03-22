# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:43:27 2020

@author: jingr

Some utils for autoBioSeqpy.

Including:
Decorate the text with colors
Part of the codes from  https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
"""

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
class TextDecorate(Bcolors):
    def decorate(self, textIn, color):
        if color == 'r':
            return self.FAIL + textIn + self.ENDC
        elif color == 'g':
            return self.OKGREEN + textIn + self.ENDC
        elif color == 'b':
            return self.OKBLUE + textIn + self.ENDC
        elif color == 'p':
            return self.HEADER + textIn + self.ENDC
        elif color == 'B':
            return self.BOLD + textIn + self.ENDC
        elif color == 'U':
            return self.UNDERLINE + textIn + self.ENDC
        elif color == 'G':
            return self.ENDC + textIn + self.ENDC
        else:
            return textIn
    def printDecorate(self, textIn, color):
        print(self.decorate(textIn,color))
        
    def printD(self, textIn, color):
        self.printDecorate(textIn, color)
        
    def printC(self, textIn, color):
        self.printDecorate(textIn, color)    
        
if __name__ == "__main__":
    #print(bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.HEADER + "HEADER: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.OKBLUE + "OKBLUE: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.OKGREEN + "OKGREEN: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.WARNING + "WARNING: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.FAIL + "FAIL: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.ENDC + "ENDC: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.BOLD + "BOLD: No active frommets remain. Continue?" + bcolors.ENDC)
#    print(Bcolors.UNDERLINE + "UNDERLINE: No active frommets remain. Continue?" + bcolors.ENDC)
    textTemp = 'This is a test for text decoration'
    td = TextDecorate()
    td.printC(textTemp,'r')
    td.printC(textTemp,'g')
    td.printC(textTemp,'b')
    td.printC(textTemp,'p')
    td.printC(textTemp,'B')
    td.printC(textTemp,'U')
    td.printC(textTemp,'G')
