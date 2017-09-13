"""
Created on July 17th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
from random import *


class PathParser():

  def matrix_printer(self, line, outfile):
    """
      copies the original input file lines, and the pastes (replaces) the perturbed values in the output files
      In: line, outfile 
      Out: None 
    """
    line = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line)
    line = line.upper().split()
    #print line
    if line[0] in self.setOfPerturbedIsotopes:
      try :
        line[1] = str(self.listedQValuesDict.get(line[0]))
        #print line[1]
      except : 
        raise Exception('Error. Check if the the actinides perturbed QValues have existing Qvalues in the unperturbed library')
    #print line 
    if len(line) > 1:
      line[0] = "{0:<7s}".format(line[0])
      line[1] = "{0:<11s}".format(line[1])
      line = ''.join(line[0]+line[1]+"\n")
      outfile.writelines(line) 
    if re.search(r'(.*?)END', line[0]):
      self.stopFlag = self.stopFlag + 1
      self.harcodingSection = 0 
          
  def __init__(self, inputFiles, **pertDict):
    """
      takes the input qvalue decay files. changes values into scientific notations.  
      In: input files, perturbed dictionart 
      Out: None 
    """
    self.inputFiles = inputFiles
    self.pertQValuesDict = pertDict
    for key, value in self.pertQValuesDict.iteritems(): 
      self.pertQValuesDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
    #print self.inputFiles
    self.fileReconstruction() 
    
    
  def fileReconstruction(self):
    """
      Method convert the formatted dictionary pertdict -> {'DECAY|ALPHA|U235':1.30} 
      into a dictionary of dictionaries that has the format -> {'U235':{'ALPHA':1.30}}
      In: Dictionary pertDict 
      Out: Dictionary of dictionaries listedDict 
    """
    #print self.genericDecayDict
    #print self.pertDict
    self.listedQValuesDict = {}
    perturbedIsotopes = []
    for i in self.pertQValuesDict.iterkeys() :
      splittedDecayKeywords = i.split('|')
      perturbedIsotopes.append(splittedDecayKeywords[1])
    #print perturbedIsotopes 
    for i in xrange (0,len(perturbedIsotopes)):
      self.listedQValuesDict[perturbedIsotopes[i]] = {}   # declare all the dictionaries
    #print self.listedQValuesDict
    for isotopeKeyName, QValue in self.pertQValuesDict.iteritems():
      isotopeName = isotopeKeyName.split('|')
      #print isotopeName
      self.listedQValuesDict[isotopeName[1]] = QValue
    #print self.listedQValuesDict
    self.setOfPerturbedIsotopes = set(self.listedQValuesDict.iterkeys())
    #print self.setOfPerturbedIsotopes
    self.printInput()
  
  def removeRandomlyNamedFiles(self, modifiedFile):
    """
      Remove the temporary file with a random name in the working directory
      In, modifiedFile, string
      Out, None 
    """
    os.remove(modifiedFile)  
    
  def generateRandomName(self):
    """
      generate a random file name for the modified file
      @ in, None
      @ Out, string
    """
    return str(randint(1,1000000000000))+'.dat' 
    
    
  def printInput(self):
    """
      Method to print out the new input
      @ In, outfile, string, optional, output file root
      @ Out, None
    """
    modifiedFile = self.generateRandomName() 
    #print modifiedFile
    #print "\n"
    sectionCounter = 0
    self.stopFlag = 0 
    self.harcodingSection = 0 
    open(modifiedFile, 'w')
    with open(modifiedFile, 'a') as outfile:
      with open(self.inputFiles) as infile:   # count the number of times QValue sections occur
        for line in infile :
          if re.search(r'(.*?)(\s?)[a-zA-Z](\s+Qvalue)',line.strip()):
            #print line 
            sectionCounter = sectionCounter + 1 
            #print sectionCounter
          if not line.split(): continue       # if the line is blank, ignore it 
          if sectionCounter == 1 and self.stopFlag == 0:     #actinide section
            self.harcodingSection = 1 
            self.matrix_printer(line, outfile)
          if sectionCounter == 2 and self.stopFlag == 1:     #FP section
            self.harcodingSection = 2 
            self.matrix_printer(line, outfile)
          #print line 
          if self.harcodingSection != 1 and self.harcodingSection !=2:
            outfile.writelines(line)
    copyfile(modifiedFile, self.inputFiles)
    self.removeRandomlyNamedFiles(modifiedFile)

    

