"""
Created on June 19th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
from random import *


class QValuesParser():

  def matrix_printer(self, infile, outfile):
  
    for line in infile :
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line[0])
      #print line 
      #print self.listedQValuesDict
      for isotopeID in self.listedQValuesDict.iterkeys():
        if line[0] == isotopeID :
          #print isotopeID  
          try :
            #print self.listedQValuesDict.get(isotopeID)
            line[1] = str(self.listedQValuesDict.get(isotopeID))
          except : 
            raise Exception('Error. Check if the the actinides perturbed QValues have existing Qvalues in the unperturbed library')
      try :
        if len(line) > 1:
          #print line 
          line[0] = "{0:<7s}".format(line[0])
          line[1] = "{0:<7s}".format(line[1])
          outfile.writelines(' '+line[0]+line[1]+"\n") 
      except KeyError: 
        pass

  def hardcopy_printer(self, input, modifiedFile):
    
    #print modifiedFile
    with open(modifiedFile, 'a') as outfile:
      with open(input) as infile:
        for line in infile:
          if not line.split(): continue   # if the line is blank, ignore it 
          #print line 
          if re.match(r'(.*?)\s+\w+\s+\d+.\d+',line) : 
            #print line             
            break
          outfile.writelines(line)
        self.matrix_printer(infile, outfile)

  def modifyValues(self, **Kwargs):
    """
      This method needs to access to the dictionary self.isotopeDecay
      and modify the values based on the input dictionary dictOfValues
      In: isotopeDecay 
      Out: 
    """
    self.typeOfDecays         = []
    self.genericDecayDict     = {}
    
    self.pertQValuesDict  = {'QVALUES|U235':180, 'QVALUES|U238':1.08E+02, 'QVALUES|CF252':1.08}
    for key, value in self.pertQValuesDict.iteritems(): 
      self.pertQValuesDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
    #print Kwargs.get('SampledVars', {})

          
  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS Decay data file and put the isotopes name as key and 
      the decay constant relative to the isotopes as values  
      In: decay.dat
      Out: self.decay (dictionary)
    """
 
    self.inputFiles = inputFiles
    self.pertQValuesDict = pertDict
    for key, value in self.pertQValuesDict.iteritems(): 
      self.pertQValuesDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
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
    open(modifiedFile, 'w')
    self.hardcopy_printer(self.inputFiles, modifiedFile)
    copyfile(modifiedFile, self.inputFiles)
    self.removeRandomlyNamedFiles(modifiedFile)

