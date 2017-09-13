"""
Created on June 25th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
from random import *

class FissionYieldParser():
     
  def matrix_printer(self, infile, outfile, spectra):
  
    isotopeCounter = 0
    #print spectra     
    for line in infile :
      line = line.strip()  
      if not line: continue   # if the line is blank, ignore it 
      #print line 
      line = line.upper().split()
      line[0] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4',line[0])
      spectraUpper = spectra.upper()
      #print self.listedYieldDict
      try : 
        for fissionProductID in self.listedYieldDict[spectraUpper].iterkeys():
          for actinideID in self.listedYieldDict[spectraUpper][fissionProductID].iterkeys():
            #  #print  self.listedYieldDict
            #print self.listedYieldDict
            if line[0] == fissionProductID :
              #print fissionProductID
              typeOfYieldPerturbed = []
              self.spectrumUpperCase = [x.upper() for x in self.spectrum]
              typeOfYieldPerturbed = self.listedYieldDict.get(spectraUpper).get(fissionProductID).keys() 
              #print typeOfYieldPerturbed[0] 
              for i in xrange (0, len(typeOfYieldPerturbed)):
                #print len(typeOfYieldPerturbed)
                #print typeOfYieldPerturbed
                #try :
                #if spectra == self.listedYieldDict.keys()[0]:
                try :
                  if self.listedYieldDict.get(spectraUpper).get(fissionProductID).get(typeOfYieldPerturbed[i]) != {} :
                    #print self.listedYieldDict.get(spectraUpper).get(fissionProductID).get(typeOfYieldPerturbed[i])
                    #print typeOfYieldPerturbed[i]
                    #print self.spectrumNumbering
                    line[self.spectrumNumbering.get(spectra).get(typeOfYieldPerturbed[i])] = str(self.listedYieldDict.get(spectraUpper).get(fissionProductID).get(typeOfYieldPerturbed[i]))
                    #print line 
                except TypeError: 
                  raise Exception ('Make sure the fission yields you are perturbing have existing values in the unperturbed fission yield library')
      except KeyError:
        pass
      try : 
        isotopeCounter = isotopeCounter + 1  
        #print line 
        line[0] = "{0:<7s}".format(line[0])
        i = 1
        #print self.spectrum[0]
        while i <= len(self.spectrumNumbering.get(spectra)) :
          #print i 
          #print line 
          line[i] = "{0:<11s}".format(line[i])
          i = i + 1 
        outfile.writelines(' '+''.join(line[0:len(self.spectrumNumbering.get(spectra)) + 1])+"\n") 
        #print isotopeCounter 
        if isotopeCounter == self.numberOfIsotopes :
          for lineInput in infile :
            lineStripped = lineInput.strip()
      except KeyError: 
        raise Exception ('Make sure the fission yields you are perturbing have existing values in the unperturbed fission yield library')
     
     
  def hardcopy_printer(self, input, spectra, modifiedFile):
  
    #print input
    #print spectra    
    flag =  0
    with open(modifiedFile, 'a') as outfile:
      with open(input) as infile:
        for line in infile:
          #print line 
          #print line.strip()
          if re.match(r'(.*?)END\s+\w+',line.strip()) and spectra == self.spectrum[1]:
            flag = 2
          #  print flag 
          if flag == 2 :
            if re.match(r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w',line.strip()) and spectra == self.spectrum[1] : 
              #print line
              outfile.writelines(line)              
              break
            outfile.writelines(line)
          if (re.match(r'(.*?)'+spectra,line.strip()) and spectra == self.spectrum[0]):
            flag = 1 
            #print flag 
          if  flag == 1 :
            if re.match(r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w',line.strip()) and spectra == self.spectrum[0] : 
              #print line 
              outfile.writelines(line)
              break
            outfile.writelines(line)
          
        self.matrix_printer(infile, outfile, spectra)
  
  def __init__(self, inputFiles, **pertDict):
    """
      Parse the PHISICS Fission Yield data file and put the isotopes name as key and 
      the FY relative to the isotopes as values  
      In: FissionYield.dat
      Out: isotopeDecay (dictionary)
    """
    numbering = {}
    self.fissionYield     = {}
    self.FYList = []
    concatenateYieldList = []
    self.allYieldList = [] 
    self.inputFiles = inputFiles 
    spectrumCounter = 0
    self.inputFiles = inputFiles 
    #print self.inputFiles
    OpenInputFile = open (self.inputFiles, "r")
    lines = OpenInputFile.readlines()
    OpenInputFile.close()
    self.spectrum = ['Thermal', 'Fast']
    self.isotopeSpectrumClassifier = {}
    self.typeOfSpectrum = None
    self.isotopeList = []
    self.spectrumNumbering = {}
    
    self.pertYieldDict = pertDict
    #print self.pertYieldDict
    for key, value in self.pertYieldDict.iteritems(): 
      self.pertYieldDict[key] = '%.3E' % Decimal(str(value)) #convert the values into scientific values
    #print self.pertDict
    
    for line in lines:  
      #print line 
      if   re.match(r'(.*?)Thermal Fission Yield', line) : self.typeOfSpectrum = self.spectrum[0]  
      elif re.match(r'(.*?)Fast Fission Yield', line)    : self.typeOfSpectrum = self.spectrum[1]
      if (re.match(r'(.*?)\w+(-?)\d+\s+\w+\s+\w(-?)\d+\s+\w',line) and any(s in "FY" for s in line)) :        # create dynamic column detector 
        count = 0 
        FYgroup = []                                   # reset the counter and the dictionary self.numbering if new colum sequence is detected 
        numbering = {}
        line = re.sub(r'FY',r'',line)                  # Remove the word 'FY' in the fission columns 
        splitStringYieldType = line.upper().split()    # Split the words into individual strings 
        #print splitStringYieldType
        for i in splitStringYieldType :
          FYgroup.append(i.replace('-',''))       # get the fission yield group's names (U235, Pu239 etc.) and remove the dash in those IDs
        concatenateYieldList = concatenateYieldList + FYgroup  # concatenate all the possible decay type (including repetition among actinides and FP)
        self.allYieldList = list(set(concatenateYieldList))
        #print FYgroup
        #print self.allYieldList
        for i in xrange(len(splitStringYieldType)) :   # assign the column position of the given yield types
          count = count + 1 
          numbering[FYgroup[i]] = count   # assign the column position of the given Yield types
          splitStringYieldType[i] = re.sub(r'(.*?)(\w+)(-)(\d+M?)',r'\1\2\4', splitStringYieldType[i])
          if self.typeOfSpectrum == self.spectrum[0]: self.spectrumNumbering[self.spectrum[0]] = numbering
          if self.typeOfSpectrum == self.spectrum[1]: self.spectrumNumbering[self.spectrum[1]] = numbering
          #print self.spectrumNumbering.get('Thermal')
          #print splitStringYieldType
          numbering[splitStringYieldType[i]] = count
        #print self.numbering
      if re.match(r'(.*?)\s+\D+(-?)\d+(M?)\s+\d+.\d', line) or re.match(r'(.*?)ALPHA\s+\d+.\d', line):
        #print line
        isotopeLines = line.split()
        self.isotopeList.append(isotopeLines[0])
    self.isotopeList = list(set(self.isotopeList))
    self.numberOfIsotopes = len(self.isotopeList)
    #print len(self.isotopeList)
    #print self.isotopeList
    self.fileReconstruction()
            
  def fileReconstruction(self):
    """
      Method convert the formatted dictionary pertdict -> {'FY|THERMAL|U235|RB87':1.30} 
      into a dictionary of dictionaries that has the format -> {'U235':{'ALPHA':1.30}}
      In: Dictionary pertDict 
      Out: Dictionary of dictionaries listedDict 
    """
    self.listedYieldDict = {}
    fissioningActinide = []
    resultingFP = []
    spectrumType = []
    for i in self.pertYieldDict.iterkeys() :
      splittedYieldKeywords = i.split('|')
      spectrumType.append(splittedYieldKeywords[1])
      fissioningActinide.append(splittedYieldKeywords[2])
      resultingFP.append(splittedYieldKeywords[3])
    #print FissioningActinide
    #print resultingFP
    for i in xrange (0,len(spectrumType)):
      self.listedYieldDict[spectrumType[i]] = {}
      #print self.listedYieldDict
      for j in xrange (0,len(resultingFP)):
        self.listedYieldDict[spectrumType[i]][resultingFP[j]] = {}   # declare all the dictionaries
        for k in xrange(0,len(fissioningActinide)):
          self.listedYieldDict[spectrumType[i]][resultingFP[j]][fissioningActinide[k]] = {} 
    #print self.listedYieldDict
    for yieldTypeKey, yieldValue in self.pertYieldDict.iteritems():
      yieldKeyWords = yieldTypeKey.split('|')
      for i in xrange (0, len(self.allYieldList)):
        self.listedYieldDict[yieldKeyWords[1]][yieldKeyWords[3]][yieldKeyWords[2]] = yieldValue
    #print self.listedYieldDict
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
    open(modifiedFile, 'w')
    for spectra in self.spectrum :
      #print spectra 
      self.hardcopy_printer(self.inputFiles, spectra, modifiedFile)
    with open(modifiedFile, 'a') as outfile:
      outfile.writelines(' end')
    copyfile(modifiedFile, self.inputFiles)
    self.removeRandomlyNamedFiles(modifiedFile)

