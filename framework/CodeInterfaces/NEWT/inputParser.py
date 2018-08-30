"""
Created on September 19th, 2017
@author: rouxpn
"""

import os
import sys
import re 
from shutil import copyfile 
import fileinput 
from decimal import Decimal 
import time
import xml.etree.ElementTree as ET
from random import *
import collections 

class inputParser():
  """
    takes the RAVEN variables (group structure) parses them and implement the group structure into the newt input 
  """
  def getNumberOfGroups(self):
    """
      finds out what is the desired broad group structure desired by the user-input
      @ In, None
      @ Out, numberOfGroups, integer, number of broad groups, 
    """
    numberOfGroups = 0
    for i in self.pertDict.iterkeys():
      numberOfGroups = numberOfGroups + 1 
    return numberOfGroups  
  
  def __init__(self, workingDir, inputFiles, numberOfLattice, **pertDict):
    """
      Parses the newt.inp data file 
      @ In, workingDir, string, path to working dir
      @ In, inputFiles, list, list of strings containing the input file names to be parsed (scale/NEWT input)
      @ In, numberOfLattice, string, number of lattice cells (in string format) considered. default 1 
      @ In, pertDict, dictionary, dictionary of perturbed variables
    """
    self.inputFiles = inputFiles
    self.fineGroupStructure = 252  
    self.pertDict = pertDict
    numberOfGroups = self.getNumberOfGroups()
    normalizedWeights, sumOfWeights_iter1 = self.generateWeights(numberOfGroups)
    normalizedWeights = self.fixFirstEnergyGroup(normalizedWeights,sumOfWeights_iter1)
    normalizedWeights = self.fixZeros(normalizedWeights)
    energyBins = self.roundBinValues(normalizedWeights)
    newtEnergyBins = self.formatIntoNewtFriendlyGroupStructure(energyBins)
    self.printInput(workingDir, newtEnergyBins,numberOfLattice)

  def generateWeights(self, numberOfGroups):
    """
      Take the variable value as an input, and turn into a weight function, in order to build the group structure. 
      The format is group|1: 105 group|2: 12 etc... for each group of the collapsed structure. 
      The goal of this function is to turn the intger value of each group into proper N broad group strucutre collapsed from 252 groups
      @ In, numberOfGroups, interger, number of groups user-input
      @ Out, collapsedStructure, list, list of floats. The intergers correspond to the number of energy bins in each broad group that have to be collapsed from the 252 groups 
    """
    sumOfWeights = 0
    normalizedWeights = []
    variableList = []
    for group,value in self.pertDict.iteritems():
      variableList.append(value)
    uniqueVariableList = self.modifyDuplicateValues(variableList)
    for i in xrange(0,len(uniqueVariableList)): 
      sumOfWeights = sumOfWeights + uniqueVariableList[i]
    for i in xrange(0,len(uniqueVariableList)): 
      normalizedWeights.append(uniqueVariableList[i]/sumOfWeights * self.fineGroupStructure) 
    return normalizedWeights, sumOfWeights
    
  def fixFirstEnergyGroup(self,normalizedWeights,sumOfWeights_iter1):
    """
      in case the bin of the first collasped group is < 3, it may end up resulting in 0 zero XS values. 
      This method fixes the float values lower than 3 in order to make a collasping group having at least three groups (the three fastest groups). 
      This methods calculates the minimum weight necessary to result in a float value equal to 3. It requires an iteration 
      over the weights, and a renormalization of the weigths after the iteration.
      @ In, normalizedWeights, list of float values representing the weight of each energy bin 
      @ In, sumOfWeights_iter1, float, sum of the weights on the first iteration 
      @ Out, normalizedWeights, lost of float values, representing the weight of each energy bin but no zeros allowed
    """
    rebalancedNormalizeWeights = []
    self.minimumCollapsedGroup = 8
    if normalizedWeights[0] < self.minimumCollapsedGroup:
      sumOfWeights_iter2,minimumWeightGr1 = self.minimumWeight(normalizedWeights,sumOfWeights_iter1)
      rebalancedNormalizeWeights.append(minimumWeightGr1/sumOfWeights_iter2 * self.fineGroupStructure)
      for i in xrange (1,len(normalizedWeights)):
        rebalancedNormalizeWeights.append(normalizedWeights[i]*sumOfWeights_iter1 / sumOfWeights_iter2)
      return rebalancedNormalizeWeights
    else:
      return normalizedWeights
  
  def minimumWeight(self,normalizedWeights,sumOfWeights_iter1):
    """
      calculates the minimum weights a group 1 has to have in order to be at least equal to 3. 
      @ In, normalizedWeights, list of float values representing the weight of each energy bin 
      @ In, sumOfWeights, float, sum of the weights on the first iteration 
    """
    minimumWeightGr1 = self.minimumCollapsedGroup * sumOfWeights_iter1 / self.fineGroupStructure
    sumOfWeights_iter2 = minimumWeightGr1 
    for i in xrange (1,len(normalizedWeights)):
      sumOfWeights_iter2 = sumOfWeights_iter2 + normalizedWeights[i] * sumOfWeights_iter1 / self.fineGroupStructure
    return sumOfWeights_iter2, minimumWeightGr1
  
  def fixZeros(self,normalizedWeights):
    """
      This methods ensures no group collapsed into 0 broad group. 
      Pragmatically, it avoids in a a string 4r1, 0r5, 4r3 etc...
      @ In, normalizedWeights, list of floats 
      @ Out, normalizedWeights, list of float, all floats greater than 1 
    """
    for i in xrange (0,len(normalizedWeights)):
      if normalizedWeights[i] < 1.00:
        normalizedWeights[i] = 1.00 
    return normalizedWeights
  
  def roundBinValues(self,normalizedWeights):
    """
      Currently, the energy bins come as a floating number. This method turns the number of bins into integer numbers, and 
      ensures the number of groups is equal to 252. 
      @ In, normalizedWeights, float, number of energy bins 
      @ Out, int(normalizedWeights), list, list of integer giving the broad group structure 
    """
    energyBins = []
    isIt252 = True
    residualList, sumOfIntegerValues = self.getResidualPart(normalizedWeights)
    copyResidualList = residualList
    
    
    if sumOfIntegerValues != self.fineGroupStructure:
      sumOfIntegerValues = self.pickTheLargestValue(copyResidualList,normalizedWeights,sumOfIntegerValues)
      isIt252 = self.verifyStructure(sumOfIntegerValues)
      print isIt252 
      for i in range (len(normalizedWeights)):
        energyBins.append(int(normalizedWeights[i]))
      if isIt252 is False: # if the round up ends up to 253, reduces the largest bin by 1
        print 'second False'
        maxE = max(energyBins)
        sumEB = sum(energyBins)
        exceeding = sumEB - self.fineGroupStructure
        for i in range (len(energyBins)):
          if energyBins[i] == maxE:
            energyBins[i] = maxE - exceeding
            break 
        print 'verif 1'
        print isIt252
        print 'current sum'
        print sumEB
        isIt252 = self.verifyStructure(sum(energyBins))
        print 'verif 2'
        print isIt252
    else:
      for i in range (len(normalizedWeights)):
        energyBins.append(int(normalizedWeights[i]))
    
    return energyBins    
  
  def getResidualPart(self,normalizedWeights):
    """
      get the residual value of the floating number. For example: 1.2536 will give 0.2536. 
      @ In, normalizedWeights, list, list of energy bin weighting values
      @ Out, ResidualList, list, list containing the resuduals of the weighting values. 
    """
    residualList = []
    sumOfIntegerValues = 0 
    for i in xrange (0,len(normalizedWeights)):
      residualList.append(normalizedWeights[i] - int(normalizedWeights[i]))
      sumOfIntegerValues = int(normalizedWeights[i]) + sumOfIntegerValues
    return residualList, sumOfIntegerValues
  
  def modifyDuplicateValues(self,variableList):
    """
      finds the duplicate in the floating part of the weighting functions. 
      If a duplicate is detected, a randomValue (positive or negative) is added arbitrarly to one of the values. 
      @ In, variableList, list, list of floating values of the group weighting functions
      @ Out, variableList, list, list modified in case there were duplicates
    """
    valueSeen = set() 
    uniqueValues    = [x for x in variableList if x not in valueSeen and not valueSeen.add(x)]
    duplicateValues = [item for item, count in collections.Counter(variableList).items() if count > 1]
    if len(uniqueValues) == len(variableList):
      return variableList
    else:  
      for n in xrange (0,len(variableList)):
        for m in xrange (0,len(duplicateValues)):
          if abs(variableList[n] - duplicateValues[m]) < 1e-15:
            randomNumber = randint(1,1e9)*1e-15
            sign = (-1)^randint(1,2)
            newResidual = variableList[n] + sign * randomNumber
            if newResidual < 0:
              newResidual = variableList[n] + randomNumber
            if newResidual > 1:
              newResidual = variableList[n] - randomNumber  
            variableList[n] = newResidual
      if n == (len(variableList)-1): 
        return variableList
  
  def pickTheLargestValue(self,copyResidualList,normalizedWeights,sumOfIntegerValues):
    """
      picks the largest residual value, and rounds up the value corresponding to it. 
      Example: if there are three bins with the weights: 1.17; 256.25; 47.88, the largest 
      floating number is 0.88, so 47 is rounded up to 48
      @ In, copyResidualList, list, list of residual values of the bin weights
      @ Out, 
    """
    while sumOfIntegerValues < self.fineGroupStructure:
      sumOfIntegerValues = 0
      largestFloat = max(copyResidualList)
      copyResidualList.remove(largestFloat)
      for i in xrange(0,len(normalizedWeights)):
        if normalizedWeights[i] == int(normalizedWeights[i]) + largestFloat:
          normalizedWeights[i] = int(normalizedWeights[i]) + 1
        sumOfIntegerValues = int(normalizedWeights[i]) + sumOfIntegerValues
    return sumOfIntegerValues
          
  def verifyStructure(self,sumOfIntegerValues):
    """
      verifies that the final structure has collapsed 252 groups. Raises an error if not. 
      @ In, sumOfIntegerValues, interger, number of groups collapsed
      @ Out, None
    """
    if sumOfIntegerValues != self.fineGroupStructure: 
      print sumOfIntegerValues
      print ('WARNING ! the number of groups of the collapsed structure is not equal to 252')
      return False
      #raise ValueError('ERROR ! the number of groups of the collapsed structure is not equal to 252')
    return True
      
  def formatIntoNewtFriendlyGroupStructure(self,energyBins):
    """
      turns the energy strcuture into a NEWT-friendly energy bins
      so from [20,10,58,150] to [20r1, 10r2, 58r3,150r4]
      @ In, energyBins, list, list of integers containing the energy bins 
      @ Out, newtEnergyBins, string, string of integers containing the energy bins in a newt-friendly format 
    """
    groupNumber = 0
    newtEnergyBinsList = [] 
    for i in xrange (0,len(energyBins)):
      groupNumber = groupNumber + 1 
      newtEnergyBinsList.append(str(energyBins[i])+'r'+str(groupNumber))
    newtEnergyBins = ' '.join(newtEnergyBinsList)
    #print newtEnergyBins
    return newtEnergyBins
   
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
    return str(randint(1,1000000000000))+'.xml'
    
  def printInput(self,workingDir,newtEnergyBins,numberOfLattice):
    """
      Method to print out the new input into a newt format
      @ In, workingDir, string, path to working dir
      @ In, outfile, string, optional, output file root
      @ In, numberOfLattice, string, number of lattice cells (in string format) in which the group structure is printed
      @ Out, None
    """
    for i in range(int(numberOfLattice)):
      inFile = 'template'+str(i)+'.inp'
      modifiedFile = self.generateRandomName() 
      copyfile(inFile, modifiedFile)
      with open(inFile, "rt") as lin:
        with open(modifiedFile, "wt") as lout:
          for line in lin:
            lout.write(line.replace('read collapse','read collapse'+"\n"+newtEnergyBins)) 
      copyfile(modifiedFile, os.path.join(workingDir,self.inputFiles[i]))  
      
      self.removeRandomlyNamedFiles(modifiedFile)

