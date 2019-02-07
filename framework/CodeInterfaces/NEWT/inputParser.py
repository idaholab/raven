"""
Created on September 19th, 2017
Modif Jan 2019
@author: rouxpn (Pascal Rouxelin)
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
    Pre-processes the variables to convert them into group structures. Implements the group structure into the newt input
  """
  def getNumberOfGroups(self):
    """
      Finds out what is the desired broad group structure desired by the user-input
      @ In, None
      @ Out, numberOfGroups, integer, number of broad groups,
    """
    numberOfGroups = 0
    for i in self.pertDict.iterkeys():
      numberOfGroups = numberOfGroups + 1
    return numberOfGroups

  def __init__(self, nFineGroups, workingDir, inputFiles, numberOfLattice, **pertDict):
    """
      Parses the newt.inp data file(s)
      @ In, nFineGroups, integer, number of collapsed groups to collaspe from
      @ In, workingDir, string, path to working directory
      @ In, inputFiles, list, list of strings containing the input file names to be parsed
      @ In, numberOfLattice, string, number of lattice cells (in string format) considered. default 1
      @ In, pertDict, dictionary, dictionary of perturbed variables
      @ Out, None
    """
    self.nFineGroups = nFineGroups
    self.inputFiles = inputFiles
    self.fineGroupStructure = nFineGroups
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
      Takes the variable value as an input, and turn it into a weight function, in order to build the group structure.
      The format is group|1: 105 group|2: 12 etc... for each group of the collapsed structure.
      @ In, numberOfGroups, interger, number of groups user-input
      @ Out, normalizedWeights, list, list of floating numbers. The intergers correspond to the number of energy bins in each broad
        group that have to be collapsed from the 252 groups
      @ Out, sumOfWeights, float, sum of the weights contained in the variable normalizedWeights
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
      If the first collapsed group includes less than 8 fine groups (if a 252 or 238 structure is used), the flux may be equal to 0.
      This methods calculates the minimum weight necessary to result in a float value equal to 8. It requires an iteration
      over the weights, and a renormalization of the weigths after the iteration.
      @ In, normalizedWeights, list of float values representing the weight of each energy bin
      @ In, sumOfWeights_iter1, float, sum of the weights on the first iteration
      @ Out, normalizedWeights, list, list of float values representing the weight of each energy bin (no zeros allowed)
    """
    rebalancedNormalizeWeights = []
    if self.nFineGroups == 252 or self.nFineGroups == 238:
      self.minimumCollapsedGroup = 8
    else:
      self.minimumCollapsedGroup = 1
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
      Calculates the minimum weight that the broad group 1 must have to absorb at least 8 fine groups.
      @ In, normalizedWeights, list of float values representing the weight of each energy bin
      @ In, sumOfWeights_iter1, float, sum of the weights on the first iteration
      @ Out, sumOfWeights_iter2, float, sum of the weights on the second iteration
      @ Out, minimumWeightGr1, float, minimum value the weight corresponding to group 1 must have to satisfy the condition
        "the first eight fine groups must be collapsed if the number of fine groups is 252 or 238"

    """
    minimumWeightGr1 = self.minimumCollapsedGroup * sumOfWeights_iter1 / self.fineGroupStructure
    sumOfWeights_iter2 = minimumWeightGr1
    for i in xrange (1,len(normalizedWeights)):
      sumOfWeights_iter2 = sumOfWeights_iter2 + normalizedWeights[i] * sumOfWeights_iter1 / self.fineGroupStructure
    return sumOfWeights_iter2, minimumWeightGr1

  def fixZeros(self,normalizedWeights):
    """
      This methods ensures the routine does not collapse 0 broad group.
      Pragmatically, it avoids in a string 4r1, 0r5, 4r3 etc...
      @ In, normalizedWeights, list, list of floats
      @ Out, normalizedWeights, list, list of float > 1
    """
    for i in xrange (0,len(normalizedWeights)):
      if normalizedWeights[i] < 1.00:
        normalizedWeights[i] = 1.00
    return normalizedWeights

  def roundBinValues(self,normalizedWeights):
    """
      Currently, the energy bins come as a floating number. This method turns the number of bins into integer numbers, and
      ensures the number of groups is equal to N (N = 252 or 238 or 56).
      @ In, normalizedWeights, float, number of energy bins
      @ Out, energyBins, list, list of integer containing the broad group structure
    """
    energyBins = []
    isItN = True
    residualList, sumOfIntegerValues = self.getDecimalPart(normalizedWeights)
    copyResidualList = residualList
    if sumOfIntegerValues != self.fineGroupStructure:
      sumOfIntegerValues = self.pickTheLargestValue(copyResidualList,normalizedWeights,sumOfIntegerValues)
      isItN = self.verifyStructure(sumOfIntegerValues)
      for i in range (len(normalizedWeights)):
        energyBins.append(int(normalizedWeights[i]))
      if isItN is False: # in 252 for example, if the round up ends up to 253, reduces the largest bin by 1
        maxE = max(energyBins)
        sumEB = sum(energyBins)
        exceeding = sumEB - self.fineGroupStructure
        for i in range (len(energyBins)):
          if energyBins[i] == maxE:
            energyBins[i] = maxE - exceeding
            break
        isItN = self.verifyStructure(sum(energyBins))
    else:
      for i in range (len(normalizedWeights)):
        energyBins.append(int(normalizedWeights[i]))
    return energyBins

  def getDecimalPart(self,normalizedWeights):
    """
      Gets the decimal value of the floating number. For example: 1.2536 will return 0.2536.
      @ In, normalizedWeights, list, list of energy bin weighting values
      @ Out, decimalList, list, list containing the resuduals of the weighting values
      @ Out, sumOfIntegerValues, integer, sum of the integer part of the weights
    """
    decimalList = []
    sumOfIntegerValues = 0
    for i in xrange (0,len(normalizedWeights)):
      decimalList.append(normalizedWeights[i] - int(normalizedWeights[i]))
      sumOfIntegerValues = int(normalizedWeights[i]) + sumOfIntegerValues
    return decimalList, sumOfIntegerValues

  def modifyDuplicateValues(self,variableList):
    """
      Finds the potential duplicate(s) in the decimal part of the weighting functions.
      If a duplicate is detected, a random value (positive or negative) is added arbitrarly to one of the values.
      @ In, variableList, list, list of floating values from the weighting functions
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

  def pickTheLargestValue(self,copyDecimalList,normalizedWeights,sumOfIntegerValues):
    """
      Picks the largest decimal value, and rounds up the value corresponding to it.
      Example: if there are three bins with the weights: 1.17; 256.25; 47.88, the largest
      decimal is 0.88, so 47.88 is rounded up to 48.
      @ In, copyDecimalList, list, list of residual values of the bin weights
      @ Out, sumOfIntegerValues, integer, sum of the integer values of the weights
    """
    while sumOfIntegerValues < self.fineGroupStructure:
      sumOfIntegerValues = 0
      largestFloat = max(copyDecimalList)
      copyDecimalList.remove(largestFloat)
      for i in xrange(0,len(normalizedWeights)):
        if normalizedWeights[i] == int(normalizedWeights[i]) + largestFloat:
          normalizedWeights[i] = int(normalizedWeights[i]) + 1
        sumOfIntegerValues = int(normalizedWeights[i]) + sumOfIntegerValues
    return sumOfIntegerValues

  def verifyStructure(self,sumOfIntegerValues):
    """
      Verifies that the final structure has collapsed N groups. Raises an error if not.
      @ In, sumOfIntegerValues, interger, number of groups collapsed
      @ Out, Boolean
    """
    if sumOfIntegerValues != self.fineGroupStructure:
      print ('WARNING ! the number of groups of the collapsed structure is not equal to 252')
      return False
    return True

  def formatIntoNewtFriendlyGroupStructure(self,energyBins):
    """
      Turns the energy strcuture into a NEWT-friendly energy bins.
      Example: [20,10,58,150] becomes [20r1, 10r2, 58r3,150r4].
      @ In, energyBins, list, list of integers containing the energy bins
      @ Out, newtEnergyBins, string, energy bins in a newt-friendly format
    """
    groupNumber = 0
    newtEnergyBinsList = []
    for i in xrange (0,len(energyBins)):
      groupNumber = groupNumber + 1
      newtEnergyBinsList.append(str(energyBins[i])+'r'+str(groupNumber))
    newtEnergyBins = ' '.join(newtEnergyBinsList)
    return newtEnergyBins

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
      copyfile(inFile, self.inputFiles[i])
      with open(inFile, "rt") as lin:
        with open(self.inputFiles[i], "wt") as lout:
          for line in lin:
            lout.write(line.replace('read collapse','read collapse'+"\n"+newtEnergyBins))
      copyfile(self.inputFiles[i], os.path.join(workingDir,self.inputFiles[i]))
