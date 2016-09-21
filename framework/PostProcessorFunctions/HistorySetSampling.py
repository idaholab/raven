"""
Created on October 28, 2015

"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


import os
import numpy as np
from scipy import interpolate
import copy


class HistorySetSampling(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to HistorySet
   The conversion is made so that each history H is re-sampled accordingly to a specific sampling strategy.
   It can be used to reduce the amount of space required by the HistorySet.
  """
  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """

    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'

    self.samplingType    = None
    self.numberOfSamples = None
    self.tolerance       = None
    self.timeID          = None
    self.interpolation   = None

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """

    for child in xmlNode:
      if child.tag == 'samplingType':
        self.samplingType = child.text
      elif child.tag == 'numberOfSamples':
        self.numberOfSamples = int(child.text)
      elif child.tag == 'tolerance':
        self.tolerance = float(child.text)
      elif child.tag == 'timeID':
        self.timeID = child.text
      elif child.tag == 'interpolation':
        self.interpolation = child.text
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.samplingType not in set(['uniform','firstDerivative','secondDerivative','filteredFirstDerivative','filteredSecondDerivative']):
      self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : sampling type is not correctly specified')
    if self.timeID == None:
      self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : time ID is not specified')

    if self.samplingType == 'uniform' or self.samplingType == 'firstDerivative' or self.samplingType == 'secondDerivative':
      if self.numberOfSamples == None or self.numberOfSamples < 0:
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : number of samples is not specified or less than 0')
      if self.interpolation not in set(['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']):
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : interpolation is not correctly specified; possible values: linear, nearest, zero, slinear, quadratic, cubic')

    if self.samplingType == 'filteredFirstDerivative' or self.samplingType == 'filteredSecondDerivative':
      if self.tolerance == None or self.tolerance < 0.0:
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : tolerance is not specified or less than 0')


  def run(self,inputDic):
    """
     Method to post-process the dataObjects
     @ In,  inputDic , dictionary
     @ Out, outputDic, dictionary
    """
    outputDic={}
    outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
    outputDic['data'] = {}
    outputDic['data']['input'] = copy.deepcopy(inputDic['data']['input'])
    outputDic['data']['output'] = {}

    for hist in inputDic['data']['output']:
      if self.samplingType == 'uniform' or self.samplingType == 'firstDerivative' or self.samplingType == 'secondDerivative':
        outputDic['data']['output'][hist] = self.varsTimeInterp(inputDic['data']['output'][hist])
      elif self.samplingType == 'filteredFirstDerivative' or self.samplingType == 'filteredSecondDerivative':
        outputDic['data']['output'][hist] = timeSeriesFilter(self.timeID,inputDic['data']['output'][hist],self.samplingType,self.tolerance)

    return outputDic

  def varsTimeInterp(self, vars):
    """ This function sample a multi-variate temporal function
    vars        : data set that contained the information of the multi-variate temporal function (this is supposed to be a dictionary:
                  {'timeID':time_array, 'var1':var1_array, ..., 'varn':varn_array})

    """

    t_min = vars[self.timeID][0]
    t_max = vars[self.timeID][-1]

    newVars={}

    if self.samplingType == 'uniform':
      newTime = np.linspace(t_min,t_max,self.numberOfSamples)
    elif self.samplingType == 'firstDerivative' or self.samplingType == 'secondDerivative':
      newTime = self.derivativeTimeValues(vars)
    else:
      self.raiseAnError(RuntimeError,'type ' + self.samplingType + ' is not a valid type. Function: varsTimeInterp (mathUtils)')

    for key in vars.keys():
      if key == self.timeID:
        newVars[key] = newTime
      else:
        interp = interpolate.interp1d(vars[self.timeID], vars[key], self.interpolation)
        newVars[key]=interp(newTime)
    return newVars


  def derivativeTimeValues(self, var):
    t_min=self.timeID[0]
    t_max=self.timeID[-1]

    newTime = np.zeros(self.numberOfSamples)
    cumDerivative = np.zeros(var[self.timeID].size)

    normalizedVar = {}

     # data normalization
    '''for keys in var.keys():
       if keys != self.timeID:
         total = np.sum(var[keys])
       else:
         total=np.float64(1.0)
      normalizedVar[keys] = var[keys]/total
      normalizedVar[keys] = var[keys]/total'''
    
    for keys in var.keys():
      normalizedVar[keys] = var[keys]/np.float64(1.0)
      if keys != self.timeID:
        min = np.min(var[keys])
        max = np.max(var[keys])
        if not max == min:
          normalizedVar[keys] = (var[keys] - min)/(max - min)
        else:
          normalizedVar[keys] = var[keys]/np.float64(1.0) 
      else:
        normalizedVar[keys]=var[keys]/np.float64(1.0)
        
    if self.samplingType=='firstDerivative':
      for t in range(1, normalizedVar[self.timeID].shape[0]):
        t_contrib=0.0
        for keys in normalizedVar.keys():
          t_contrib += abs(normalizedVar[keys][t] - normalizedVar[keys][t-1])/(normalizedVar[self.timeID][t]-normalizedVar[self.timeID][t-1])
        cumDerivative[t] = cumDerivative[t-1] + t_contrib

    elif self.samplingType=='secondDerivative':
      for t in range(1, normalizedVar[self.timeID].shape[0]-1):
        t_contrib=0.0
        for keys in normalizedVar.keys():
          t_contrib += abs(normalizedVar[keys][t+1] - 2.0 * normalizedVar[keys][t] + normalizedVar[keys][t-1])/(normalizedVar[self.timeID][t]-normalizedVar[self.timeID][t-1])**2
        cumDerivative[t] = cumDerivative[t-1] + t_contrib
      cumDerivative[-1] = cumDerivative[normalizedVar[self.timeID].shape[0]-2]

    else:self.raiseAnError(RuntimeError,'type ' + self.samplingType + ' is not a valid type. Function: derivativeTimeValues')

    cumDamageInstant = np.linspace(cumDerivative[0],cumDerivative[-1],self.numberOfSamples)

    for i in range(self.numberOfSamples-1):
      index = (np.abs(cumDerivative - cumDamageInstant[i])).argmin()
      newTime[i] = var[self.timeID][index]
    newTime[-1] = var[self.timeID][-1]
    return newTime

def timeSeriesFilter(timeID, vars, filterType, filterValue):
  """ This function sample a multi-variate temporal function
  timeID      : the ID of the temporal variable
  vars        : data set that contained the information of the multi-variate temporal function (this is supposed to be a
                dictionary: {'timeID':time_array, 'var1':var1_array, ..., 'varn':varn_array})
  samplingType: type of sampling used to determine the coordinate of the numSamples samples ('firstDerivative', 'secondDerivative')
  filterValue : value associated to the filter
  """
  derivative = np.zeros(vars[timeID].size)

  if filterType=='filteredFirstDerivative':
    for t in range(1, len(vars[timeID])):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs((vars[keys][t] - vars[keys][t-1])/(vars[keys][t]))
      derivative[t] = t_contrib
  elif filterType=='filteredSecondDerivative':
    for t in range(1, vars[timeID].size-1):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs((vars[keys][t+1] - 2.0 * vars[keys][t] + vars[keys][t-1])/(vars[keys][t]))
      derivative[t] = t_contrib
    derivative[-1] = derivative[len(vars[timeID])-2]

  newVars = {}
  for key in vars:
    newVars[key]=np.array(vars[key][0])

  for t in range(derivative.size):
    if derivative[t] > filterValue:
      for key in vars:
        newVars[key]=np.append(newVars[key],vars[key][t])

  for key in vars:
    newVars[key]=np.append(newVars[key],vars[key][-1])

  return newVars
