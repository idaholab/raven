# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on October 28, 2015

@author: mandd
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


import os
import numpy as np
from scipy import interpolate
from scipy import integrate
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
    self.pivotParameter  = None
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
      elif child.tag == 'pivotParameter':
        self.pivotParameter = child.text
      elif child.tag == 'interpolation':
        self.interpolation = child.text
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.samplingType not in set(['uniform','firstDerivative','secondDerivative','filteredFirstDerivative','filteredSecondDerivative']):
      self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : sampling type is not correctly specified')
    if self.pivotParameter is None:
      self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : time ID is not specified')

    if self.samplingType == 'uniform' or self.samplingType == 'firstDerivative' or self.samplingType == 'secondDerivative':
      if self.numberOfSamples is None or self.numberOfSamples < 0:
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : number of samples is not specified or less than 0')
      if self.interpolation not in set(['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'intervalAverage']):
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : interpolation is not correctly specified; possible values: linear, nearest, zero, slinear, quadratic, cubic')

    if self.samplingType == 'filteredFirstDerivative' or self.samplingType == 'filteredSecondDerivative':
      if self.tolerance is  None or self.tolerance < 0.0:
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : tolerance is not specified or less than 0')


  def run(self,inputDic):
    """
      Method to post-process the dataObjects
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary of resampled histories
    """
    # check that we only have one data object
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')

    # grab the first (and only) data object
    inputDic = inputDic[0]
    outputDic={'data':{}}
    # load up the input data into the output
    for var in inputDic['inpVars']:
      outputDic['data'][var] = copy.deepcopy(inputDic['data'][var])

    # loop over realizations and find the desired sample points
    for hist in range(inputDic['numberRealizations']):
      # set up the realization
      rlz={}
      for var in inputDic['outVars']:
        rlz[var] = inputDic['data'][var][hist]
      rlz[self.pivotParameter]=inputDic['data'][self.pivotParameter][hist]

      # do the sampling based on what the user requested
      if self.samplingType in ['uniform','firstDerivative','secondDerivative']:
        outData = self.varsTimeInterp(rlz)
      elif self.samplingType in ['filteredFirstDerivative','filteredSecondDerivative']:
        outData = timeSeriesFilter(self.pivotParameter,rlz,self.samplingType,self.tolerance)
      else:
        self.raiseAnError(IOError, 'HistorySetSampling Interfaced Post-Processor ' + str(self.name) + ' : not recognized samplingType')

      for var in outData.keys():
        outputDic['data'][var] = np.zeros(inputDic['numberRealizations'], dtype=object)
        outputDic['data'][var][hist] = outData[var]

    if 'ProbabilityWeight' in inputDic['data'].keys():
      outputDic['data']['ProbabilityWeight'] = inputDic['data']['ProbabilityWeight']
    if 'prefix' in inputDic['data'].keys():
      outputDic['data']['prefix'] = inputDic['data']['prefix']
    outputDic['dims'] = copy.deepcopy(inputDic['dims'])
    return outputDic

  def varsTimeInterp(self, vars):
    """
      This function samples a multi-variate temporal function
      @ In, vars, dict, data set that contained the information of the multi-variate temporal function (this is supposed to be a dictionary:
                      {'pivotParameter':time_array, 'var1':var1_array, ..., 'varn':varn_array})
      @ Out, newVars, dict, data set that is a sampled version of vars
    """

    localPivotParameter = vars[self.pivotParameter]
    tMin = localPivotParameter[0]
    tMax = localPivotParameter[-1]

    newVars={}

    if self.samplingType == 'uniform':
      if self.interpolation == 'intervalAverage':
        newTime = np.linspace(tMin,tMax,self.numberOfSamples+1)[0:-1]
      else:
        newTime = np.linspace(tMin,tMax,self.numberOfSamples)
    elif self.samplingType == 'firstDerivative' or self.samplingType == 'secondDerivative':
      newTime = self.derivativeTimeValues(vars)
    else:
      self.raiseAnError(RuntimeError,'type ' + self.samplingType + ' is not a valid type. Function: varsTimeInterp (mathUtils)')

    for key in vars.keys():
      if key == self.pivotParameter:
        newVars[key] = newTime
      else:
        if self.interpolation == 'intervalAverage':
          newVars[key] = np.zeros(shape=newTime.shape)
          deltaT = newTime[1]-newTime[0] if len(newTime) > 1 else tMax
          for tIdx in range(len(newTime)):
            t = newTime[tIdx]
            extractCondition = (localPivotParameter>=t) * (localPivotParameter<=t+deltaT)
            extractVar = np.extract(extractCondition, vars[key])
            extractTime = np.extract(extractCondition, localPivotParameter)
            newVars[key][tIdx] = integrate.trapz(extractVar, extractTime) / deltaT
        else:
          interp = interpolate.interp1d(vars[self.pivotParameter], vars[key], self.interpolation)
          newVars[key]=interp(newTime)
    return newVars


  def derivativeTimeValues(self, var):
    """
      This function computes the new temporal variable
      @ In, vars, dict, data set that contained the information of the multi-variate temporal function (this is supposed to be a dictionary:
                      {'pivotParameter':time_array, 'var1':var1_array, ..., 'varn':varn_array})
      @ Out, newTime, list, values of the new temporal variable
    """
    newTime = np.zeros(self.numberOfSamples)
    cumDerivative = np.zeros(var[self.pivotParameter].size)

    normalizedVar = {}

    for keys in var.keys():
      normalizedVar[keys] = var[keys]
      if keys != self.pivotParameter:
        minVal = np.min(var[keys])
        maxVal = np.max(var[keys])
        if not max == min:
          normalizedVar[keys] = (var[keys] - minVal)/(maxVal - minVal)
        else:
          normalizedVar[keys] = var[keys]/np.float64(1.0)
      else:
        normalizedVar[keys]=var[keys]/np.float64(1.0)

    if self.samplingType=='firstDerivative':
      for t in range(1, normalizedVar[self.pivotParameter].shape[0]):
        t_contrib=0.0
        for keys in normalizedVar.keys():
          t_contrib += abs(normalizedVar[keys][t] - normalizedVar[keys][t-1])/(normalizedVar[self.pivotParameter][t]-normalizedVar[self.pivotParameter][t-1])
        cumDerivative[t] = cumDerivative[t-1] + t_contrib

    elif self.samplingType=='secondDerivative':
      for t in range(1, normalizedVar[self.pivotParameter].shape[0]-1):
        t_contrib=0.0
        for keys in normalizedVar.keys():
          t_contrib += abs(normalizedVar[keys][t+1] - 2.0 * normalizedVar[keys][t] + normalizedVar[keys][t-1])/(normalizedVar[self.pivotParameter][t]-normalizedVar[self.pivotParameter][t-1])**2
        cumDerivative[t] = cumDerivative[t-1] + t_contrib
      cumDerivative[-1] = cumDerivative[normalizedVar[self.pivotParameter].shape[0]-2]

    else:
      self.raiseAnError(RuntimeError,'type ' + self.samplingType + ' is not a valid type. Function: derivativeTimeValues')

    cumDamageInstant = np.linspace(cumDerivative[0],cumDerivative[-1],self.numberOfSamples)

    for i in range(self.numberOfSamples-1):
      index = (np.abs(cumDerivative - cumDamageInstant[i])).argmin()
      newTime[i] = var[self.pivotParameter][index]
    newTime[-1] = var[self.pivotParameter][-1]
    return newTime

def timeSeriesFilter(pivotParameter, vars, filterType, filterValue):
  """ This function sample a multi-variate temporal function
  pivotParameter      : the ID of the temporal variable
  vars        : data set that contained the information of the multi-variate temporal function (this is supposed to be a
                dictionary: {'pivotParameter':time_array, 'var1':var1_array, ..., 'varn':varn_array})
  samplingType: type of sampling used to determine the coordinate of the numSamples samples ('firstDerivative', 'secondDerivative')
  filterValue : value associated to the filter
  """
  derivative = np.zeros(vars[pivotParameter].size)

  if filterType=='filteredFirstDerivative':
    for t in range(1, len(vars[pivotParameter])):
      t_contrib=0.0
      for keys in vars.keys():
        if keys != pivotParameter:
          t_contrib += abs((vars[keys][t] - vars[keys][t-1])/(vars[pivotParameter][t] - vars[pivotParameter][t-1]))
      derivative[t] = t_contrib
  elif filterType=='filteredSecondDerivative':
    for t in range(1, vars[pivotParameter].size-1):
      t_contrib=0.0
      for keys in vars.keys():
        t_contrib += abs((vars[keys][t+1] - 2.0 * vars[keys][t] + vars[keys][t-1])/(vars[pivotParameter][t] - vars[pivotParameter][t-1])**2)
      derivative[t] = t_contrib
    derivative[-1] = derivative[len(vars[pivotParameter])-2]

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
