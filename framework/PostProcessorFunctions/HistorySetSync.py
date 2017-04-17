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

"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


class HistorySetSync(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to HistorySet
   The conversion is made so that all histories are syncronized in time.
   It can be used to allow the histories to be sampled at the same time instant.
  """

  def initialize(self, numberOfSamples=None, pivotParameter=None, extension=None, syncMethod=None):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'

    self.numberOfSamples = numberOfSamples
    self.pivotParameter          = pivotParameter
    self.extension       = extension
    self.syncMethod      = syncMethod


  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'numberOfSamples':
        self.numberOfSamples = int(child.text)
      elif child.tag == 'syncMethod':
        self.syncMethod = child.text
      elif child.tag == 'pivotParameter':
        self.pivotParameter = child.text
      elif child.tag == 'extension':
        self.extension = child.text
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    validSyncMethods = ['all','grid','max','min']
    if self.syncMethod not in validSyncMethods:
      self.raiseAnError(NotImplementedError,'Method for synchronizing was not recognized: \"',self.syncMethod,'\". Options are:',validSyncMethods)
    if self.syncMethod is 'grid' and not isinstance(self.numberOfSamples, int):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : number of samples is not correctly specified (either not specified or not integer)')
    if self.pivotParameter == None:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : pivotParameter is not specified')
    if self.extension == None or not (self.extension == 'zeroed' or self.extension == 'extended'):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : extension type is not correctly specified (either not specified or not one of its possible allowed values: zeroed or extended)')


  def run(self,inputDic):
    """
     Method to post-process the dataObjects
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputPSDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDic = inputDic[0]
      outputDic={}
      outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
      outputDic['data'] = {}
      outputDic['data']['input'] = copy.deepcopy(inputDic['data']['input'])
      outputDic['data']['output'] = {}

      newTime = []
      if self.syncMethod == 'grid':
        maxEndTime = []
        minInitTime = []
        for hist in inputDic['data']['output']:
          maxEndTime.append(inputDic['data']['output'][hist][self.pivotParameter][-1])
          minInitTime.append(inputDic['data']['output'][hist][self.pivotParameter][0])
        maxTime = max(maxEndTime)
        minTime = min(minInitTime)
        newTime = np.linspace(minTime,maxTime,self.numberOfSamples)
      elif self.syncMethod == 'all':
        times = set()
        for hist in inputDic['data']['output']:
          for value in inputDic['data']['output'][hist][self.pivotParameter]:
            times.add(value)
        times = list(times)
        times.sort()
        newTime = np.array(times)
      elif self.syncMethod in ['min',"max"]:
        notableHist = None   #set on first iteration
        notableLength = None #set on first iteration
        for h,hist in enumerate(inputDic['data']['output'].keys()):
          l = len(inputDic['data']['output'][hist][self.pivotParameter])
          if (h==0) or (self.syncMethod == 'max' and l > notableLength) or (self.syncMethod == 'min' and l < notableLength):
            notableHist = inputDic['data']['output'][hist][self.pivotParameter][:]
            notableLength = l
        newTime = np.array(notableHist)
      for hist in inputDic['data']['output']:
        outputDic['data']['output'][hist] = self.resampleHist(inputDic['data']['output'][hist],newTime)
      return outputDic

  def resampleHist(self, vars, newTime):
    newVars={}
    for key in vars.keys():
      if key != self.pivotParameter:
        newVars[key]=np.zeros(newTime.size)
        pos=0
        for newT in newTime:
          if newT<vars[self.pivotParameter][0]:
            if self.extension == 'extended':
              newVars[key][pos] = vars[key][0]
            elif self.extension == 'zeroed':
              newVars[key][pos] = 0.0
          elif newT>vars[self.pivotParameter][-1]:
            if self.extension == 'extended':
              newVars[key][pos] = vars[key][-1]
            elif self.extension == 'zeroed':
              newVars[key][pos] = 0.0
          else:
            index = np.searchsorted(vars[self.pivotParameter],newT)
            newVars[key][pos] = vars[key][index-1] + (vars[key][index]-vars[key][index-1])/(vars[self.pivotParameter][index]-vars[self.pivotParameter][index-1])*(newT-vars[self.pivotParameter][index-1])
          pos=pos+1

    newVars[self.pivotParameter] = copy.deepcopy(newTime)
    return newVars
