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

  def initialize(self, numberOfSamples=None, pivotParameter=None, extension=None, syncMethod=None, boundaries=None):
    """
      Method to initialize the Interfaced Post-processor
      @ In, numberOfSamples, int, (default None)
      @ In, pivotParameter, str, ID of the pivot paramter (e.g., time)
      @ In, extension, type of extension to be employed
      @ In, syncMethod, type of syncrhonization method
      @ Out, None,
    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'

    self.numberOfSamples = numberOfSamples
    self.pivotParameter = pivotParameter
    self.extension = extension
    self.syncMethod = syncMethod
    self.boundaries = boundaries # min and maximum boundaries of the pivot parameter (optional)

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
      elif child.tag == 'boundaries':
        self.boundaries = [float(el) for el in child.text.split(",")]
        if len(self.boundaries) != 2:
          self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) +
                            ' must containe a comma separated list (lenhgt 2) of floats (boundaries)')
      elif child.tag !='method':
        self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    validSyncMethods = ['all','grid','max','min','bound']
    if self.syncMethod not in validSyncMethods:
      self.raiseAnError(NotImplementedError,'Method for synchronizing was not recognized: \'',self.syncMethod,'\'. Options are:',validSyncMethods)
    if self.syncMethod == 'grid' and not isinstance(self.numberOfSamples, int):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) +
                        ' : <numberOfSamples>  is not correctly specified (either not specified or not integer)')
    if self.syncMethod is 'bound' and not isinstance(self.numberOfSamples, int) and self.boundaries is None:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) +
                        ' : <numberOfSamples> is not correctly specified (either not specified or not integer) or <boundaries> not provided')
    if self.pivotParameter is None:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : pivotParameter is not specified')
    if self.extension is None or not (self.extension == 'zeroed' or self.extension == 'extended'):
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

      newTime = []
      if self.syncMethod == 'grid':
        maxEndTime = []
        minInitTime = []
        for hist in inputDic['data'][self.pivotParameter]:
          maxEndTime.append(hist[-1])
          minInitTime.append(hist[0])
        maxTime = max(maxEndTime)
        minTime = min(minInitTime)
        newTime = np.linspace(minTime,maxTime,self.numberOfSamples)
      elif self.syncMethod == 'bound':
        newTime = np.linspace(min(self.boundaries),max(self.boundaries),self.numberOfSamples)
      elif self.syncMethod == 'all':
        times = []
        for hist in inputDic['data'][self.pivotParameter]:
            times.extend(hist)
        times = list(set(times))
        times.sort()
        newTime = np.array(times)
      elif self.syncMethod in ['min','max']:
        notableHist   = None   #set on first iteration
        notableLength = None   #set on first iteration

        for h,elem in np.ndenumerate(inputDic['data'][self.pivotParameter]):
          l=len(elem)
          if (h[0] == 0) or (self.syncMethod == 'max' and l > notableLength) or (self.syncMethod == 'min' and l < notableLength):
            notableHist = inputDic['data'][self.pivotParameter][h[0]]
            notableLength = l
        newTime = np.array(notableHist)

      outputDic['data']={}
      for var in inputDic['outVars']:
        outputDic['data'][var] = np.zeros(inputDic['numberRealizations'], dtype=object)
      outputDic['data'][self.pivotParameter] = np.zeros(inputDic['numberRealizations'], dtype=object)

      for var in inputDic['inpVars']:
        outputDic['data'][var] = copy.deepcopy(inputDic['data'][var])

      for rlz in range(inputDic['numberRealizations']):
        outputDic['data'][self.pivotParameter][rlz] = newTime
        for var in inputDic['outVars']:
          oldTime = inputDic['data'][self.pivotParameter][rlz]
          outputDic['data'][var][rlz] = self.resampleHist(inputDic['data'][var][rlz], oldTime, newTime)

      # add meta variables back
      for key in inputDic['metaKeys']:
        outputDic['data'][key] = inputDic['data'][key]
      outputDic['dims'] = copy.deepcopy(inputDic['dims'])

      return outputDic

  def resampleHist(self, variable, oldTime, newTime):
    """
      Method the re-sample on ''newTime'' the ''variable'' originally sampled on ''oldTime''
      @ In, variable, np.array, array containing the sampled values of the dependent variable
      @ In, oldTime,  np.array, array containing the sampled values of the temporal variable
      @ In, newTime,  np.array, array containing the sampled values of the new temporal variable
      @ Out, variable, np.array, array containing the sampled values of the dependent variable re-sampled on oldTime
    """
    newVar=np.zeros(newTime.size)
    pos=0
    for newT in newTime:
      if newT<oldTime[0]:
        if self.extension == 'extended':
          newVar[pos] = variable[0]
        elif self.extension == 'zeroed':
          newVar[pos] = 0.0
      elif newT>oldTime[-1]:
        if self.extension == 'extended':
          newVar[pos] = variable[-1]
        elif self.extension == 'zeroed':
          newVar[pos] = 0.0
      else:
        index = np.searchsorted(oldTime,newT)
        newVar[pos] = variable[index-1] + (variable[index]-variable[index-1])/(oldTime[index]-oldTime[index-1])*(newT-oldTime[index-1])
      pos=pos+1
    return newVar
