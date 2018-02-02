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
Created on Feb 02, 2018
@author: alfoa
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


class HStoPSoperator(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to PointSet
   The conversion is performed based on any of the following operations:
   - row value
   - pivot value
   - operator 
  """

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None
     @ Out, None
    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'PointSet'
    #pivotParameter identify the ID of the temporal variable in the data set based on which
    # the operations are performed. Optional (defaul=time)
    self.pivotParameter = 'time'
    self.settings       = {'inputSpace':{'operationType':None,'operationValue':None},
                           'outputSpace':{'operationType':None,'operationValue':None}}  
    
  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    foundPivot = False
    for child in xmlNode:
      if child.tag == 'pivotParameter':
        foundPivot, self.pivotParameter = True,child.text
      elif child.tag in ['inputSpace','outputSpace']:
        row        = child.find("row")
        pivotValue = child.find("pivotValue")
        operator   = child.find("operator")
        if [row,pivotValue,operator].count(None) < 2:
          self.raiseAnError(IOError, 'Only one among the parameters "row", "pivotValue" and "operator" can be specified!')
        if row:
          self.settings[child.tag]['operationType'] = 'row'
          self.settings[child.tag]['operationValue'] = int(row.text)
        if pivotValue:
          self.settings[child.tag]['operationType'] = 'pivotValue'
          self.settings[child.tag]['operationValue'] = float(pivotValue.text)
        if operator:
          self.settings[child.tag]['operationType'] = 'operator'
          self.settings[child.tag]['operationValue'] = None
      elif child.tag !='method':
        self.raiseAnError(IOError, 'XML node ' + str(child.tag) + ' is not recognized')

    if not foundPivot:
      self.raiseAWarning('"pivotParameter" is not inputted! Default is "'+ self.pivotParameter +'"!')


  def run(self,inputDic):
    """
    This method performs the actual transformation of the data object from history set to point set
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'Only one DataObject is accepted!')
    else:
      inputDic = inputDic[0]
      outputDic={}
      outputDic['metadata'] = copy.deepcopy(inputDic['metadata'])
      outputDic['data'] = {}
      outputDic['data']['output'] = {}
      outputDic['data']['input']  = {}

      #generate the input part of the output dictionary
      inputVars = inputDic['data']['input'][inputDic['data']['input'].keys()[0]].keys()
      for inputVar in inputVars:
        outputDic['data']['input'][inputVar] = np.empty(0)

      for hist in inputDic['data']['input']:
        for inputVar in inputVars:
          outputDic['data']['input'][inputVar] = np.append(outputDic['data']['input'][inputVar], copy.deepcopy(inputDic['data']['input'][hist][inputVar]))

      #generate the output part of the output dictionary
      if self.features == 'all':
        self.features = []
        historiesID = inputDic['data']['output'].keys()
        self.features = inputDic['data']['output'][historiesID[0]].keys()

      referenceHistory = inputDic['data']['output'].keys()[0]
      referenceTimeAxis = inputDic['data']['output'][referenceHistory][self.pivotParameter]
      for hist in inputDic['data']['output']:
        if (str(inputDic['data']['output'][hist][self.pivotParameter]) != str(referenceTimeAxis)):
          self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different time scale')

      tempDict = {}

      for hist in inputDic['data']['output'].keys():
        tempDict[hist] = np.empty(0)
        for feature in self.features:
          if feature != self.pivotParameter:
            tempDict[hist] = np.append(tempDict[hist],copy.deepcopy(inputDic['data']['output'][hist][feature]))
        length = np.size(tempDict[hist])

      for hist in tempDict:
        if np.size(tempDict[hist]) != length:
          self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different length')

      for key in range(length):
        if key != self.pivotParameter:
          outputDic['data']['output'][key] = np.empty(0)

      for hist in inputDic['data']['output'].keys():
        for key in outputDic['data']['output'].keys():
          outputDic['data']['output'][key] = np.append(outputDic['data']['output'][key], copy.deepcopy(tempDict[hist][int(key)]))

      self.transformationSettings['vars'] = copy.deepcopy(self.features)
      self.transformationSettings['vars'].remove(self.pivotParameter)
      self.transformationSettings['timeLength'] = int(length/len(self.transformationSettings['vars']))
      self.transformationSettings['timeAxis'] = inputDic['data']['output'][1][self.pivotParameter]
      self.transformationSettings['dimID'] = outputDic['data']['output'].keys()

      return outputDic
