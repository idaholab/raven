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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import numpy as np
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


class HStoPSOperator(PostProcessorInterfaceBase):
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
    self.settings       = {'operationType':None,'operationValue':None,'pivotStrategy':'nearest'}

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    foundPivot = False
    for child in xmlNode:
      if child.tag  == 'pivotParameter':
        foundPivot, self.pivotParameter = True,child.text.strip()
      elif child.tag in ['row','pivotValue','operator']:
        self.settings['operationType'] = child.tag
        self.settings['operationValue'] = float(child.text) if child.tag != 'operator' else child.text
      elif child.tag  == 'pivotStrategy':
        self.settings[child.tag] = child.text.strip()
        if child.text not in ['nearest','floor','ceiling','interpolate']:
          self.raiseAnError(IOError, '"pivotStrategy" can be only "nearest","floor","ceiling" or "interpolate"!')
      elif child.tag !='method':
        self.raiseAnError(IOError, 'XML node ' + str(child.tag) + ' is not recognized')
    if not foundPivot:
      self.raiseAWarning('"pivotParameter" is not inputted! Default is "'+ self.pivotParameter +'"!')
    if self.settings['operationType'] is None:
      self.raiseAnError(IOError, 'No operation has been inputted!')
    if self.settings['operationType'] == 'operator' and self.settings['operationValue'] not in ['max','min','average']:
      self.raiseAnError(IOError, '"operator" can be either "max", "min" or "average"!')

  def run(self,inputDic):
    """
      This method performs the actual transformation of the data object from history set to point set
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'Only one DataObject is accepted!')
    else:
      inputDict = inputDic[0]
      outputDic = {'data': {}}
      outputDic['dims'] = {}
      numSamples = inputDict['numberRealizations']

      # generate the input part and metadata of the output dictionary
      outputDic['data'].update(inputDict['data'])

      # generate the output part of the output dictionary
      for outputVar in inputDict['outVars']:
        outputDic['data'][outputVar] = np.empty(0)

      # check if pivot value is present
      if self.settings['operationType'] == 'pivotValue':
        if self.pivotParameter not in inputDict['data']:
            self.raiseAnError(RuntimeError,'Pivot Variable "'+str(self.pivotParameter)+'" not found in data !')

      for hist in range(numSamples):
        for outputVar in inputDict['outVars']:
          if self.settings['operationType'] == 'row':
            if int(self.settings['operationValue']) >= len(inputDict['data'][outputVar][hist]):
              self.raiseAnError(RuntimeError,'row value > of size of history "'+str(hist)+'" !')
            outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(inputDict['data'][outputVar][hist][int(self.settings['operationValue'])]))
          elif self.settings['operationType'] == 'pivotValue':
            if self.settings['pivotStrategy'] in ['nearest','floor','ceiling']:
              idx = (np.abs(np.asarray(outputDic['data'][self.pivotParameter][hist])-float(self.settings['operationValue']))).argmin()
              if self.settings['pivotStrategy'] == 'floor':
                if np.asarray(outputDic['data'][self.pivotParameter][hist])[idx] > self.settings['operationValue']:
                  idx-=1
              if self.settings['pivotStrategy'] == 'ceiling':
                if np.asarray(outputDic['data'][self.pivotParameter][hist])[idx] < self.settings['operationValue']:
                  idx+=1
                  outputDic['data'][self.pivotParameter][hist]
              if idx > len(inputDict['data'][outputVar][hist]):
                idx = len(inputDict['data'][outputVar][hist])-1
              elif idx < 0:
                idx = 0
              outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(inputDict['data'][outputVar][hist][idx]))
            else:
              # interpolate
              interpValue = np.interp(self.settings['operationValue'], np.asarray(inputDict['data'][self.pivotParameter][hist]), np.asarray(inputDict['data'][outputVar][hist]))
              outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], interpValue)
          else:
            # operator
            if self.settings['operationValue'] == 'max':
              outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(np.max(inputDict['data'][outputVar][hist])))
            elif self.settings['operationValue'] == 'min':
              outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(np.min(inputDict['data'][outputVar][hist])))
            elif self.settings['operationValue'] == 'average':
              outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(np.mean(inputDict['data'][outputVar][hist])))
      return outputDic
