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
    self.settings       = {'operationType':None,'operationValue':None}

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    foundPivot = False
    for child in xmlNode:
      if child.tag  == 'pivotParameter':
        foundPivot, self.pivotParameter = True,child.text
      elif child.tag in ['row','pivotValue','operator']:
        self.settings['operationType'] = child.tag
        self.settings['operationValue'] = float(child.text) if child.tag != 'operator' else child.text
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

      outputVars = inputDic['data']['output'][inputDic['data']['output'].keys()[0]].keys()
      for outputVar in outputVars:
        outputDic['data']['output'][outputVar] = np.empty(0)

      # check if pivot value is present
      if self.settings['operationType'] == 'pivotValue':
        if self.pivotParameter not in outputVars:
            self.raiseAnError(RuntimeError,'Pivot Variable "'+str(self.pivotParameter)+'" not found in data !')

      for hist in inputDic['data']['output']:
        for outputVar in outputVars:
          if self.settings['operationType'] == 'row':
            if int(self.settings['operationValue']) >= len(inputDic['data']['output'][hist][outputVar]):
              self.raiseAnError(RuntimeError,'row value > of size of history "'+str(hist)+'" !')
            outputDic['data']['output'][outputVar] = np.append(outputDic['data']['output'][outputVar], copy.deepcopy(inputDic['data']['output'][hist][outputVar][int(self.settings['operationValue'])]))
          elif self.settings['operationType'] == 'pivotValue':
            idx = (np.abs(np.asarray(inputDic['data']['output'][hist][self.pivotParameter])-float(self.settings['operationValue']))).argmin()
            outputDic['data']['output'][outputVar] = np.append(outputDic['data']['output'][outputVar], copy.deepcopy(inputDic['data']['output'][hist][outputVar][idx]))
          else:
            # operator
            if self.settings['operationValue'] == 'max':
              outputDic['data']['output'][outputVar] = np.append(outputDic['data']['output'][outputVar], copy.deepcopy(np.max(inputDic['data']['output'][hist][outputVar])))
            elif self.settings['operationValue'] == 'min':
              outputDic['data']['output'][outputVar] = np.append(outputDic['data']['output'][outputVar], copy.deepcopy(np.min(inputDic['data']['output'][hist][outputVar])))
            elif self.settings['operationValue'] == 'average':
              outputDic['data']['output'][outputVar] = np.append(outputDic['data']['output'][outputVar], copy.deepcopy(np.mean(inputDic['data']['output'][hist][outputVar])))
      return outputDic
