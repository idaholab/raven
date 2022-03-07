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

#External Modules------------------------------------------------------------------------------------
import os
import copy
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------


class HStoPSOperator(PostProcessorReadyInterface):
  """
   This Post-Processor performs the conversion from HistorySet to PointSet
   The conversion is performed based on any of the following operations:
   - row value
   - pivot value
   - operator
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("row", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotValue", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("operator", contentType=InputTypes.StringType))
    PivotStategyType = InputTypes.makeEnumType("PivotStategy", "PivotStategyType", ['nearest','floor','ceiling','interpolate'])
    inputSpecification.addSub(InputData.parameterInputFactory("pivotStrategy", contentType=PivotStategyType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.pivotParameter = 'time' #pivotParameter identify the ID of the temporal variabl
    self.settings       = {'operationType':None,'operationValue':None,'pivotStrategy':'nearest'}
    self.setInputDataType('dict')
    self.keepInputMeta(True)
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')
    if inputs[0].type != 'HistorySet':
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only HistorySet dataObject, but got "{}"'.format(inputs[0].type))

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    foundPivot = False
    for child in paramInput.subparts:
      if child.getName()  == 'pivotParameter':
        foundPivot, self.pivotParameter = True,child.value.strip()
      elif child.getName() in ['row','pivotValue','operator']:
        self.settings['operationType'] = child.getName()
        self.settings['operationValue'] = child.value
      elif child.getName()  == 'pivotStrategy':
        self.settings[child.getName()] = child.value.strip()
      else:
        self.raiseAnError(IOError, 'XML node ' + str(child.tag) + ' is not recognized')
    if not foundPivot:
      self.raiseAWarning('"pivotParameter" is not inputted! Default is "'+ self.pivotParameter +'"!')
    if self.settings['operationType'] is None:
      self.raiseAnError(IOError, 'No operation has been inputted!')
    if self.settings['operationType'] == 'operator' and self.settings['operationValue'] not in ['max','min','average','all']:
      self.raiseAnError(IOError, '"operator" can be either "max", "min", "average" or "all"!')

  def run(self,inputIn):
    """
      This method performs the actual transformation of the data object from history set to point set
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
        inputIn = {'Data':listData, 'Files':listOfFiles},
        listData has the following format: (listOfInputVars, listOfOutVars, DataDict) with
        DataDict is a dictionary that has the format
            dataDict['dims']     = dict {varName:independentDimensions}
            dataDict['metadata'] = dict {metaVarName:metaVarValue}
            dataDict['type'] = str TypeOfDataObject
            dataDict['inpVars'] = list of input variables
            dataDict['outVars'] = list of output variables
            dataDict['numberRealization'] = int SizeOfDataObject
            dataDict['name'] = str DataObjectName
            dataDict['metaKeys'] = list of meta variables
            dataDict['data'] = dict {varName: varValue(1-D or 2-D numpy array)}
      @ Out, outputDic, dict, output dictionary
    """
    _, _, inputDict = inputIn['Data'][0]
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

    if self.settings['operationValue'] == 'all':
      #First of all make a new input variable of the time samples
      origPivot = inputDict['data'][self.pivotParameter]
      newPivot = np.concatenate(origPivot)
      outputDic['data'][self.pivotParameter] = newPivot
      #next, expand each of the input and meta parameters by duplicating them
      for inVar in inputDict['inpVars']+inputDict['metaKeys']:
        origSamples = outputDic['data'][inVar]
        outputDic['data'][inVar] = np.empty(0)
        for hist in range(numSamples):
          #for each sample, need to expand since same in each time sample
          outputDic['data'][inVar] = np.append(outputDic['data'][inVar],
                                               np.full(origPivot[hist].shape,
                                                       origSamples[hist]))


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
          elif self.settings['operationValue'] == 'all':
            outputDic['data'][outputVar] = np.append(outputDic['data'][outputVar], copy.deepcopy(inputDict['data'][outputVar][hist]))

    return outputDic
