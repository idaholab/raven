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
#External Modules------------------------------------------------------------------------------------
import os
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes

class HistorySetSync(PostProcessorReadyInterface):
  """
    This Post-Processor performs the conversion from HistorySet to HistorySet
    The conversion is made so that all histories are syncronized in time.
    It can be used to allow the histories to be sampled at the same time instant.
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
    inputSpecification.addSub(InputData.parameterInputFactory("numberOfSamples", contentType=InputTypes.IntegerType))
    HSSSyncType = InputTypes.makeEnumType("HSSSync", "HSSSyncType", ['all','grid','max','min'])
    inputSpecification.addSub(InputData.parameterInputFactory("syncMethod", contentType=HSSSyncType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("extension", contentType=InputTypes.StringType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.pivotParameter = 'time' #pivotParameter identify the ID of the temporal variabl
    self.setInputDataType('dict')
    self.keepInputMeta(True)
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned
    self.validDataType = ['HistorySet'] # The list of accepted types of DataObject
    self.numberOfSamples = None
    self.extension       = None
    self.syncMethod      = None

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

  def setParams(self, numberOfSamples, pivotParameter, extension, syncMethod):
    """
    """
    self.numberOfSamples = numberOfSamples
    self.pivotParameter = pivotParameter
    self.extension = extension
    self.syncMethod = syncMethod

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'numberOfSamples':
        self.numberOfSamples = child.value
      elif child.getName() == 'syncMethod':
        self.syncMethod = child.value
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'extension':
        self.extension = child.value
      else:
        self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.syncMethod == 'grid' and not isinstance(self.numberOfSamples, int):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : number of samples is not correctly specified (either not specified or not integer)')
    if self.pivotParameter is None:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : pivotParameter is not specified')
    if self.extension is None or not (self.extension == 'zeroed' or self.extension == 'extended'):
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' : extension type is not correctly specified (either not specified or not one of its possible allowed values: zeroed or extended)')

  def run(self,inputIn):
    """
      Method to post-process the dataObjects
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
      @ Out, outputPSDic, dict, output dictionary
    """
    _, _, inputDic = inputIn['Data'][0]
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
