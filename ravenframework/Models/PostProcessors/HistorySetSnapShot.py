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
import os
import numpy as np
from scipy import interpolate
import copy
import importlib

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes

class HistorySetSnapShot(PostProcessorReadyInterface):
  """
    This Post-Processor performs the conversion from HistorySet to PointSet
    The conversion is made so that each history H is converted to a single point P.
    Assume that each history H is a dict of n output variables x_1=[...],x_n=[...],
    then the resulting point P can be as follows accordingly to the specified type:
     - type = timeSlice: at time instant t: P=[x_1[t],...,x_n[t]]
     - type = min, max, average, value
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
    HSSSTypeType = InputTypes.makeEnumType("HSSSType", "HSSSTypeType", ['min','max','average','value','timeSlice','mixed'])
    inputSpecification.addSub(InputData.parameterInputFactory("type", contentType=HSSSTypeType))
    inputSpecification.addSub(InputData.parameterInputFactory("numberOfSamples", contentType=InputTypes.IntegerType))
    HSSSExtensionType = InputTypes.makeEnumType("HSSSExtension", "HSSSExtensionType",  ['zeroed','extended'])
    inputSpecification.addSub(InputData.parameterInputFactory("extension", contentType=HSSSExtensionType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotVar", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("pivotVal", contentType=InputTypes.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("timeInstant", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("mixed", contentType=InputTypes.StringListType))
    for tag in ['min','max','average']:
      inputSpecification.addSub(InputData.parameterInputFactory(tag, contentType=InputTypes.StringListType))
    valueSub = InputData.parameterInputFactory("value")
    valueSub.addParam("pivotVar", InputTypes.StringType)
    valueSub.addParam("pivotVal", InputTypes.StringType)
    inputSpecification.addSub(valueSub)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('dict')
    self.keepInputMeta(True)
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    self.type            = None
    self.pivotParameter  = None #pivotParameter identify the ID of the temporal variabl
    self.pivotVar        = None
    self.pivotVal        = None
    self.timeInstant     = None
    self.numberOfSamples = None
    self.interpolation   = None
    self.classifiers = {} #for "mixed" mode

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
    #sync if needed
    if self.type == 'timeSlice':
      #for syncing, need numberOfSamples, extension
      if self.numberOfSamples is None:
        self.raiseIOError(IOError,'When using "timeSlice" a "numberOfSamples" must be specified for synchronizing!')
      if self.extension is None:
        self.raiseAnError(IOError,'When using "timeSlice" an "extension" method must be specified for synchronizing!')
      #perform sync
      # Delayed import, import HistorySetSync as HSS
      from .Factory import factory as interfaceFactory
      self.HSsyncPP = interfaceFactory.returnInstance('HistorySetSync')
      self.HSsyncPP.setParams(self.numberOfSamples,self.pivotParameter,self.extension,syncMethod='grid')
      self.HSsyncPP.initialize(runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      tag = child.getName()
      if tag =='type':
        self.type = child.value
      elif tag == 'numberOfSamples':
        self.numberOfSamples = child.value
      elif tag == 'extension':
        self.extension = child.value
      elif tag == 'pivotParameter':
        self.pivotParameter = child.value
      elif tag == 'pivotVar':
        self.pivotVar = child.value
      elif tag == 'pivotVal':
        self.pivotVal = child.value
      elif tag == 'timeInstant':
        self.timeInstant = child.value
      elif self.type == 'mixed':
        entries = child.value
        if tag not in self.classifiers.keys():
          self.classifiers[tag] = []
        #min,max,avg need no additional information to run, so list is [varName, varName, ...]
        if tag in ['min','max','average']:
          self.classifiers[tag].extend(entries)
        #for now we remove timeSlice in mixed mode, until we recall why it might be desirable for a user
        #timeSlice requires the time at which to slice, so list is [ (varName,time), (varName,time), ...]
        #elif tag in ['timeSlice']:
        #  time = child.attrib.get('value',None)
        #  if time is None:
        #    self.raiseAnError('For "mixed" mode, must specify "value" as an attribute for each "timeSlice" node!')
        #  for entry in entries:
        #    self.classifiers[tag].append( (entry,float(time)) )
        #value requires the dependent variable and dependent value, so list is [ (varName,depVar,depVal), ...]
        elif tag == 'value':
          depVar = child.parameterValues.get('pivotVar',None)
          depVal = child.parameterValues.get('pivotVal',None)
          if depVar is None or depVal is None:
            self.raiseAnError('For "mixed" mode, must specify both "pivotVar" and "pivotVal" as an attribute for each "value" node!')
          for entry in entries:
            self.classifiers[tag].append( (entry,depVar,float(depVal)) )
        elif tag != 'method':
          self.raiseAnError(IOError,'Unrecognized node for HistorySetSnapShot in "mixed" mode:',tag)
      else:
        self.raiseAnError(IOError, 'HistorySetSnapShot Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child.tag) + ' is not recognized')

    needspivotParameter = ['average','timeSlice']
    if self.type in needspivotParameter or any(mode in self.classifiers.keys() for mode in needspivotParameter):
      if self.pivotParameter is None:
        self.raiseAnError(IOError,'"pivotParameter" is required for',needspivotParameter,'but not provided!')

  def run(self,inputIn, pivotVal=None):
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
      @ In, pivotVal,  float, value associated to the variable considered (default None)
      @ Out, outputPSDic, dict, output dictionary
    """
    _, _, inputDic = inputIn['Data'][0]
    #for timeSlice we call historySetWindow
    if self.type == 'timeSlice':
      outputHSDic = self.HSsyncPP.run(inputIn)
      outDict = historySetWindow(outputHSDic,self.timeInstant,inputDic['inpVars'],inputDic['outVars'],inputDic['numberRealizations'],self.pivotParameter)
      for key in inputDic['metaKeys']:
        outDict['data'][key] = inputDic['data'][key]
      return outDict

    #for other non-mixed methods we call historySnapShot
    elif self.type != 'mixed':
      outputPSDic = historySnapShot(inputDic,self.pivotVar,self.type,self.pivotVal,self.pivotParameter)
      return outputPSDic
    #   mixed is more complicated: we pull out values by method instead of a single slice type
    #   We use the same methods to get slices, then pick out only the requested variables
    else:
      #establish the output dict
      outDict = {'data':{}}
      #replicate input space
      for var in inputDic['inpVars']:
        outDict['data'][var]  = inputDic['data'][var]
      # replicate metadata
      # add meta variables back
      for key in inputDic['metaKeys']:
        outDict['data'][key] = inputDic['data'][key]
      outDict['dims'] = {key:[] for key in inputDic['dims'].keys()}
      #loop over the methods requested to fill output space
      for method,entries in self.classifiers.items():
        #min, max take no special effort
        if method in ['min','max']:
          for var in entries:
            getDict = historySnapShot(inputDic,var,method)
            outDict['data'][var] = getDict['data'][var]
        #average requires the pivotParameter
        elif method == 'average':
          for var in entries:
            getDict = historySnapShot(inputDic,var,method,tempID=self.pivotParameter,pivotVal=self.pivotParameter)
            outDict['data'][var] = getDict['data'][var]
        #timeSlice requires the time value
        #functionality removed for now until we recall why it's desirable
        #elif method == 'timeSlice':
        #  for var,time in entries:
        #    getDict = historySetWindow(inputDic,time,self.pivotParameter)
        #value requires the dependent variable and value
        elif method == 'value':
          for var,depVar,depVal in entries:
            getDict = historySnapShot(inputDic,depVar,method,pivotVal=depVal)
            outDict['data'][var] = getDict['data'][var]
      return outDict

def historySnapShot(inputDic, pivotVar, snapShotType, pivotVal=None, tempID = None):
  """
    Method do to compute a conversion from HistorySet to PointSet using the methods: min,max,average,value
    @ In, inputDic, dict, it is an historySet
    @ In, pivotVar,  string, variable considered
    @ In, pivotVal,  float, value associated to the variable considered (deault None)
    @ In, snapShotType, string, type of snapShot: min, max, average, value
    @ In, tempID, string, name of the temporal variable (default None)
    @ Out, outputDic, dict, it contains the temporal slice of all histories
  """
  # place to store data results
  outputDic={'data':{}}
  # collect metadata, if it exists, to pass through
  for key in inputDic['metaKeys']:
    outputDic['data'][key] = inputDic['data'][key]

  # place to store dimensionalities
  outputDic['dims'] = {key: [] for key in inputDic['dims'].keys()}

  for var in inputDic['inpVars']:
    outputDic['data'][var] = inputDic['data'][var]

  outVars = inputDic['data'].keys()
  outVars = [var for var in outVars if 'Probability' not in var]
  try:
    outVars.remove('prefix')
  except ValueError:
    pass
  vars = [var for var in outVars if var not in inputDic['inpVars']]

  for var in vars:
    outputDic['data'][var] = np.zeros(inputDic['numberRealizations'], dtype=object)
    for history in range(inputDic['numberRealizations']):
      if snapShotType == 'min':
        idx = np.argmin(inputDic['data'][pivotVar][history])
        outputDic['data'][var][history] = inputDic['data'][var][history][idx]
      if snapShotType == 'max':
        idx = np.argmax(inputDic['data'][pivotVar][history])
        outputDic['data'][var][history] = inputDic['data'][var][history][idx]
      elif snapShotType == 'value':
        idx = returnIndexFirstPassage(inputDic['data'][pivotVar][history],pivotVal)
        if inputDic['data'][pivotVar][history][idx]>pivotVal:
          intervalFraction = (pivotVal-inputDic['data'][pivotVar][history][idx-1])/(inputDic['data'][pivotVar][history][idx]-inputDic['data'][pivotVar][history][idx-1])
          outputDic['data'][var][history] = inputDic['data'][var][history][idx-1] + (inputDic['data'][var][history][idx]-inputDic['data'][var][history][idx-1])*intervalFraction
        else:
          intervalFraction = (pivotVal-inputDic['data'][pivotVar][history][idx])/(inputDic['data'][pivotVar][history][idx+1]-inputDic['data'][pivotVar][history][idx])
          outputDic['data'][var][history] = inputDic['data'][var][history][idx] + (inputDic['data'][var][history][idx+1]-inputDic['data'][var][history][idx])*intervalFraction
      elif snapShotType == 'average':
        cumulative=0.0
        for t in range(1,len(inputDic['data'][tempID][history])):
          cumulative += (inputDic['data'][var][history][t] + inputDic['data'][var][history][t-1]) / 2.0 * (inputDic['data'][tempID][history][t] - inputDic['data'][tempID][history][t-1])
        outputDic['data'][var][history] = cumulative / (inputDic['data'][tempID][history][-1] - inputDic['data'][tempID][history][0])
  return outputDic

def historySetWindow(inputDic,timeStepID,inpVars,outVars,N,pivotParameter):
  """
    Method do to compute a conversion from HistorySet to PointSet using the temporal slice of the historySet
    @ In, inputDic, dict, it is an historySet
    @ In, timeStepID, int, number of time sample of each history
    @ In, inpVars, list, list of input variables
    @ In, outVars, list, list of output variables
    @ In, pivotParameter, string, ID name of the temporal variable
    @ In, N, int, number of realizations
    @ Out, outDic, dict, it contains the temporal slice of all histories
  """
  outputDic={'data':{}}
  outputDic['dims'] = {key:[] for key in inputDic['dims'].keys()}
  for var in inpVars:
    outputDic['data'][var] = inputDic['data'][var]

  for var in outVars:
    outputDic['data'][var] = np.zeros(N, dtype=object)
    for rlz in range(N):
      outputDic['data'][var][rlz] = inputDic['data'][var][rlz][timeStepID]
  return outputDic

def returnIndexFirstPassage(array,value):
  """
    Function that return the index of the element that firstly crosses value
    @ In, array, np.array, array to be considered in the search
    @ In, value, double, query value
    @ Out, index, int, index of the element in the array closest to value
  """
  index=-1
  for i in range(1,array.size):
    if (array[i]>=value and array[i-1]<=value) or (array[i]<=value and array[i-1]>=value):
      index = i
      break
  return index
