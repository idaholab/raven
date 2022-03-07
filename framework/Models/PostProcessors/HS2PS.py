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
Created on Feb 04, 2016
@author: mandd, wangc
"""
#External Modules------------------------------------------------------------------------------------
import os
import copy
import numpy as np
import xarray as xr
#External Modules End--------------------------------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from ...utils import InputData, InputTypes
from .PostProcessorReadyInterface import PostProcessorReadyInterface
#Internal Modules End-----------------------------------------------------------

class HS2PS(PostProcessorReadyInterface):
  """
    This Post-Processor performs the conversion from HistorySet to PointSet
    The conversion is made so that each history H is converted to a single point P.
    Assume that each history H is a dict of n output variables x_1=[...],x_n=[...],
    then the resulting point P is as follows; P=[x_1,...,x_n]
    Note!!!! Here it is assumed that all histories have been sync so that they have the same length,
    start point and end point. If you are not sure, do a pre-processing the the original history set
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
    inputSpecification.addSub(InputData.parameterInputFactory("features", contentType=InputTypes.StringListType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag  = 'POSTPROCESSOR HS2PS'
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True
    self.pivotParameter = None
    self.features = 'all'
    self.transformationSettings = {}
    ## dataset option
    # self.setInputDataType('xrDataset')
    self.setInputDataType('dict')
    self.keepInputMeta(True)

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
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'features':
        self.features = 'all' if 'all' in child.value else child.value
    if self.pivotParameter == None:
      self.raiseAnError(IOError, 'HS2PS Post-Processor', self.name, ': pivotParameter is not specified')

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
      @ Out, outputDic, dict, output dictionary contains the converted PointSet data
    """
    #### convert to use Dataset, the following has been tested. However, this changes
    #### will cause the failure of DataMining since interfacePP is used by DataMining
    #### for pre-processing data. When we convert all interfacePP to use Dataset, we can
    #### use the following code, and eventually remove the dict option
    #      @ In, inputIn, dict, dictionary of data.
    #          inputIn = {'Data':listData, 'Files':listOfFiles},
    #          listData has the following format: (listOfInputVars, listOfOutVars, xr.Dataset)
    #      @ Out, outDataset, xarray.Dataset, output dataset
    # inpVars, outVars, data = inputIn['Data'][0]
    # if self.features == 'all':
    #   self.features = outVars
    # outDataset = data.drop_dims(self.pivotParameter)
    # featDataset = data[self.features]
    # if featDataset[self.features[-1]].isnull().sum() > 0:
    #   self.raiseAnError(IOError, 'Found misalignment in provided DataObject!')
    # numRlz = data.dims['RAVEN_sample_ID']
    # featData = featDataset.to_array().values.transpose(1, 0, 2).reshape(numRlz, -1)
    # varNames = [str(i) for i in range(featData.shape[-1])]
    # convertedFeat = xr.DataArray(featData, dims=('RAVEN_sample_ID', 'outVars'), coords={'RAVEN_sample_ID':data['RAVEN_sample_ID'], 'outVars':varNames})
    # convertedFeatDataset = convertedFeat.to_dataset(dim='outVars')
    # outDataset = xr.merge([outDataset, convertedFeatDataset])
    # ## self.transformationSettings is used by _inverse method when doing DataMining
    # self.transformationSettings['vars'] = copy.deepcopy(self.features)
    # self.transformationSettings['timeLength'] = data[self.pivotParameter].size
    # self.transformationSettings['timeAxis'] = data[self.pivotParameter][0]
    # self.transformationSettings['dimID'] = list(outDataset.keys())
    # return outDataset
    #########

    inpVars, outVars, inputDict = inputIn['Data'][0]
    outputDic = {'data': {}}
    outputDic['dims'] = {}
    numSamples = inputDict['numberRealizations']

    # generate the input part of the output dictionary
    for inputVar in inputDict['inpVars']:
     outputDic['data'][inputVar] = inputDict['data'][inputVar]

    # generate the output part of the output dictionary
    if self.features == 'all':
     self.features = inputDict['outVars']

    historyLength = len(inputDict['data'][self.features[0]][0])
    numVariables = historyLength*len(self.features)
    for history in inputDict['data'][self.features[0]]:
     if len(history) != historyLength:
       self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : one or more histories in the historySet have different time scale')

    tempDict = {}
    matrix = np.zeros((numSamples,numVariables))
    for i in range(numSamples):
     temp = np.empty(0)
     for feature in self.features:
       temp=np.append(temp,inputDict['data'][feature][i])
     matrix[i,:]=temp

    for key in range(numVariables):
     outputDic['data'][str(key)] = np.empty(0)
     outputDic['data'][str(key)] = matrix[:,key]
     outputDic['dims'][str(key)] = []
    # add meta variables back
    for key in inputDict['metaKeys']:
     outputDic['data'][key] = inputDict['data'][key]

    self.transformationSettings['vars'] = copy.deepcopy(self.features)
    self.transformationSettings['timeLength'] = historyLength
    self.transformationSettings['timeAxis'] = inputDict['data'][self.pivotParameter][0]
    self.transformationSettings['dimID'] = outputDic['data'].keys()
    return outputDic

  def _inverse(self,inputDic):
    """
      This method is aimed to return the inverse of the action of this PostProcessor
      @ In, inputDic, dict, dictionary which contains the transformed data of this PP
      @ Out, data, dict, the dictionary containing the inverse of the data (the orginal space)
    """
    data = {}
    for hist in inputDic.keys():
      data[hist]= {}
      tempData = inputDic[hist].reshape((len(self.transformationSettings['vars']),self.transformationSettings['timeLength']))
      for index,var in enumerate(self.transformationSettings['vars']):
        data[hist][var] = tempData[index,:]
      data[hist][self.pivotParameter] = self.transformationSettings['timeAxis']
    return data
