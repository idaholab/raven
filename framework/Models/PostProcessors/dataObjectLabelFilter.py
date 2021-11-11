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

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from utils import InputData, InputTypes

class dataObjectLabelFilter(PostProcessorReadyInterface):
  """
   This Post-Processor filters out the points or histories accordingly to a chosen clustering label
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
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{dataObjectLabelFilter} post-processor allows to filter the portion of a dataObject, either \xmlNode{PointSet}
                            or \xmlNode{HistorySet}, with a given clustering label. A clustering algorithm associates a unique cluster label to each
                            element of the dataObject (PointSet or HistorySet). This cluster label is a natural number ranging from 0 (or 1
                            depending on the algorithm) to \textit{N} where \textit{N} is the number of obtained clusters. Recall that some clustering
                            algorithms (e.g., K-Means) receive N as input while others (e.g., Mean-Shift) determine N after clustering has
                            been performed. Thus, this post-processor is naturally employed after a data-mining clustering techniques has been
                            performed on a dataObject so that each clusters can be analyzed separately.
                        """
    specs.addSub(InputData.parameterInputFactory("label", contentType=InputTypes.StringType,
                                                 descr=r"""name of the clustering label""", default=0))
    specs.addSub(InputData.parameterInputFactory("clusterIDs", contentType=InputTypes.IntegerListType,
                                                 descr=r"""IDs of the selected clusters. Note that more than one ID can be provided as input""", default=0))
    specs.addSub(InputData.parameterInputFactory("clusteringMethod", contentType=InputTypes.makeEnumType('clusteringMethod',
                              'clusteringMethodType', ['point', 'extended'])),
                                                 descr=r"""Clustering Method: if extended, the cluster label will be used to cluster over time.  (=> only the portion of the history that matches the cluster IDs will be retrived), otherwise it will be used to cluster the history based on last values of the clusterIDs. Note: IT HAS MEANING FOR HISTORY SET ONLY!""", default='extended'))
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
    self.validDataType = ['HistorySet','PointSet'] # The list of accepted types of DataObject
    self.label        = None # name of the clustering label
    self.clusterIDs   = None # ID of the selected clusters. Note that more than one ID can be provided as input
    self.clusteringMethod = 'extended' # if extended, the cluster label will be used to cluster over time
                                       # (=> only the portion of the history that matches the cluster IDs will be retrived),
                                       # otherwise it will be used to cluster the history based on
                                       # last values of the clusterIDs. Note: IT HAS MEANING FOR HISTORY SET ONLY!

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'label':
        self.label = child.value
      elif child.getName() == 'clusterIDs':
        self.clusterIDs = child.value
      elif child.getName() == 'clusteringMethod':
        self.clusteringMethod = child.value
      else:
        self.raiseAnError(IOError, 'Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

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
     @ Out, outputDic, dictionary, output dictionary to be provided to the base class
    """
    _, _, inputDict = inputIn['Data'][0]
    outputDict = {}
    if self.label not in inputDict['data']:
      self.raiseAnError(IOError, 'Not found label ' + str(self.label) + ' in input dataObject!')
    outputDict['data'] ={}
    outputDict['dims'] = {}
    outputDict['metadata'] = copy.deepcopy(inputDict['metadata']) if 'metadata' in inputDict.keys() else {}
    labelType = type(inputDict['data'][self.label][0])
    if labelType != np.ndarray:
      indexes = np.where(np.in1d(inputDict['data'][self.label],self.clusterIDs))[0]
      for key in inputDict['data'].keys():
        outputDict['data'][key] = inputDict['data'][key][indexes]
        outputDict['dims'][key] = []
    else:
      for key in inputDict['data'].keys():
        if type(inputDict['data'][key][0]) == np.ndarray:
          temp = []
          for cnt in range(len(inputDict['data'][self.label])):
            indexes = np.where(np.in1d(inputDict['data'][self.label][cnt],self.clusterIDs))[0]
            if len(indexes) > 0:
              temp.append(copy.deepcopy(inputDict['data'][key][cnt][indexes]))
              if self.clusteringMethod == 'extended':
                temp.append(copy.deepcopy(inputDict['data'][key][cnt][indexes]))
              else:
                temp.append(copy.deepcopy(inputDict['data'][key][cnt]))
          outputDict['data'][key] = np.asanyarray(temp)
          outputDict['dims'][key] = []
        else:
          outputDict['data'][key] = np.empty(0)
          for cnt in range(len(inputDict['data'][self.label])):
            indexes = np.where(np.in1d(inputDict['data'][self.label][cnt],self.clusterIDs))[0]
            if len(indexes) > 0:
              outputDict['data'][key] = np.append(outputDict['data'][key], copy.deepcopy(inputDict['data'][key][cnt]))
            outputDict['dims'][key] = []
    return outputDict
