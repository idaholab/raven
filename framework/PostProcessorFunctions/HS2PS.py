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
from utils import InputData, InputTypes


class HS2PS(PostProcessorInterfaceBase):
  """
   This Post-Processor performs the conversion from HistorySet to PointSet
   The conversion is made so that each history H is converted to a single point P.
   Assume that each history H is a dict of n output variables x_1=[...],x_n=[...], then the resulting point P is as follows; P=[x_1,...,x_n]
   Note!!!! Here it is assumed that all histories have been sync so that they have the same length, start point and end point.
            If you are not sure, do a pre-processing the the original history set
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
    inputSpecification.addCheckedParams({"name":"HS2PS",
                                         "subType":"InterfacedPostProcessor"})
    inputSpecification.addSub(InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("features", contentType=InputTypes.StringListType))
    #Should method be in super class?
    inputSpecification.addSub(InputData.parameterInputFactory("method", contentType=InputTypes.StringType))
    return inputSpecification

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'PointSet'

    self.pivotParameter       = None
    #pivotParameter identify the ID of the temporal variable in the data set; it is used so that in the
    #conversion the time array is not inserted since it is not needed (all histories have same length)
    self.features     = 'all'


  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    paramInput = HS2PS.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'features':
        self.features = child.value
      elif child.getName() !='method':
        self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if self.pivotParameter == None:
      self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' : pivotParameter is not specified')

  def run(self,inputDic):
    """
    This method performs the actual transformation of the data object from history set to point set
      @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, output dictionary
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HS2PS Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDict = inputDic[0]
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
