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
'''
Created on December 1, 2015

'''
from __future__ import division, print_function, unicode_literals, absolute_import

import copy
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
from utils import InputData, InputTypes

class testInterfacedPP_PointSet(PostProcessorInterfaceBase):
  """ This class represents the most basic interfaced post-processor
      This class inherits form the base class PostProcessorInterfaceBase and it contains the three methods that need to be implemented:
      - initialize
      - run
      - readMoreXML
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
    inputSpecification.addSubSimple("xmlNodeExample", InputTypes.StringType)
    inputSpecification.addSubSimple("method", InputTypes.StringType)
    return inputSpecification

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'PointSet'
    self.outputFormat = 'PointSet'

  def run(self,inputDic):
    """
    This method is transparent: it passes the inputDic directly as output
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, inputDic, dict, same inputDic dictionary

    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'testInterfacedPP_PointSet Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDict = inputDic[0]
      outputDict = {'data':{}}
      outputDict['dims'] = copy.deepcopy(inputDict['dims'])
      for key in inputDict['data'].keys():
        outputDict['data'][key] = copy.deepcopy(inputDict['data'][key])
      # add meta variables back
      for key in inputDict['metaKeys']:
        outputDict['data'][key] = inputDict['data'][key]
      return outputDict

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    paramInput = testInterfacedPP_PointSet.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == 'xmlNodeExample':
        self.xmlNodeExample = child.value
