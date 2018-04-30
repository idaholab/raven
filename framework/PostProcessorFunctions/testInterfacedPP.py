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
import warnings
warnings.simplefilter('default',DeprecationWarning)

import copy
import itertools
import numpy as np

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

class testInterfacedPP(PostProcessorInterfaceBase):
  """ This class represents the most basic interfaced post-processor
      This class inherits form the base class PostProcessorInterfaceBase and it contains the three methods that need to be implemented:
      - initialize
      - run
      - readMoreXML
  """

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,
    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'

  def run(self,inputDic):
    """
     This method is transparent: it passes the inputDic directly as output
     @ In, inputDic, dict, dictionary which contains the data inside the input DataObject
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
      return outputDict

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'xmlNodeExample':
        self.xmlNodeExample = child.text
