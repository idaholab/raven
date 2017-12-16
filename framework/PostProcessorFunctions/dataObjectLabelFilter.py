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

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


import os
import numpy as np
from scipy import interpolate
import copy


class dataObjectLabelFilter(PostProcessorInterfaceBase):
  """
   This Post-Processor filters out the points or histories accordingly to a chosen clustering label
  """
  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """

    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = None
    self.outputFormat = None

    self.label        = None
    self.clusterIDs   = []

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """

    for child in xmlNode:
      if child.tag == 'dataType':
        dataType = child.text
        if dataType in set(['HistorySet','PointSet']):
          self.inputFormat  = dataType
          self.outputFormat = dataType
        else:
          self.raiseAnError(IOError, 'dataObjectLabelFilter Interfaced Post-Processor ' + str(self.name) + ' : dataType ' + str(dataType) + ' is not recognized (available are HistorySet, PointSet)')
      elif child.tag == 'label':
        self.label = child.text
      elif child.tag == 'clusterIDs':
        for clusterID in child.text.split(','):
          clusterID = clusterID.strip()
          self.clusterIDs.append(int(clusterID))
      elif child.tag !='method':
        self.raiseAnError(IOError, 'dataObjectLabelFilter Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

  def run(self,inputDic):
    """
     Method to post-process the dataObjects
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputDic, dictionary, output dictionary to be provided to the base class
    """
    if len(inputDic)>1:
      self.raiseAnError(IOError, 'HistorySetSync Interfaced Post-Processor ' + str(self.name) + ' accepts only one dataObject')
    else:
      inputDict = inputDic[0]
      indexes = np.where(np.in1d(inputDic['data'][self.label],self.clusterIDs))[0]
      for key in inputDict['data'].keys():
        inputDict['data'][key] = inputDict['data'][key][indexes]
    return inputDict
