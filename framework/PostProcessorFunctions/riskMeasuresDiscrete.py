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
Created on November 2016

@author: mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

class riskMeasuresDiscrete(PostProcessorInterfaceBase):
  """ This class implements the four basic risk-importance measures
      This class inherits form the base class PostProcessorInterfaceBase and it contains three methods:
      - initialize
      - run
      - readMoreXML
  """

  def initialize(self):
    """
      Method to initialize the Interfaced Post-processor
      @ In, None
      @ Out, None

    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'PointSet'
    self.outputFormat = 'PointSet'

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    self.variables = {}
    self.target    = {}

    self.IEdata = {}

    for child in xmlNode:
      if child.tag == 'measures':
        self.measures = child.text.split(',')

      elif child.tag == 'variable':
        variableID = child.text
        self.variables[variableID] = {}
        if 'R0values' in child.attrib.keys():
          values = child.attrib['R0values'].split(',')
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong R0values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R0low']  = min(val1,val2)
          self.variables[variableID]['R0high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R0 is not present for XML node: ' + str(child) )
        if 'R1values' in child.attrib.keys():
          values = child.attrib['R1values'].split(',')
          if len(values)>2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 for XML node: ' + str(child) + ' has more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong R1values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R1low']  = min(val1,val2)
          self.variables[variableID]['R1high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node R1 is not present for XML node: ' + str(child) )

      elif child.tag == 'target':
        self.target['targetID'] = child.text
        if 'values' in child.attrib.keys():
          values = child.attrib['values'].split(',')
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = float(values[0])
            val2 = float(values[1])
          except:
            self.raiseAnError(IOError,' Wrong target values associated to riskMeasuresDiscrete Post-Processor')
          self.target['low']  = min(val1,val2)
          self.target['high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : attribute node values is not present for XML node: ' + str(child) )

      elif child.tag == 'data':
        self.IEdata[child.text] = float(child.attrib['freq'])

      elif child.tag !='method':
        self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : XML node ' + str(child) + ' is not recognized')

    if not set(self.measures).issubset(['B','FV','RAW','RRW']):
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : measures '
                        + str(set(self.measures).issubset([B,FV,RAW,RRW])) + ' are not recognized')

  def run(self,inputDic):
    """
     This method perform the actual calculation of the risk measures
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputDic, dict, dictionary which contains the risk measures
    """
    riskImportanceMeasures = {}
    for variable in self.variables:
      macroR0 = 0
      macroRMinus = 0
      macroRPlus = 0

      r0Low = self.variables[variable]['R0low']
      r0High = self.variables[variable]['R0high']
      r1Low = self.variables[variable]['R1low']
      r1High = self.variables[variable]['R1high']

      for inp in inputDic:
        ## Get everything out of the inputDic at the outset, the hope is to have no string literals on the interior
        ## of this function.
        inputName = inp['name']
        inputDataIn = inp['data']['input']
        inputDataOut = inp['data']['output']
        targetVar = np.asarray(inputDataOut[self.target['targetID']])
        inputMetadata = inp['metadata'] if 'metadata' in inp else None

        if inputMetadata is not None and 'ProbabilityWeight' in inputMetadata:
          inputWeights = np.asarray(inp['metadata']['ProbabilityWeight'])
          pbWeights = inputWeights/np.sum(inputWeights)
        else:
          ## Any variable will do, so just count the first input. We could also have count the outputs, but this could
          ## be tricky if the data is a HistorySet and thus multidimensional.
          pointCount = len(inputDataIn.values()[0])
          pbWeights = np.ones(pointCount)/float(pointCount)

        if inputName in self.IEdata.keys():
          multiplier = self.IEdata[inputName]
        else:
          multiplier = 1.0
          self.raiseAWarning('RiskMeasuresDiscrete Interfaced Post-Processor: the dataObject ' + str (inputName) + ' does not have the frequency of the IE specified. It is assumed that the frequency of the IE is 1.0')

        ## Calculate R0, Rminus, Rplus

        ## Step 1: Retrieve points that contain system failure
        indexSystemFailure = np.where(np.logical_and(targetVar >= self.target['low'], targetVar <= self.target['high']))[0]

        if variable in inputDataIn.keys():
          inputVar = np.asarray(inputDataIn[variable])

          ## Step 2: Retrieve points from original dataset that contain component reliability values equal to 1
          ##         (indexComponentMinus) and 0 (indexComponentPlus)
          indexComponentMinus = np.where(np.logical_and(inputVar >= r1Low, inputVar <= r1High))[0]
          indexComponentPlus  = np.where(np.logical_and(inputVar >= r0Low, inputVar <= r0High))[0]

          ## Step 3: Retrieve points from Step 1 that contain component reliability values equal to 1
          ##         (indexFailureMinus) and 0 (indexFailurePlus)
          indexFailureMinus = np.intersect1d(indexSystemFailure,indexComponentMinus)
          indexFailurePlus  = np.intersect1d(indexSystemFailure,indexComponentPlus)

          ## Step 4: Sum pb weights for the subsets retrieved in Steps 1, 2, and 3
          ## R0 = pb of system failure
          R0     = np.sum(pbWeights[indexSystemFailure])
          ## Rminus = pb of system failure given component reliability is 1
          Rminus = np.sum(pbWeights[indexFailureMinus]) / np.sum(pbWeights[indexComponentMinus])
          ## Rplus = pb of system failure given component reliability is 0
          Rplus  = np.sum(pbWeights[indexFailurePlus]) / np.sum(pbWeights[indexComponentPlus])
        else:
          R0 = Rminus = Rplus = np.sum(pbWeights[indexSystemFailure])

        macroR0     += multiplier * R0
        macroRMinus += multiplier * Rminus
        macroRPlus  += multiplier * Rplus

      if 'RRW' in self.measures:
        RRW = riskImportanceMeasures[variable + '_RRW'] = np.asanyarray([macroR0/macroRMinus])
        self.raiseADebug(str(variable) + ' RRW = ' + str(RRW))

      if 'RAW' in self.measures:
        RAW = riskImportanceMeasures[variable + '_RAW'] = np.asanyarray([macroRPlus/macroR0])
        self.raiseADebug(str(variable) + ' RAW = ' + str(RAW))

      if 'FV' in self.measures:
        FV = riskImportanceMeasures[variable + '_FV']  = np.asanyarray([(macroR0-macroRMinus)/macroR0])
        self.raiseADebug( str(variable) + ' FV = ' + str(FV))

      if 'B' in self.measures:
        B = riskImportanceMeasures[variable + '_B']   = np.asanyarray([macroRPlus-macroRMinus])
        self.raiseADebug(str(variable) + ' B  = ' + str(B))

    outputDic = {
                  'data': {
                           'input': {},
                           'output': riskImportanceMeasures
                          },
                  'metadata': {}
                }
    ## If for whatever reason passing an empty input back causes errors, then you may want to add some sort of dummy
    ## value.
    # outputDic['data']['input'] = {} # {'dummy' : np.asanyarray(0)}

    return outputDic
