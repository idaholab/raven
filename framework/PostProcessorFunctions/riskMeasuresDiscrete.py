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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
#External Modules End--------------------------------------------------------------------------------

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
from utils import InputData, InputTypes

class riskMeasuresDiscrete(PostProcessorInterfaceBase):
  """
    This class implements the four basic risk-importance measures
    This class inherits form the base class PostProcessorInterfaceBase and it contains three methods:
      - initialize
      - run
      - readMoreXML
  """
  _availableMeasures = set(['B','FV','RAW','RRW','R0'])
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
    inputSpecification.addSubSimple("measures", InputTypes.StringListType)
    variableSub = InputData.parameterInputFactory("variable", contentType=InputTypes.StringType)
    variableSub.addParam("R0values", InputTypes.FloatListType)
    variableSub.addParam("R1values", InputTypes.FloatListType)
    inputSpecification.addSub(variableSub)
    targetSub = InputData.parameterInputFactory("target", contentType=InputTypes.StringType)
    targetSub.addParam("values", InputTypes.FloatListType)
    inputSpecification.addSub(targetSub)
    dataSub = InputData.parameterInputFactory("data")
    dataSub.addParam("freq", InputTypes.FloatType)
    inputSpecification.addSub(dataSub)
    inputSpecification.addSubSimple("temporalID", InputTypes.StringType)
    #Should method be in super class?
    inputSpecification.addSubSimple("method", contentType=InputTypes.StringType)
    return inputSpecification


  def availableMeasures(cls):
    """
      A class level constant that tells developers what measures are available from this class
      @ In, cls, the RiskMeasureDiscrete class of which this object will be a type
    """
    return cls._availableMeasures

  def initialize(self):
    """
      Method to initialize the Interfaced Post-processor
      @ In, None
      @ Out, None
    """
    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'PointSet|HistorySet'
    self.outputFormat = 'PointSet'

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """

    paramInput = riskMeasuresDiscrete.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    self.variables = {}
    self.target    = {}
    self.IEData = {}
    self.temporalID = None

    for child in paramInput.subparts:
      if child.getName() == 'measures':
        self.measures = set(child.value)

        if not self.measures.issubset(self.availableMeasures()):
          unrecognizedMeasures  = self.measures.difference(self.availableMeasures())
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) + ' : measures '
                            + str(list(unrecognizedMeasures)) + ' are not recognized')

      elif child.getName() == 'variable':
        variableID = child.value
        self.variables[variableID] = {}
        if 'R0values' in child.parameterValues:
          values = child.parameterValues['R0values']
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                              ' : attribute node R0 for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = values[0]
            val2 = values[1]
          except:
            self.raiseAnError(IOError,' Wrong R0values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R0low']  = min(val1,val2)
          self.variables[variableID]['R0high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                            ' : attribute node R0 is not present for XML node: ' + str(child) )
        if 'R1values' in child.parameterValues:
          values = child.parameterValues['R1values']
          if len(values)>2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                              ' : attribute node R1 for XML node: ' + str(child) + ' has more than two values')
          try:
            val1 = values[0]
            val2 = values[1]
          except:
            self.raiseAnError(IOError,' Wrong R1values associated to riskMeasuresDiscrete Post-Processor')
          self.variables[variableID]['R1low']  = min(val1,val2)
          self.variables[variableID]['R1high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                            ' : attribute node R1 is not present for XML node: ' + str(child) )

      elif child.getName() == 'target':
        self.target['targetID'] = child.value
        if 'values' in child.parameterValues:
          values = child.parameterValues['values']
          if len(values) != 2:
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                              ' : attribute node values for XML node: ' + str(child) + ' has one or more than two values')
          try:
            val1 = values[0]
            val2 = values[1]
          except:
            self.raiseAnError(IOError,' Wrong target values associated to riskMeasuresDiscrete Post-Processor')
          self.target['low']  = min(val1,val2)
          self.target['high'] = max(val1,val2)
        else:
          self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                            ' : attribute node values is not present for XML node: ' + str(child) )

      elif child.getName() == 'data':
        self.IEData[child.value] = child.parameterValues['freq']

      elif child.getName() == 'temporalID':
        self.temporalID = child.value

      elif child.getName() !='method':
        self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                          ' : XML node ' + str(child) + ' is not recognized')

  def run(self,inputDic):
    """
     This method perform the actual calculation of the risk measures
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ Out, outputDic, dict, dictionary which contains the risk measures
    """
    # Check how many HistorySets (checkHSs) have been provided
    checkHSs=0
    for inp in inputDic:
      if inp['type'] == 'HistorySet':
        timeDepData = copy.deepcopy(inp)
        inputDic.remove(inp)
        checkHSs +=1

    if checkHSs == 0:
      # if no HistorySet has been provided run the static form of this PP
      outputDic = self.runStatic(inputDic)
      outputDic['dims'] = {}
      for key in outputDic['data'].keys():
        outputDic['dims'][key] = []
    elif checkHSs == 1:
       # if one HistorySet has been provided run the dynamic form of this PP
      self.outputFormat = 'HistorySet'
      outputDic = self.runDynamic(inputDic,timeDepData)
      outputDic['dims'] = {}
      for key in outputDic['data'].keys():
        outputDic['dims'][key] = [self.temporalID]
      for var in timeDepData['data'].keys():
        if var != self.temporalID:
          ## Are there any values in timeDepData['data']['output'][1][var] that are not 0 or 1?
          if len(np.setdiff1d(timeDepData['data'][var][0], [0,1])):
            self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                              ' : the provided HistorySet contains the variable ' + str(var) + ' which has elements different than 0 or 1')
      outputDic['data'][self.temporalID] = np.zeros(1, dtype=object)
      outputDic['data'][self.temporalID][0] = copy.deepcopy(timeDepData['data'][self.temporalID][0])

      for var in timeDepData['inpVars']:
        outputDic['data'][var] = copy.deepcopy(timeDepData['data'][var])
    else: # checkHSs >= 2:
      # only one HistorySet should be provided
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                        ' : more than one HistorySet has been provided')

    # replicate metadata
    # add meta variables back
    for key in inputDic[-1]['metaKeys']:
      outputDic['data'][key] = np.asanyarray(1.0)

    return outputDic

  def runStatic(self,inputDic, componentConfig=None):
    """
     This method perform the static calculation of the risk measures
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ In, componentConfig, dict, dictionary containing the boolean status (0 or 1) of a sub set of the input variables
     @ Out, outputDic, dict, dictionary which contains the risk measures
    """
    if self.temporalID is not None and componentConfig is None:
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor ' + str(self.name) +
                        ' : a temporalID variable is specified but an HistorySet is not provided')

    riskImportanceMeasures = {}
    for variable in self.variables:
      macroR0     = 0
      macroRMinus = 0
      macroRPlus  = 0

      r0Low  = self.variables[variable]['R0low']
      r0High = self.variables[variable]['R0high']
      r1Low  = self.variables[variable]['R1low']
      r1High = self.variables[variable]['R1high']

      for inp in inputDic:
        ## Get everything out of the inputDic at the outset, the hope is to have no string literals on the interior
        ## of this function.
        if componentConfig is None:
          inputName     = inp['name']
          inputDataIn   = {key: inp['data'][key] for key in inp['inpVars']}
          inputDataOut  = {key: inp['data'][key] for key in inp['outVars']}
          targetVar     = np.asarray(inp['data'][self.target['targetID']])
          inputMetadata = {}
          inputMetadata['ProbabilityWeight'] = inp['data']['ProbabilityWeight']
        else:
          # if componentConfig is provided, then only a subset of the original data must be considered
          # only the data points that contains componentConfig are in fact considered
          # indexUpdatedData contains the indexes of those data points
          indexUpdatedData = None

          for var in componentConfig.keys():
            if componentConfig[var] == 0:
              inputVar = np.asarray(inp['data'][var])
              indexCompOut = np.where(inputVar==1)
              if indexUpdatedData is None:
                indexUpdatedData = copy.deepcopy(indexCompOut)
              else:
                indexUpdatedData = np.intersect1d(indexUpdatedData,indexCompOut)

          if indexUpdatedData is not None:
            inputName    = inp['name']
            inputDataIn  = {}
            inputDataOut = {}
            for var in inp['inpVars']:
              inputDataIn[var]  = inp['data'][var][indexUpdatedData]
            for var in inp['outVars']:
              inputDataOut[var] = inp['data'][var][indexUpdatedData]
            targetVar = np.asarray(inputDataOut[self.target['targetID']])
            inputMetadata = {}
            inputMetadata['ProbabilityWeight'] = inp['data']['ProbabilityWeight'][indexUpdatedData]
          else:
            inputName     = inp['name']
            inputDataIn   = {key: inp['data'][key] for key in inp['inpVars']} #inp['data']['input']
            inputDataOut  = {key: inp['data'][key] for key in inp['outVars']} #inp['data']['output']
            targetVar     = np.asarray(inputDataOut[self.target['targetID']])
            inputMetadata = {}
            inputMetadata['ProbabilityWeight'] = inp['data']['ProbabilityWeight']

        if inputMetadata is not None and 'ProbabilityWeight' in inputMetadata:
          inputWeights = copy.deepcopy(inputMetadata['ProbabilityWeight'])
          pbWeights = inputWeights/np.sum(inputWeights)
        else:
          ## Any variable will do, so just count the first input. We could also have count the outputs, but this could
          ## be tricky if the data is a HistorySet and thus multidimensional.
          pointCount = inp['numberRealizations']
          pbWeights  = np.ones(pointCount)/float(pointCount)

        if inputName in self.IEData.keys():
          multiplier = self.IEData[inputName]
        else:
          multiplier = 1.0
          self.raiseAWarning('RiskMeasuresDiscrete Interfaced Post-Processor: the dataObject '
                             + str (inputName) + ' does not have the frequency of the IE specified. It is assumed that the frequency of the IE is 1.0')

        ## Calculate R0, RMinus, RPlus

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
          if componentConfig is None or variable not in componentConfig.keys():
            # Coordinate BE2
            ## R0 = pb of system failure
            R0     = np.sum(pbWeights[indexSystemFailure])
            RMinus = np.sum(pbWeights[indexFailureMinus]) / np.sum(pbWeights[indexComponentMinus])
            RPlus  = np.sum(pbWeights[indexFailurePlus]) / np.sum(pbWeights[indexComponentPlus])
          else:
            # Coordinate BE3
            R0 = np.sum(pbWeights[indexSystemFailure])
            if   componentConfig[variable] == 0:
              RMinus = RPlus = np.sum(pbWeights[indexFailurePlus]) / np.sum(pbWeights[indexComponentPlus])
            elif componentConfig[variable] == 1:
              if indexFailureMinus.size:
                RMinus = np.sum(pbWeights[indexFailureMinus]) / np.sum(pbWeights[indexComponentMinus])
              else:
                RMinus = R0
              if indexComponentPlus.size:
                RPlus  = np.sum(pbWeights[indexFailurePlus]) / np.sum(pbWeights[indexComponentPlus])
              else:
                RPlus = R0
        else:
          # Coordinate BE1
          R0 = RMinus = RPlus = np.sum(pbWeights[indexSystemFailure])

        macroR0     += multiplier * R0
        macroRMinus += multiplier * RMinus
        macroRPlus  += multiplier * RPlus

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

      if 'R0' in self.measures:
        riskImportanceMeasures['R0']   = np.asanyarray([macroR0])
        self.raiseADebug(' R0  = ' + str(macroR0))

    outputDic = {'data': riskImportanceMeasures}

    ## If for whatever reason passing an empty input back causes errors, then you may want to add some sort of dummy
    ## value.
    # outputDic['data']['input'] = {} # {'dummy' : np.asanyarray(0)}

    return outputDic

  def runDynamic(self,inputDic,timeHistory):
    """
     This method performs the dynamic calculation of the risk measures
     FIXME - Note for new development: clean this part so that the [0] index is removed from the timeHistory
     RECALL - The gold files for this PP are theoretical values and hence they should be part of the analytical tests
     @ In, inputDic, list, list of dictionaries which contains the data inside the input DataObjects
     @ In, timeHistory, dict, dictionary containing  boolean temporal profiles (0 or 1) of a sub set of the input variables. Note that this
                              history must contain a single history
     @ Out, outputDic, dict, dictionary which contains the risk measures
    """
    # timeHistory format values:
    # - 0 component is disconnected
    # - 1 component is connected

    if self.temporalID is None:
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor: an HistorySet is provided but no temporalID variable is specified')

    if self.temporalID not in timeHistory['data'].keys():
      self.raiseAnError(IOError, 'RiskMeasuresDiscrete Interfaced Post-Processor: the specified temporalID variable '
                        + str(self.temporalID) + ' is not part of the HistorySet variables')

    outputDic = {}
    outputDic['data'] = {}

    for measure in self.measures:
      if measure=='R0':
        outputDic['data'][measure] = np.zeros(1, dtype=object)
        outputDic['data'][measure][0] = np.zeros(len(timeHistory['data'][self.temporalID][0]))
      else:
        for var in self.variables:
          outputDic['data'][var + '_' + measure]    = np.zeros(1, dtype=object)
          outputDic['data'][var + '_' + measure][0] = np.zeros(len(timeHistory['data'][self.temporalID][0]))

    previousSystemConfig = {}
    for index,value in enumerate(timeHistory['data'][self.temporalID][0]):
      systemConfig={}
      # Retrieve the system configuration at time instant "index"
      for var in timeHistory['outVars']:
        if var != self.temporalID:
          systemConfig[var] = timeHistory['data'][var][0][index]
      # Do not repeat the calculation if the system configuration is identical to the one of previous time instant
      if systemConfig == previousSystemConfig:
        for key in outputDic['data'].keys():
          outputDic['data'][key][0][index] = outputDic['data'][key][0][index-1]
      else:
        staticOutputDic = self.runStatic(inputDic,systemConfig)
        for key in outputDic['data'].keys():
          outputDic['data'][key][0][index] = staticOutputDic['data'][key][0]
      previousSystemConfig = copy.deepcopy(systemConfig)
    return outputDic
