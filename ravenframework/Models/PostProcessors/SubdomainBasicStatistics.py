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
Created on May 10, 2023

@author: aalfonsi
"""
#External Modules---------------------------------------------------------------
import numpy as np
import os
import copy
from collections import OrderedDict, defaultdict
import six
import xarray as xr
import scipy.stats as stats
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from .BasicStatistics import BasicStatistics
from ...utils import utils
from ...utils import InputData, InputTypes
from ...utils import mathUtils
#Internal Modules End-----------------------------------------------------------

class SubdomainBasicStatistics(PostProcessorInterface):
  """
    Subdomain basic statitistics class. It computes all statistics on subdomains
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
    # We get the input specs from the basic statistics and we just add
    # the subdomain info
    inputSpecification = BasicStatistics.getInputSpecification()

    subdomainInputs = InputData.parameterInputFactory("subdomain", printPriority=100,
              descr='defines the subdomain specs to be used for the subdomain statistics')
    variableInput = InputData.parameterInputFactory("variable", printPriority=80,
              descr="defines the variables to be used for the subdomain statistics.")
    variableInput.addParam("name", InputTypes.StringNoLeadingSpacesType,
        descr=r"""Name of the variable for this grid/subdomain. \nb As for the other objects,
              this is the name that can be used to refer to this specific entity from other input blocks""")
    gridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.StringType)
    gridInput.addParam("type", InputTypes.StringType)
    gridInput.addParam("construction", InputTypes.StringType)
    gridInput.addParam("steps", InputTypes.IntegerType)
    variableInput.addSub(gridInput)
    subdomainInputs.addSub(variableInput)
    inputSpecification.addSub(subdomainInputs)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    from ...Models.PostProcessors import factory as ppFactory # delay import to allow definition
    self.stat = ppFactory.returnInstance('BasicStatistics')
    self.validDataType  = ['PointSet', 'HistorySet', 'DataSet']
    self.printTag = 'PostProcessor SUBDOMAIN STATISTICS'

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding variable probability weight
    """
    # The BasicStatistics postprocessor only accept DataObjects
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    pbWeights = None
    if type(currentInput).__name__ == 'tuple':
      return currentInput
    # TODO: convert dict to dataset, I think this will be removed when DataSet is used by other entities that
    # are currently using this Basic Statisitics PostProcessor.
    if type(currentInput).__name__ == 'dict':
      if 'targets' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets in the input dictionary')
      inputDataset = xr.Dataset()
      for var, val in currentInput['targets'].items():
        inputDataset[var] = val
      if 'metadata' in currentInput.keys():
        metadata = currentInput['metadata']
        self.pbPresent = True if 'ProbabilityWeight' in metadata else False
        if self.pbPresent:
          pbWeights = xr.Dataset()
          self.realizationWeight = xr.Dataset()
          self.realizationWeight['ProbabilityWeight'] = metadata['ProbabilityWeight']/metadata['ProbabilityWeight'].sum()
          for target in self.parameters['targets']:
            pbName = 'ProbabilityWeight-' + target
            if pbName in metadata:
              pbWeights[target] = metadata[pbName]/metadata[pbName].sum()
            elif self.pbPresent:
              pbWeights[target] = self.realizationWeight['ProbabilityWeight']
        else:
          self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
      else:
        self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
      if 'RAVEN_sample_ID' not in inputDataset.sizes.keys():
        self.raiseAWarning('BasicStatisitics postprocessor did not detect RAVEN_sample_ID! Assuming the first dimension of given data...')
        self.sampleTag = utils.first(inputDataset.sizes.keys())
      return inputDataset, pbWeights

    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    try:
      inputDataset = dataSet[self.parameters['targets']]
    except KeyError:
      missing = [var for var in self.parameters['targets'] if var not in dataSet]
      self.raiseAnError(KeyError, "Variables: '{}' missing from dataset '{}'!".format(", ".join(missing),currentInput.name))
    self.sampleTag = currentInput.sampleTag

    if currentInput.type == 'HistorySet':
      dims = inputDataset.sizes.keys()
      if self.pivotParameter is None:
        if len(dims) > 1:
          self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter \
                got inputted!')
      elif self.pivotParameter not in dims:
        self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter, 'is not the associated index for \
                requested variables', ','.join(self.parameters['targets']))
      else:
        self.dynamic = True
        if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
          self.raiseAnError(IOError, "The data provided by the data objects", currentInput.name, "is not synchronized!")
        self.pivotValue = inputDataset[self.pivotParameter].values
        if self.pivotValue.size != len(inputDataset.groupby(self.pivotParameter)):
          msg = "Duplicated values were identified in pivot parameter, please use the 'HistorySetSync'" + \
          " PostProcessor to syncronize your data before running 'BasicStatistics' PostProcessor."
          self.raiseAnError(IOError, msg)
    # extract all required meta data
    metaVars = currentInput.getVars('meta')
    self.pbPresent = True if 'ProbabilityWeight' in metaVars else False
    if self.pbPresent:
      pbWeights = xr.Dataset()
      self.realizationWeight = dataSet[['ProbabilityWeight']]/dataSet[['ProbabilityWeight']].sum()
      for target in self.parameters['targets']:
        pbName = 'ProbabilityWeight-' + target
        if pbName in metaVars:
          pbWeights[target] = dataSet[pbName]/dataSet[pbName].sum()
        elif self.pbPresent:
          pbWeights[target] = self.realizationWeight['ProbabilityWeight']
    else:
      self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')

    return inputDataset, pbWeights

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the BasicStatistic pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    self.stat.intialize(runInfo, inputs, initDict)


  def _handleInput(self, paramInput, childVals=None):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ In, childVals, list, optional, quantities requested from child statistical object
      @ Out, None
    """
    # initialize basic stats
    subdomain = paramInput.popSub('subdomain')
    if subdomain is None:
      self.raiseAnError(IOError,'<subdomain> tag not found!')
    self.stat._handleInput(paramInput, childVals)

    for child in subdomain.subparts:
      if child.tag == 'variable':
        varName = child.parameterValues['name']
        for childChild in child.subparts:




        # variable for tracking if distributions or functions have been declared
        foundDistOrFunc = False
        # store variable name for re-use
        varName = child.parameterValues['name']
        # set shape if present
        if 'shape' in child.parameterValues:
          self.variableShapes[varName] = child.parameterValues['shape']
        # read subnodes
        for childChild in child.subparts:
          if childChild.getName() == 'distribution':
            # can only have a distribution if doesn't already have a distribution or function
            if foundDistOrFunc:
              self.raiseAnError(IOError, 'A sampled variable cannot have both a distribution and a function, or more than one of either!')
            else:
              foundDistOrFunc = True
            # name of the distribution to sample
            toBeSampled = childChild.value
            varData = {}
            varData['name'] = childChild.value
            # variable dimensionality
            if 'dim' not in childChild.parameterValues:
              dim = 1
            else:
              dim = childChild.parameterValues['dim']
            varData['dim'] = dim
            # set up mapping for variable to distribution
            self.variables2distributionsMapping[varName] = varData
            # flag distribution as needing to be sampled
            self.toBeSampled[prefix + varName] = toBeSampled
          elif childChild.getName() == 'function':
            # can only have a function if doesn't already have a distribution or function
            if not foundDistOrFunc:
              foundDistOrFunc = True
            else:
              self.raiseAnError(IOError, 'A sampled variable cannot have both a distribution and a function!')
            # function name
            toBeSampled = childChild.value
            # track variable as a functional sample
            self.dependentSample[prefix + varName] = toBeSampled


      else:
        if tag not in childVals:
          self.raiseAWarning('Unrecognized node in BasicStatistics "',tag,'" has been ignored!')









  def __runLocal(self, inputData):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In, inputData, tuple,  (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding
        variable probability weight
      @ Out, outputSet or outputDict, xarray.Dataset or dict, dataset or dictionary containing the results
    """
    inputDataset, pbWeights = inputData[0], inputData[1]


    return outputDict





  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputSet, xarray.Dataset or dictionary, dataset or dictionary containing the results
    """
    inputData = self.inputToInternal(inputIn)
    outputSet = self.__runLocal(inputData)
    return outputSet

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    super().collectOutput(finishedJob, output)
