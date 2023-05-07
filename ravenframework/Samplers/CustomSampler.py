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
  This module contains the Custom sampling strategy

  Created on May 21, 2016
  @author: alfoa
"""
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import copy
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .Sampler import Sampler
from ..utils import InputData, InputTypes, utils, mathUtils
# Internal Modules End------------------------------------------------------------------------------

class CustomSampler(Sampler):
  """
    Custom Sampler
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
    inputSpecification = super(CustomSampler, cls).getInputSpecification()
    sourceInput = InputData.parameterInputFactory("Source", contentType=InputTypes.StringType)
    sourceInput.addParam("type", InputTypes.StringType)
    sourceInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(sourceInput)

    inputSpecification.addSub(InputData.parameterInputFactory('index', contentType=InputTypes.IntegerListType))

    # add "nameInSource" attribute to <variable>
    var = inputSpecification.popSub('variable')
    var.addParam("nameInSource", InputTypes.StringType, required=False)
    inputSpecification.addSub(var)
    inputSpecification.addSub(InputData.parameterInputFactory('batch', contentType=InputTypes.IntegerType))

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.pointsToSample = {}
    self.infoFromCustom = {}
    self.nameInSource = {} # dictionary to map the variable's sampled name to the name it has in Source
    self.addAssemblerObject('Source', InputData.Quantity.one)
    self.printTag = 'SAMPLER CUSTOM'
    self.readingFrom = None # either File or DataObject, determines sample generation
    self.indexes = None
    self.sourceIndexMap = {}
    self.batch = 1    # number of samples in each batch
    self.batchId = 0  # ID for each batch

  def _readMoreXMLbase(self, xmlNode):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    # TODO remove using xmlNode
    self.readSamplerInit(xmlNode)
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    self.nameInSource = {}
    for child in paramInput.subparts:
      if child.getName() == 'variable':
        # acquire name
        name = child.parameterValues['name']
        # check for an "alias" source name
        self.nameInSource[name] = child.parameterValues.get('nameInSource',name)
        # determine if a sampling function is used
        funct = child.findFirst("function")
        if funct is None:
          # custom samples use a "custom" distribution
          self.toBeSampled[name] = 'custom'
        else:
          self.dependentSample[name] = funct.value

      elif child.getName() == 'constant':
        name, value = self._readInConstant(child)
        self.constants[name] = value

      elif child.getName() == 'Source':
        okaySourceClasses = ['Files', 'DataObjects']
        sourceClass = child.parameterValues.get('class')
        if sourceClass not in okaySourceClasses:
          self.raiseAnError(IOError, (f'For CustomSampler "{self.name}" node "<Source>" with attribute ' +
                                      f'"class", received "{sourceClass}" but must be one of {okaySourceClasses}!'))
      elif child.getName() == 'index':
        self.indexes = child.value
      elif child.getName() == 'batch':
        self.batch = max(child.value, 1)
    if len(self.toBeSampled.keys()) == 0:
      self.raiseAnError(IOError, f'CustomSampler "{self.name}" has no variables to sample!')

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed (in this case it is empty, since no distribtions are needed and the Source is loaded automatically)
    """
    needDict = {}
    needDict['Functions'] = [] # In case functions have been inputted
    for func in self.dependentSample.values():
      needDict['Functions'].append((None, func))

    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    #it is called for the ensemble sampler
    for key, value in self.assemblerObjects.items():
      if key == 'Source':
        self.assemblerDict[key] =  []
        for entity, etype, name in value:
          self.assemblerDict[key].append([entity, etype, name, initDict[entity][name]])
    for key, val in self.dependentSample.items():
      if val not in initDict['Functions'].keys():
        self.raiseAnError(f'Function {val} was not found among the available functions: {initDict["Functions"].keys()}')
      self.funcDict[key] = initDict['Functions'][val]
      # check if the correct method is present
      if "evaluate" not in self.funcDict[key].availableMethods():
        self.raiseAnError(IOError, f'Function {self.funcDict[key].name} does not contain a method named "evaluate". It must be present if this needs to be used in a Sampler!')

    if 'Source' not in self.assemblerDict:
      self.raiseAnError(IOError, "No Source object has been found!")

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler.
      @ In, None
      @ Out, None
    """
    # check the source
    if self.assemblerDict['Source'][0][0] == 'Files':
      self.readingFrom = 'File'
      csvFile = self.assemblerDict['Source'][0][3]
      csvFile.open(mode='r')
      headers = [x.replace("\n","").strip() for x in csvFile.readline().split(",")]
      data = np.loadtxt(self.assemblerDict['Source'][0][3], dtype=np.float64, delimiter=',', skiprows=1, ndmin=2)
      lenRlz = len(data)
      csvFile.close()
      for var in self.toBeSampled:
        for subVar in var.split(','):
          subVar = subVar.strip()
          sourceName = self.nameInSource[subVar]
          if sourceName not in headers:
            self.raiseAnError(IOError, f"variable {sourceName} not found in the file {csvFile.getFilename()}")
          self.pointsToSample[subVar] = data[:,headers.index(sourceName)]
          subVarPb = 'ProbabilityWeight-'
          if subVarPb+sourceName in headers:
            self.infoFromCustom[subVarPb+subVar] = data[:, headers.index(subVarPb+sourceName)]
          else:
            self.infoFromCustom[subVarPb+subVar] = np.ones(lenRlz)
      if 'PointProbability' in headers:
        self.infoFromCustom['PointProbability'] = data[:, headers.index('PointProbability')]
      else:
        self.infoFromCustom['PointProbability'] = np.ones(lenRlz)
      if 'ProbabilityWeight' in headers:
        self.infoFromCustom['ProbabilityWeight'] = data[:, headers.index('ProbabilityWeight')]
      else:
        self.infoFromCustom['ProbabilityWeight'] = np.ones(lenRlz)

      self.limit = len(utils.first(self.pointsToSample.values()))
    else:
      self.readingFrom = 'DataObject'
      dataObj = self.assemblerDict['Source'][0][3]
      lenRlz = len(dataObj)
      self.pointsToSample = dataObj.sliceByIndex(dataObj.sampleTag)
      for var in self.toBeSampled:
        for subVar in var.split(','):
          subVar = subVar.strip()
          sourceName = self.nameInSource[subVar]
          if sourceName not in dataObj.getVars() + dataObj.getVars('indexes'):
            self.raiseAnError(IOError, f"the variable {sourceName} not found in {dataObj.type} {dataObj.name}")
      self.limit = len(self.pointsToSample)
      self.sourceIndexMap = dataObj.getDimensions()
    # if "index" provided, limit sampling to those points
    if self.indexes is not None:
      self.limit = len(self.indexes)
      maxIndex = max(self.indexes)
      if maxIndex > len(self.pointsToSample) - 1:
        self.raiseAnError(IndexError, f'Requested index "{maxIndex}" from custom sampler, but highest index sample is "{len(self.pointsToSample) - 1}"!')
    # TODO: add restart capability here!
    if self.restartData:
      self.raiseAnError(IOError, "restart capability not implemented for CustomSampler yet!")
    if self.batch > 1:
      self.addMetaKeys(["batchId"])

  def localGenerateInput(self, model, myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    if self.batch > 1:
      self.inputInfo['batchMode'] = True
      batchData = []
      self.batchId += 1
    else:
      self.inputInfo['batchMode'] = False
    for _ in range(self.batch):
      if self.indexes is None:
        index = self.counter - 1
      else:
        index = self.indexes[self.counter-1]
      if self.counter == self.limit + 1:
        break
      if self.readingFrom == 'DataObject':
        # data is stored as slices of a data object, so take from that
        rlz = self.pointsToSample[index]
        for var in self.toBeSampled:
          for subVar in var.split(','):
            subVar = subVar.strip()
            sourceName = self.nameInSource[subVar]
            # get the value(s) for the variable for this realization
            self.values[subVar] = mathUtils.npZeroDToEntry(rlz[sourceName].values)
            # get supporting indices (e.g. 'time')
            for dim in rlz.dims:
              if dim not in self.values:
                self.values[dim] = rlz[self.nameInSource.get(dim, dim)]
            # set the probability weight due to this variable (default to 1)
            pbWtName = 'ProbabilityWeight-'
            self.inputInfo[pbWtName+subVar] = rlz.get(pbWtName+sourceName,1.0)
        # get realization-level required meta information, or default to 1
        for meta in ['PointProbability', 'ProbabilityWeight']:
          self.inputInfo[meta] = rlz.get(meta, 1.0)
      elif self.readingFrom == 'File':
        # data is stored in file, so we already parsed the values
        # create values dictionary
        for var in self.toBeSampled:
          for subVar in var.split(','):
            subVar = subVar.strip()
            # assign the custom sampled variables values to the sampled variables
            self.values[subVar] = self.pointsToSample[subVar][index]
            # This is the custom sampler, assign the ProbabilityWeights based on the provided values
            self.inputInfo[f'ProbabilityWeight-{subVar}'] = self.infoFromCustom[f'ProbabilityWeight-{subVar}'][index]
        # Construct probabilities based on the user provided information
        self.inputInfo['PointProbability'] = self.infoFromCustom['PointProbability'][index]
        self.inputInfo['ProbabilityWeight'] = self.infoFromCustom['ProbabilityWeight'][index]
      self.values['_indexMap'] = self.sourceIndexMap
      self.inputInfo['SamplerType'] = 'Custom'
      if self.inputInfo['batchMode']:
        self.inputInfo['SampledVars'] = self.values
        self.inputInfo['batchId'] = self.name + str(self.batchId)
        batchData.append(copy.deepcopy(self.inputInfo))
        self._incrementCounter()
    if self.inputInfo['batchMode']:
      self.inputInfo['batchInfo'] = {'nRuns': self.batch, 'batchRealizations': batchData, 'batchId': self.name + str(self.batchId)}

  def flush(self):
    """
      Reset Sampler attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.infoFromCustom = {}
    self.pointsToSample = {}
    self.readingFrom = None
