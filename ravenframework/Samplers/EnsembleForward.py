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
  This module contains the Ensemble Forward sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
import copy
from operator import mul
from functools import reduce
from collections import namedtuple

from ..utils import InputData, InputTypes
from .Sampler               import Sampler
from .MonteCarlo            import MonteCarlo
from .Grid                  import Grid
from .Stratified            import Stratified
from .FactorialDesign       import FactorialDesign
from .ResponseSurfaceDesign import ResponseSurfaceDesign
from .CustomSampler         import CustomSampler
from .. import GridEntities
from ..Realizations import Realization

class EnsembleForward(Sampler):
  """
    Ensemble Forward sampler. This sampler is aimed to combine Forward Sampling strategies
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
    inputSpecification = super(EnsembleForward, cls).getInputSpecification()

    inputSpecification.addSub(MonteCarlo.getInputSpecification())
    inputSpecification.addSub(Grid.getInputSpecification())
    inputSpecification.addSub(Stratified.getInputSpecification())
    inputSpecification.addSub(FactorialDesign.getInputSpecification())
    inputSpecification.addSub(ResponseSurfaceDesign.getInputSpecification())
    inputSpecification.addSub(CustomSampler.getInputSpecification())

    samplerInitInput = InputData.parameterInputFactory("samplerInit")

    samplerInitInput.addSub(InputData.parameterInputFactory("initialSeed", contentType=InputTypes.IntegerType))

    inputSpecification.addSub(samplerInitInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Sampler.__init__(self)
    self.acceptableSamplers   = ['MonteCarlo',
                                 'Stratified',
                                 'Grid',
                                 'FactorialDesign',
                                 'ResponseSurfaceDesign',
                                 'CustomSampler']
    self.printTag             = 'SAMPLER EnsembleForward'
    self.instantiatedSamplers = {}
    self.samplersCombinations = {}
    self.dependentSample      = {}
    self.gridEnsemble         = None

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    # TODO remove using xmlNode
    # this import happens here because a recursive call is made if we attempt it in the header, but still really bad practice
    from .Factory import factory
    for child in xmlNode:
      #sampler initialization
      if child.tag == 'samplerInit':
        Sampler.readSamplerInit(self,xmlNode)
      # read in samplers
      elif child.tag in self.acceptableSamplers:
        child.attrib['name'] = child.tag
        self.instantiatedSamplers[child.tag] = factory.returnInstance(child.tag)
        # FIXME the variableGroups needs to be fixed
        self.instantiatedSamplers[child.tag].readXML(child, variableGroups={}, globalAttributes=self.globalAttributes)
        # fill toBeSampled so that correct check for samplable variables occurs
        self.toBeSampled.update(self.instantiatedSamplers[child.tag].toBeSampled)
      # function variables are defined outside the individual samplers
      elif child.tag=='variable':
        for childChild in child:
          if childChild.tag == 'function':
            self.dependentSample[child.attrib['name']] = childChild.text
          else:
            self.raiseAnError(IOError, f"Variable {child.attrib['name']} must be defined by a function since it is located outside the samplers block")
      # constants are handled in the base class
      elif child.tag == 'constant':
        pass
      # some samplers aren't eligible for ensembling
      elif child.tag in factory.knownTypes():
        self.raiseAnError(IOError, f'Sampling strategy "{child.tag}" is not usable in "{self.type}".  Available options include: {", ".join(self.acceptableSamplers)}.')
      # catch-all for bad inputs
      else:
        self.raiseAnError(IOError, f'Unrecognized sampling strategy: "{child.tag}". Available options include: {", ".join(self.acceptableSamplers)}.')

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented here because this Sampler requests special objects
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed
    """
    # clear out toBeSampled, since Sampler uses it for assembling
    self.toBeSampled = {}
    needDict = Sampler._localWhatDoINeed(self)
    for combSampler in self.instantiatedSamplers.values():
      preNeedDict = combSampler.whatDoINeed()
      for key, value in preNeedDict.items():
        if key not in needDict:
          needDict[key] = []
        needDict[key] = needDict[key] + value

    return needDict

  def _localGenerateAssembler(self, initDict):
    """
      It is used for sending to the instantiated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    for combSampler in self.instantiatedSamplers.values():
      if combSampler.type != 'CustomSampler':
        combSampler._generateDistributions(availableDist, availableFunc)
      combSampler._localGenerateAssembler(initDict)
    self.raiseADebug("Distributions initialized!")

    for key, val in self.dependentSample.items():
      if val not in availableFunc:
        self.raiseAnError(IOError, f'Function {val} was not found among the available functions: {availableFunc.keys()}')
      fPointer = namedtuple("func", ['methodName', 'instance'])
      mName = 'evaluate'
      # check if the correct method is present
      if val not in availableFunc[val].availableMethods():
        if "evaluate" not in availableFunc[val].availableMethods():
          self.raiseAnError(IOError, f'Function {availableFunc[val].name} does contain neither a method named "{val}" nor "evaluate". '
                            'It must be present if this needs to be used in a Sampler!')
      else:
        mName = val
      self.funcDict[key] = fPointer(mName, availableFunc[val])

    # evaluate function order in custom sampler
    self._evaluateFunctionsOrder()

  def localInitialize(self):
    """
      Initialize the EnsembleForward sampler. It calls the localInitialize method of all the Samplers defined in this input
      @ In, None
      @ Out, None
    """
    self.limits['samples'] = 1
    cnt = 0
    lowerBounds = {}
    upperBounds = {}
    metadataKeys = []
    metaParams = {}
    for samplingStrategy, sampler in self.instantiatedSamplers.items():
      sampler.initialize(externalSeeding=self.initSeed, solutionExport=None)
      self.samplersCombinations[samplingStrategy] = []
      self.limits['samples'] *= sampler.limits['samples']
      lowerBounds[samplingStrategy] = 0
      upperBounds[samplingStrategy] = sampler.limits['samples']
      while sampler.amIreadyToProvideAnInput():
        rlz = Realization()
        sampler.counters['samples'] += 1
        sampler.localGenerateInput(rlz, None, None)
        rlz.inputInfo['prefix'] = sampler.counter
        self.samplersCombinations[samplingStrategy].append(copy.deepcopy(rlz.asDict()))
      cnt += 1
      mKeys, mParams = sampler.provideExpectedMetaKeys()
      metadataKeys.extend(mKeys)
      metaParams.update(mParams)
    metadataKeys = list(set(metadataKeys))
    self.raiseAMessage(f'Total Number of Combined Samples is {self.limits["samples"]}!')
    # create a grid of combinations (no tensor)
    self.gridEnsemble = GridEntities.factory.returnInstance('GridEntity')
    initDict = {'dimensionNames': self.instantiatedSamplers.keys(),
                'stepLength': dict.fromkeys(self.instantiatedSamplers.keys(), [1]),
                'lowerBounds': lowerBounds,
                'upperBounds': upperBounds,
                'computeCells': False,
                'constructTensor': False,
                'excludeBounds': {'lowerBounds': False,'upperBounds': True}}
    self.gridEnsemble.initialize(initDict)
    # add meta data keys
    self.addMetaKeys(metadataKeys, params=metaParams)

  def localGenerateInput(self, rlz, model, modelInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the rlz.inputInfo should be ready to be sent
      to the model
      @ In, rlz, Realization, dict-like object to fill with sample
      @ In, model, model instance, an instance of a model
      @ In, modelInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    index = self.gridEnsemble.returnPointAndAdvanceIterator(returnDict=True)
    coordinate = []
    for samplingStrategy in self.instantiatedSamplers:
      coordinate.append(self.samplersCombinations[samplingStrategy][int(index[samplingStrategy])])
    for combination in coordinate:
      for key, value in combination.items():
        # FIXME we don't know what's inputInfo and what's sampled vars!
        if key in self.toBeSampled:
          rlz[key] = value
        elif key not in rlz.inputInfo:
          rlz.inputInfo[key] = value
        else:
          if isinstance(rlz.inputInfo[key], dict) and len(value):
            rlz.inputInfo[key].update(value)
          else:
            raise RuntimeError # can we get here?
    rlz.inputInfo['PointProbability'] = reduce(mul, rlz.inputInfo['SampledVarsPb'].values())
    rlz.inputInfo['ProbabilityWeight' ] = 1.0
    for key in rlz.inputInfo:
      if key.startswith('ProbabilityWeight-'):
        rlz.inputInfo['ProbabilityWeight' ] *= rlz.inputInfo[key]
    rlz.inputInfo['SamplerType'] = 'EnsembleForward'

    # Update dependent variables
    self._functionalVariables(rlz) # FIXME does this want batch or single?

  def flush(self):
    """
      Reset EnsembleForward attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.samplersCombinations = {}
    self.gridEnsemble = None
