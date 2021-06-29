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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
from operator import mul
from functools import reduce
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes
from .ForwardSampler        import ForwardSampler
from .MonteCarlo            import MonteCarlo
from .Grid                  import Grid
from .Stratified            import Stratified
from .FactorialDesign       import FactorialDesign
from .ResponseSurfaceDesign import ResponseSurfaceDesign
from .CustomSampler         import CustomSampler
import GridEntities
#Internal Modules End--------------------------------------------------------------------------------

class EnsembleForward(ForwardSampler):
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

    #It would be nice if Factory.knownTypes could be used to do that,
    # but that seems to cause recursive problems
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
    ForwardSampler.__init__(self)
    self.acceptableSamplers   = ['MonteCarlo','Stratified','Grid','FactorialDesign','ResponseSurfaceDesign','CustomSampler']
    self.printTag             = 'SAMPLER EnsembleForward'
    self.instanciatedSamplers = {}
    self.samplersCombinations = {}
    self.dependentSample      = {}

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    # this import happens here because a recursive call is made if we attempt it in the header
    from .Factory import factory
    for child in xmlNode:
      #sampler initialization
      if child.tag == 'samplerInit':
        ForwardSampler.readSamplerInit(self,xmlNode)
      # read in samplers
      elif child.tag in self.acceptableSamplers:
        child.attrib['name'] = child.tag
        self.instanciatedSamplers[child.tag] = factory.returnInstance(child.tag)
        #FIXME the variableGroups needs to be fixed
        self.instanciatedSamplers[child.tag].readXML(child, variableGroups={}, globalAttributes=self.globalAttributes)
        # fill toBeSampled so that correct check for samplable variables occurs
        self.toBeSampled.update(self.instanciatedSamplers[child.tag].toBeSampled)
      # function variables are defined outside the individual samplers
      elif child.tag=='variable':
        for childChild in child:
          if childChild.tag == 'function':
            self.dependentSample[child.attrib['name']] = childChild.text
          else:
            self.raiseAnError(IOError,"Variable " + str(child.attrib['name']) + " must be defined by a function since it is located outside the samplers block")
      # constants are handled in the base class
      elif child.tag == 'constant':
        pass
      # some samplers aren't eligible for ensembling
      elif child.tag in factory.knownTypes():
        self.raiseAnError(IOError,'Sampling strategy "{}" is not usable in "{}".  Available options include: {}.'.format(child.tag,self.type,", ".join(self.acceptableSamplers)))
      # catch-all for bad inputs
      else:
        self.raiseAnError(IOError,'Unrecognized sampling strategy: "{}". Available options include: {}.'.format(child.tag,", ".join(self.acceptableSamplers)))

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented here because this Sampler requests special objects
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed
    """
    # clear out toBeSampled, since ForwardSampler uses it for assembling
    self.toBeSampled = {}
    needDict = ForwardSampler._localWhatDoINeed(self)
    for combSampler in self.instanciatedSamplers.values():
      preNeedDict = combSampler.whatDoINeed()
      for key,value in preNeedDict.items():
        if key not in needDict.keys():
          needDict[key] = []
        needDict[key] = needDict[key] + value
    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    for combSampler in self.instanciatedSamplers.values():
      if combSampler.type != 'CustomSampler':
        combSampler._generateDistributions(availableDist,availableFunc)
      combSampler._localGenerateAssembler(initDict)
    self.raiseADebug("Distributions initialized!")

    for key,val in self.dependentSample.items():
      if val not in availableFunc.keys():
        self.raiseAnError(IOError, 'Function ',val,' was not found among the available functions:',availableFunc.keys())
      self.funcDict[key] = availableFunc[val]
      # check if the correct method is present
      if "evaluate" not in self.funcDict[key].availableMethods():
        self.raiseAnError(IOError,'Function '+self.funcDict[key].name+' does not contain a method named "evaluate". It must be present if this needs to be used in a Sampler!')

  def localInitialize(self):
    """
      Initialize the EnsembleForward sampler. It calls the localInitialize method of all the Samplers defined in this input
      @ In, None
      @ Out, None
    """
    self.limit = 1
    cnt = 0
    lowerBounds, upperBounds = {}, {}
    metadataKeys, metaParams = [], {}
    for samplingStrategy in self.instanciatedSamplers.keys():
      self.instanciatedSamplers[samplingStrategy].initialize(externalSeeding=self.initSeed,solutionExport=None)
      self.samplersCombinations[samplingStrategy] = []
      self.limit *= self.instanciatedSamplers[samplingStrategy].limit
      lowerBounds[samplingStrategy],upperBounds[samplingStrategy] = 0, self.instanciatedSamplers[samplingStrategy].limit
      while self.instanciatedSamplers[samplingStrategy].amIreadyToProvideAnInput():
        self.instanciatedSamplers[samplingStrategy].counter +=1
        self.instanciatedSamplers[samplingStrategy].localGenerateInput(None,None)
        self.instanciatedSamplers[samplingStrategy].inputInfo['prefix'] = self.instanciatedSamplers[samplingStrategy].counter
        self.samplersCombinations[samplingStrategy].append(copy.deepcopy(self.instanciatedSamplers[samplingStrategy].inputInfo))
      cnt+=1
      mKeys, mParams = self.instanciatedSamplers[samplingStrategy].provideExpectedMetaKeys()
      metadataKeys.extend(mKeys)
      metaParams.update(mParams)
    metadataKeys = list(set(metadataKeys))
    self.raiseAMessage('Number of Combined Samples are ' + str(self.limit) + '!')
    # create a grid of combinations (no tensor)
    self.gridEnsemble = GridEntities.factory.returnInstance('GridEntity')
    initDict = {'dimensionNames':self.instanciatedSamplers.keys(),
                'stepLength':dict.fromkeys(self.instanciatedSamplers.keys(),[1]),
                'lowerBounds':lowerBounds,
                'upperBounds':upperBounds,
                'computeCells':False,
                'constructTensor':False,
                'excludeBounds':{'lowerBounds':False,'upperBounds':True}}
    self.gridEnsemble.initialize(initDict)
    # add meta data keys
    self.addMetaKeys(metadataKeys, params=metaParams)

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    index = self.gridEnsemble.returnPointAndAdvanceIterator(returnDict = True)
    coordinate = []
    for samplingStrategy in self.instanciatedSamplers.keys():
      coordinate.append(self.samplersCombinations[samplingStrategy][int(index[samplingStrategy])])
    for combination in coordinate:
      for key in combination.keys():
        if key not in self.inputInfo.keys():
          self.inputInfo[key] = combination[key]

        else:
          if type(self.inputInfo[key]).__name__ == 'dict':
            self.inputInfo[key].update(combination[key])
    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = 1.0
    for key in self.inputInfo.keys():
      if key.startswith('ProbabilityWeight-'):
        self.inputInfo['ProbabilityWeight' ] *= self.inputInfo[key]
    self.inputInfo['SamplerType'] = 'EnsembleForward'

    # Update dependent variables
    for var in self.dependentSample.keys():
      test=self.funcDict[var].evaluate("evaluate",self.inputInfo['SampledVars'])
      for corrVar in var.split(","):
        self.values[corrVar.strip()] = test
        self.inputInfo['SampledVars'][corrVar.strip()] = test

