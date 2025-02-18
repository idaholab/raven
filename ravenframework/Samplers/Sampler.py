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
Created on Feb 16, 2013

@author: alfoa
"""

import sys
import abc
import json
import itertools
from collections import namedtuple

import numpy as np

from ..BaseClasses.InputDataUser import InputDataUser

from ..utils import utils,randomUtils,InputData, InputTypes
from ..utils.graphStructure import evaluateModelsOrder
from ..BaseClasses import BaseEntity, Assembler
from ..Realizations import RealizationBatch

_vectorPostfixFormat = '__RVEC__{ID}'

class Sampler(utils.metaclass_insert(abc.ABCMeta, BaseEntity), Assembler, InputDataUser):
  """
    This is the base class for samplers
    Samplers own the sampling strategy (Type) and they generate the input values using the associate distribution.
  """

  #### INITIALIZATION METHODS ####
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
    # FIXME the DET HybridSampler doesn't use the "name" param for the samples it creates,
    #      so we can't require the name yet
    # -> it's also in the base class ...
    # inputSpecification.addParam("name", InputTypes.StringType)

    variableInput = InputData.parameterInputFactory("variable", printPriority=100,
              descr='defines the input space variables to be sampled through various means.')
    # Added by alfoa: the variable name is always considered a single string. If a comma is present, we remove any leading spaces here
    # from StringType to StringNoLeadingSpacesType
    variableInput.addParam("name", InputTypes.StringNoLeadingSpacesType,
        descr=r"""user-defined name of this Sampler. \nb As for the other objects,
              this is the name that can be used to refer to this specific entity from other input blocks""")
    variableInput.addParam("shape", InputTypes.IntegerListType, required=False,
        descr=r"""determines the number of samples and shape of samples
              to be taken.  For example, \xmlAttr{shape}=``2,3'' will provide a 2 by 3
              matrix of values, while \xmlAttr{shape}=``10'' will produce a vector of 10 values.
              Omitting this optional attribute will result in a single scalar value instead.
              Each of the values in the matrix or vector will be the same as the single sampled value.
              \nb A model interface must be prepared to handle non-scalar inputs to use this option.""")
    variableInput.addParam("dims", InputTypes.StringListType, required=False,
        descr=r"""names the indexes that correspond to the shape of this variable. Required when \xmlAttr{shape}
              is provided. For example, with \xmlAttr{shape}=``2,3'', if the dimensions of the variable
              are ``years'' and ``hours'', then \xmlAttr{dims}=``year,hour'' tells RAVEN that the first
              dimension (with length 2) is called ``year'' and the second dimension (with length 3) is called
              ``hour``. Order must be the same as provided for \xmlAttr{shape}.
              \nb A model interface must be prepared to handle non-scalar inputs to use this option.""")
    distributionInput = InputData.parameterInputFactory("distribution", contentType=InputTypes.StringType,
        descr=r"""name of the distribution that is associated to this variable.
              Its name needs to be contained in the \xmlNode{Distributions} block explained
              in Section \ref{sec:distributions}. In addition, if NDDistribution is used,
              the attribute \xmlAttr{dim} is required. \nb{Alternatively, this node must be omitted
              if the \xmlNode{function} node is supplied.}""")
    distributionInput.addParam("dim", InputTypes.IntegerType,
        descr=r"""for an NDDistribution, indicates the dimension within the NDDistribution that corresponds
              to this variable.""")
    variableInput.addSub(distributionInput)
    gridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.StringType)
    gridInput.addParam("type", InputTypes.StringType)
    gridInput.addParam("construction", InputTypes.StringType)
    gridInput.addParam("steps", InputTypes.IntegerType)
    variableInput.addSub(gridInput)
    functionInput = InputData.parameterInputFactory("function", contentType=InputTypes.StringType,
        descr=r"""name of the function that
              defines the calculation of this variable from other distributed variables.  Its name
              needs to be contained in the \xmlNode{Functions} block explained in Section
              \ref{sec:functions}. This function module must contain and implement a method either with the same name of the function or a method named  ``evaluate''.
              \nb{Each \xmlNode{variable} must contain only one \xmlNode{Function} or
              \xmlNode{Distribution}, but not both.} """)
    variableInput.addSub(functionInput)
    inputSpecification.addSub(variableInput)

    constantInput = InputData.parameterInputFactory("constant", contentType=InputTypes.InterpretedListType,
        printPriority=110,
        descr=r"""allows variables that do not change value to be part of the input space.""")
    # Added by alfoa: the variable name is always considered a single string. If a comma is present, we remove any leading spaces here
    # from StringType to StringNoLeadingSpacesType
    constantInput.addParam("name", InputTypes.StringNoLeadingSpacesType, required=True,
        descr=r"""variable name for this constant, which will be provided to the Model. """)
    constantInput.addParam("shape", InputTypes.IntegerListType, required=False,
        descr=r"""determines the shape of samples of the constant value.
              For example, \xmlAttr{shape}=``2,3'' will shape the values into a 2 by 3
              matrix, while \xmlAttr{shape}=``10'' will shape into a vector of 10 values.
              Unlike the \xmlNode{variable}, the constant requires each value be entered; the number
              of required values is equal to the product of the \xmlAttr{shape} values, e.g. 6 entries for shape ``2,3'').
              \nb A model interface must be prepared to handle non-scalar inputs to use this option. """)
    constantInput.addParam("source", InputTypes.StringType, required=False,
        descr=r"""the name of the DataObject containing the value to be used for this constant.
              Requires \xmlNode{ConstantSource} node with a \xmlNode{DataObject} identified for this
              Sampler/Optimizer.""")
    constantInput.addParam("index", InputTypes.IntegerType, required=False,
        descr=r"""the index of the realization in the \xmlNode{ConstantSource} \xmlNode{DataObject}
                  containing the value for this constant. Requires \xmlNode{ConstantSource} node with
                  a \xmlNode{DataObject} identified for this Sampler/Optimizer.""")
    inputSpecification.addSub(constantInput)

    sourceInput = InputData.parameterInputFactory("ConstantSource", contentType=InputTypes.StringType,
        printPriority=111,
        descr=r"""identifies a \xmlNode{DataObject} to provide \xmlNode{constant} values to the input
              space of this entity while sampling. As an alternative to providing predefined values
              for constants, the \xmlNode{ConstantSource} provides a dynamic means of always providing
              the same value for a constant. This is often used as part of a larger multi-workflow
              calculation.""")
    sourceInput.addParam("class", InputTypes.StringType,
        descr=r"""The RAVEN class for this source. Options include \xmlString{DataObject}. """)
    sourceInput.addParam("type", InputTypes.StringType,
        descr=r"""The RAVEN type for this source. Options include any valid \xmlNode{DataObject} type,
              such as HistorySet or PointSet.""")
    inputSpecification.addSub(sourceInput)

    restartInput = InputData.parameterInputFactory("Restart", contentType=InputTypes.StringType,
        printPriority=200,
        descr=r"""name of a DataObject. Used to leverage existing data when sampling a model. For
              example, if a Model has
              already been sampled, but some samples were not collected, the successful samples can
              be stored and used instead of rerunning the model for those specific samples. This RAVEN
              entity definition must be a DataObject with contents including the input and output spaces
              of the Model being sampled.""")
    restartInput.addParam("class", InputTypes.StringType,
        descr=r"""The RAVEN class for this source. Options include \xmlString{DataObject}. """)
    restartInput.addParam("type", InputTypes.StringType,
        descr=r"""The RAVEN type for this source. Options include any valid \xmlNode{DataObject} type,
              such as HistorySet or PointSet.""")
    inputSpecification.addSub(restartInput)

    restartToleranceInput = InputData.parameterInputFactory("restartTolerance", contentType=InputTypes.FloatType,
        printPriority=210,
        descr=r"""specifies how strictly a matching point from a \xmlNode{Restart} DataObject must match
              the desired sample point in order to be used. If a potential restart point is within a
              relative Euclidean distance (as specified by the value in this node) of a desired sample point,
              the restart point will be used instead of sampling the Model. \default{1e-15} """)
    inputSpecification.addSub(restartToleranceInput)

    variablesTransformationInput = InputData.parameterInputFactory("variablesTransformation",
        printPriority=500,
        descr=r"""Allows transformation of variables via translation matrices. This defines two spaces,
              a ``latent'' transformed space sampled by RAVEN and a ``manifest'' original space understood
              by the Model.""")
    variablesTransformationInput.addParam('distribution', InputTypes.StringType,
        descr=r"""the name for the distribution defined in the XML node \xmlNode{Distributions}.
              This attribute indicates the values of \xmlNode{manifestVariables} are drawn from
              \xmlAttr{distribution}. """)
    variablesTransformationInput.addSub(InputData.parameterInputFactory("latentVariables", contentType=InputTypes.StringListType,
        descr=r"""user-defined latent variables that are used for the variables transformation.
              All the variables listed under this node should be also mentioned in \xmlNode{variable}. """))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("manifestVariables", contentType=InputTypes.StringListType,
        descr=r"""user-defined manifest variables that can be used by the \xmlNode{Model}. """))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("manifestVariablesIndex", contentType=InputTypes.StringListType,
        descr=r"""user-defined manifest variables indices paired with \xmlNode{manifestVariables}.
              These indices indicate the position of manifest variables associated with multivariate normal
              distribution defined in the XML node \xmlNode{Distributions}.
              The indices should be postive integer. If not provided, the code will use the positions
              of manifest variables listed in \xmlNode{manifestVariables} as the indices. """))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("method", contentType=InputTypes.StringType,
        descr=r"""the method that is used for the variables transformation. The currently available method is \xmlString{pca}. """))
    inputSpecification.addSub(variablesTransformationInput)

    rom = InputData.assemblyInputFactory('ROM', contentType=InputTypes.StringType,
                                          printPriority=175,
                                          descr=r"""Name of a Model that optimizers may want to use during optimization. For example, the
                                          Bayesian Optimizer requires a ROM to select points during optimization. The model is defined in
                                          detail with in the \xmlNode{Models} as in other uses. This node should be provided a string referencing
                                          the model definition's name.""")
    inputSpecification.addSub(rom)

    return inputSpecification

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, vars, dict, {varName: manual description} for each solution export option
    """
    ok = super(Sampler, cls).getSolutionExportVariableNames()
    new = {
        'batchID': 'identifier for the sampling batch. If not batching, same as sample identifier.'
    }
    ok.update(new)
    return ok


  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    ### COUNTERS AND FLAGS ###
    self.batch = 0            # determines the size of each sampling batch to run, 0 means none
    self.counters = {
                     'batches': 0, # Counter of number of batches submitted. Same as "samples" if not batching.
                     'samples': 0, # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
                     'seeding': 0, # Used to control consecutive seeding, was "auxcnt"
    }
    self.limits = {
                   'samples': sys.maxsize # limits the number of samples that can be taken. Other samples can add additional keywords.
    }
    self.initSeed = None      # if not provided the seed is randomly generated at the initialization of the sampler, the step can override the seed by sending in another one
    self.printTag = self.type # prefix for all prints (sampler type)
    self.reseedAtEachIteration = False # Logical flag. True if every newer evaluation is performed after a new reseeding
    self.onlySampleAfterCollecting = True # if True, then no new samples unless collection has occurred
    self.ableToHandelFailedRuns = False # is this sampler able to handle failed runs?

    ### INFO DICTS ###
    self.samplerInfo = {      # depending on the sampler several different type of keywarded information could be present only one is mandatory, see below
      'crowDist': {},         # Stores a dictionary that contains the information to create a crow distribution.  Stored as a json object
    }
    self.toBeSampled = {}     # Sampling mapping dictionary {'Variable Name':'name of the distribution'}
    self.distDict = {}        # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.dependentSample = {} # Sampling mapping dictionary for dependent variables {'Variable Name':'name of the external function'}
                              #   element 0 (methodName): name of the method in the function to be be invoked. Either the default "evaluate", or the function name
    self.ndVariables = {}     # stores the dimensionality (names and shapes) of each variable by name, as tuple e.g. shape = (2,3) for [[#,#,#],[#,#,#]]
    self.constants = {}       # Unsampled constant variables mapped to values
    self.constantSources = {} # storage for the way to obtain constant information
    self.constantSourceData = None  # dictionary of data objects from which constants can take values
    self.distributions2variablesIndexList = {}

    ### FUNCTION EVALUATIONS ###
    self.funcDict = {}        # Mapping between variable name and the a 2-element namedtuple namedtuple('func', ['methodName', 'instance']) containing:
    self.variableFunctionExecutionList = [] # This is an ordered sequence of functional variable
                                            # (linked to functions) that need to be performed (in case of
                                            # interdependency). This list is always created. If no interdependence
                                            # is detected, the order is just random, otherwise the order is
                                            # determined through graph theory.
                                            # element 1 (instance): instance of the function to be used, it is created every time the sampler is initialized.
    ### JOB MANAGEMENT ###
    self._jobsToEnd = []               # list of strings, containing job prefixes that should be cancelled.
    self._endJobRunnable = sys.maxsize # max number of inputs creatable by the sampler right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)

    ### RESTART DATA ###
    self.restartData = None         # presampled points to restart from
    self.restartTolerance = 1e-14   # strictness with which to find matches in the restart data
    self.restartIsCompatible = None # flags restart as compatible with the sampling scheme (used to speed up checking)

    ### ND MAPPING ###
    # for each variable 'varName'  , the following informations are included:  'varName': {'dim': 1, 'reducedDim': 1,'totDim': 2, 'name': 'distName'} ;
    #                                                                           dim = dimension of the variable;
    #                                                                           reducedDim = dimension of the variable in the transformed space;
    #                                                                           totDim = total dimensionality of its associated distribution
    self.variables2distributionsMapping = {}
    # for each variable 'distName' , the following informations are included: 'distName': [{'var1': 1}, {'var2': 2}]} where for each var it is indicated the var dimension
    self.distributions2variablesMapping = {}
    # this dictionary contains a dictionary for each ND distribution (key). This latter dictionary contains the initialization parameters of the
    # ND inverseCDF ('initialGridDisc' and 'tolerance')
    self.NDSamplingParams = {}

    ### PCA TRANSFORM ###
    self.variablesTransformationDict = {}         # for each variable 'modelName', the following informations are included:
                                                     # {'modelName': {latentVariables:[latentVar1, latentVar2, ...], manifestVariables:[manifestVar1,manifestVar2,...]}}
    self.transformationMethod = {}         # transformation method used in variablesTransformation node {'modelName':method}
    self.entitiesToRemove = []         # This variable is used in order to make sure the transformation info is printed once in the output xml file.

    ### ASSEMBLING ###
    self.addAssemblerObject('Restart', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('ConstantSource', InputData.Quantity.zero_to_infinity)

  def _generateDistributions(self, availableDist, availableFunc):
    """
      Generates the distributions and functions.
      @ In, availableDist, dict, dict of distributions
      @ In, availableFunc, dict, dict of functions
      @ Out, None
    """
    if self.initSeed is not None:
      randomUtils.randomSeed(self.initSeed)
    for var, dist in self.toBeSampled.items():
      if dist not in availableDist:
        self.raiseAnError(IOError, f'Distribution "{dist}" not found among available distributions (check input)!')
      self.distDict[var] = availableDist[dist]
      self.samplerInfo['crowDist'][var] = json.dumps(self.distDict[var].getCrowDistDict())
    for key, val in self.dependentSample.items():
      if val not in availableFunc.keys():
        self.raiseAnError(ValueError, f'Function {val} was not found among the available functions:', availableFunc.keys())
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

    # evaluate function execution order
    self._evaluateFunctionsOrder()

  def _localGenerateAssembler(self, initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    self._generateDistributions(availableDist, availableFunc)

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [] # Every sampler requires Distributions OR a Function
    needDict['Functions']     = [] # Every sampler requires Distributions OR a Function
    for dist in self.toBeSampled.values():
      needDict['Distributions'].append((None,dist))
    for func in self.dependentSample.values():
      needDict['Functions'].append((None,func))

    return needDict

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      The text is supposed to contain the info where and which variable to change.
      In case of a code the syntax is specified by the code interface itself
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    #TODO remove using xmlNode
    Assembler._readMoreXML(self,xmlNode)
    paramInput = self._readMoreXMLbase(xmlNode)
    self.localInputAndChecks(xmlNode, paramInput)
    if self.type not in ['MonteCarlo', 'Metropolis']:
      if not self.toBeSampled:
        self.raiseAnError(IOError, f'<{self.type}> sampler named "{self.name}" requires at least one sampled <variable>!')
    self._checkNDVariables()

  def _readMoreXMLbase(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to the base sampler only
      and initialize some stuff based on the inputs got
      The text is supposed to contain the info where and which variable to change.
      In case of a code the syntax is specified by the code interface itself
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node1
      @ Out, paramInput, InputData.ParameterInput the parsed paramInput
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      prefix = ""

      if child.getName() == 'Distribution':
        for childChild in child.subparts:
          if childChild.getName() =='distribution':
            prefix = "<distribution>"
            toBeSampled = childChild.value
        self.toBeSampled[prefix+child.parameterValues['name']] = toBeSampled

      elif child.getName() == 'variable':
        self._readInVariable(child, prefix)

      elif child.getName() == "variablesTransformation":
        transformationDict = {}
        listIndex = None
        for childChild in child.subparts:
          if childChild.getName() == "latentVariables":
            transformationDict[childChild.getName()] = list(childChild.value)
          elif childChild.getName() == "manifestVariables":
            transformationDict[childChild.getName()] = list(childChild.value)
          elif childChild.getName() == "manifestVariablesIndex":
            # the index provided by the input file starts from 1, but the index used by the code starts from 0.
            listIndex = list(int(inp) - 1  for inp in childChild.value)
          elif childChild.getName() == "method":
            self.transformationMethod[child.parameterValues['distribution']] = childChild.value
        if listIndex is None:
          self.raiseAWarning('Index is not provided for manifestVariables, default index will be used instead!')
          listIndex = range(len(transformationDict["manifestVariables"]))
        transformationDict["manifestVariablesIndex"] = listIndex
        self.variablesTransformationDict[child.parameterValues['distribution']] = transformationDict

      elif child.getName() == "constant":
        name, value = self._readInConstant(child)
        self.constants[name] = value

      elif child.getName() == "restartTolerance":
        self.restartTolerance = child.value

    if len(self.constants) > 0:
      # check if constant variables are also part of the sampled space. In case, error out
      if not set(self.toBeSampled.keys()).isdisjoint(self.constants.keys()):
        self.raiseAnError(IOError,"Some constant variables are also in the sampling space:" +
                                  ' '.join([i if i in self.toBeSampled else "" for i in self.constants])  )

    if self.initSeed is None:
      self.initSeed = randomUtils.randomIntegers(0, 2**31, self)
    # Creation of the self.distributions2variablesMapping dictionary: {'distName': [{'variable_name1': dim1}, {'variable_name2': dim2}]}
    for variable in self.variables2distributionsMapping:
      distName = self.variables2distributionsMapping[variable]['name']
      dim      = self.variables2distributionsMapping[variable]['dim']
      listElement = {}
      listElement[variable] = dim
      if distName in self.distributions2variablesMapping:
        self.distributions2variablesMapping[distName].append(listElement)
      else:
        self.distributions2variablesMapping[distName] = [listElement]

    # creation of the self.distributions2variablesIndexList dictionary:{'distName':[dim1,dim2,...,dimN]}
    self.distributions2variablesIndexList = {}
    for distName in self.distributions2variablesMapping:
      positionList = []
      for var in self.distributions2variablesMapping[distName]:
        position = utils.first(var.values())
        positionList.append(position)
      if sum(set(positionList)) > 1 and len(positionList) != len(set(positionList)):
        dups = set(str(var) for var in positionList if positionList.count(var) > 1)
        self.raiseAnError(IOError, f'Each of the following dimensions are assigned to multiple variables in Samplers: "{", ".join(dups)}"',
                ' associated to ND distribution ', distName, '. This is currently not allowed!')
      positionList = list(set(positionList))
      positionList.sort()
      self.distributions2variablesIndexList[distName] = positionList

    for key in self.variables2distributionsMapping:
      distName = self.variables2distributionsMapping[key]['name']
      dim      = self.variables2distributionsMapping[key]['dim']
      reducedDim = self.distributions2variablesIndexList[distName].index(dim) + 1
      self.variables2distributionsMapping[key]['reducedDim'] = reducedDim  # the dimension of variable in the transformed space
      self.variables2distributionsMapping[key]['totDim'] = max(self.distributions2variablesIndexList[distName]) # We will reset the value if the node <variablesTransformation> exist in the raven input file
      if not self.variablesTransformationDict and self.variables2distributionsMapping[key]['totDim'] > 1:
        if self.variables2distributionsMapping[key]['totDim'] != len(self.distributions2variablesIndexList[distName]):
          self.raiseAnError(IOError,'The "dim" assigned to the variables insider Sampler are not correct! the "dim" should start from 1, and end with the full dimension of given distribution')

    # Checking the variables transformation
    if self.variablesTransformationDict:
      for dist, varsDict in self.variablesTransformationDict.items():
        maxDim = len(varsDict['manifestVariables'])
        listLatentElement = varsDict['latentVariables']
        if len(set(listLatentElement)) != len(listLatentElement):
          dups = set(var for var in listLatentElement if listLatentElement.count(var) > 1)
          self.raiseAnError(IOError, f'The following are duplicated variables listed in the latentVariables: {dups}')
        if len(set(varsDict['manifestVariables'])) != len(varsDict['manifestVariables']):
          dups = set(var for var in varsDict['manifestVariables'] if varsDict['manifestVariables'].count(var) > 1)
          self.raiseAnError(IOError, f'The following are duplicated variables listed in the manifestVariables: {dups}')
        if len(set(varsDict['manifestVariablesIndex'])) != len(varsDict['manifestVariablesIndex']):
          dups = set(var+1 for var in varsDict['manifestVariablesIndex'] if varsDict['manifestVariablesIndex'].count(var) > 1)
          self.raiseAnError(IOError, f'The following are duplicated variables indices listed in the manifestVariablesIndex: {dups}')
        listElement = self.distributions2variablesMapping[dist]
        for var in listElement:
          self.variables2distributionsMapping[utils.first(var.keys())]['totDim'] = maxDim #reset the totDim to reflect the totDim of original input space
        tempListElement = {k.strip():v for x in listElement for ks,v in x.items() for k in list(ks.strip().split(','))}
        listIndex = []
        for var in listLatentElement:
          if var not in set(tempListElement.keys()):
            self.raiseAnError(IOError, f'The variable listed in latentVariables {var} is not listed in the given distribution: {dist}')
          listIndex.append(tempListElement[var]-1)
        if max(listIndex) > maxDim:
          self.raiseAnError(IOError, f'The maximum dim = {max(listIndex)} defined for latent variables has exceeded the dimension of the problem {maxDim}')
        if len(set(listIndex)) != len(listIndex):
          dups = set(var+1 for var in listIndex if listIndex.count(var) > 1)
          self.raiseAnError(IOError, f'Each of the following dimensions are assigned to multiple latent variables in Samplers: {dups}')
        # update the index for latentVariables according to the 'dim' assigned for given var defined in Sampler
        self.variablesTransformationDict[dist]['latentVariablesIndex'] = listIndex

    return paramInput

  def _readInVariable(self, child, prefix):
    """
      Reads in a "variable" input parameter node.
      @ In, child, utils.InputData.ParameterInput, input parameter node to read from
      @ In, prefix, str, variable prefix, if any
      @ Out, None
    """
    # variable for tracking if distributions or functions have been declared
    foundDistOrFunc = False
    # store variable name for re-use
    varName = child.parameterValues['name']
    # set shape if present
    shape = child.parameterValues.get('shape', None)
    if shape is not None:
      dims = child.parameterValues.get('dims', None)
      # TODO move this check to an input check
      # -> if "shape" is present, "dims" must be present as well!
      if dims is None:
        self.raiseAnError(IOError, f'For variable "{varName}" the "shape" parameter was provided without the "dims" parameter!')
      if len(shape) != len(dims):
        self.raiseAnError(IOError, f'For variable "{varName}" the number of entries in "shape" and "dims" does not match!')
      self.ndVariables[varName] = {'shape': shape,
                                   'dims': dims}
    # read subnodes
    for childChild in child.subparts:
      if childChild.getName() == 'distribution':
        # can only have a distribution if doesn't already have a distribution or function
        if foundDistOrFunc:
          self.raiseAnError(IOError, 'A sampled variable cannot have both a distribution and a function, or more than one of either!')
        else:
          foundDistOrFunc = True
        distName = childChild.value
        varData = {}
        varData['name'] = distName
        # variable dimensionality
        dim = childChild.parameterValues.get('dim', 1)
        varData['dim'] = dim
        # flag distribution as needing to be sampled
        # if a ND variable, loop over elements and set them each
        # to be sampled as if they were independent variables.
        # If not a ND variable, treat it like a length-1 array.
        if varName in self.ndVariables:
          shape = self.ndVariables.get(varName)['shape']
        else:
          shape = 1
        totalIndices = np.zeros(shape).size
        for i in range(totalIndices):
          name = varName
          if totalIndices > 1:
            name += _vectorPostfixFormat.format(ID=str(i))
          self.toBeSampled[prefix + name] = distName
          # set up mapping for variable to distribution
          self.variables2distributionsMapping[name] = varData
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

    if not foundDistOrFunc:
      self.raiseAnError(IOError, 'Sampled variable', varName, 'has neither a <distribution> nor <function> node specified!')

  def _readInConstant(self, inp):
    """
      Reads in a "constant" input parameter node.
      @ In, inp, utils.InputParameter.ParameterInput, input parameter node to read from
      @ Out, name, string, name of constant
      @ Out, value, float or np.array,
    """
    # constantSources
    value = inp.value
    name = inp.parameterValues['name']
    shape = inp.parameterValues.get('shape',None)
    source = inp.parameterValues.get('source',None)
    # if constant's value is provided directly by value ...
    if source is None:
      # if single entry, remove array structure; if multiple entries, cast them as numpy array
      if len(value) == 1:
        value = value[0]
      else:
        value = np.asarray(value)
      # if specific shape requested, then reshape it
      if shape is not None:
        try:
          value = value.reshape(shape)
        except ValueError:
          self.raiseAnError(IOError,
              (f'Requested shape "{shape}" ({np.prod(shape)} entries) for constant "{name}"' +\
              f' is not consistent with the provided values ({len(value)} entries)!'))
    # else if constant's value is provided from a DataObject ...
    else:
      self.constantSources[name] = {'shape'    : shape,
                                    'source'   : source,
                                    'index'    : inp.parameterValues.get('index', -1),
                                    'sourceVar': value[0]} # generally, constants are a list, but in this case just take the only entry

    return name, value

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for variable in self.toBeSampled.items():
      paramDict["sampled variable: "+variable[0]] = 'is sampled using the distribution ' +variable[1]
    paramDict['limit' ] = self.limits['samples']
    paramDict['initial seed' ] = self.initSeed
    paramDict.update(self.localGetInitParams())

    return paramDict

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, in goal oriented sampling (a.k.a. adaptive sampling this is where the space/point satisfying the constrains)
      @ Out, None
    """
    if self.initSeed is None:
      self.initSeed = randomUtils.randomIntegers(0,2**31,self)
    self.counters['samples'] = 0
    self.counters['batches'] = 0
    if not externalSeeding:
      randomUtils.randomSeed(self.initSeed) # use the sampler initialization seed
      self.counters['seeding'] = self.initSeed
    elif externalSeeding=='continue':
      pass # in this case the random sequence needs to be preserved
    else:
      randomUtils.randomSeed(externalSeeding) # the external seeding is used
      self.counters['seeding'] = externalSeeding
    # grab restart dataobject if it's available, then in localInitialize the sampler can deal with it.
    if 'Restart' in self.assemblerDict:
      self.raiseADebug('Restart object: '+str(self.assemblerDict['Restart']))
      self.restartData = self.assemblerDict['Restart'][0][3]
      # check the right variables are in the restart
      need = set(itertools.chain(self.toBeSampled.keys(), self.dependentSample.keys()))
      if not need.issubset(set(self.restartData.getVars())):
        missing = need - set(self.restartData.getVars())
        # TODO this could be a warning, instead, but user wouldn't see it until the run was deep in
        self.raiseAnError(KeyError, f'Restart data object "{self.restartData.name}" is missing the following variables: "{", ".join(missing)}". No restart can be performed.')
      else:
        self.raiseAMessage(f'Restarting from {self.restartData.name}')
      # we used to check distribution consistency here, but we want to give more flexibility to using
      # restart data, so do NOT check distributions of restart data.
    else:
      self.raiseAMessage(f'No restart for {self.printTag}')

    if 'ConstantSource' in self.assemblerDict:
      # find all the sources requested in the sampler, map data objects to their requested names
      self.constantSourceData = dict((a[2],a[3]) for a in self.assemblerDict['ConstantSource'])
      for var,data in self.constantSources.items():
        source = self.constantSourceData[data['source']]
        rlz = source.realization(index=data['index'])
        if data['sourceVar'] not in rlz:
          self.raiseAnError(IOError, f'Requested variable "{data["sourceVar"]}" from DataObject "{source.name}" to set constant "{var}",'+\
                                    f' but "{data["sourceVar"]}" is not a variable in "{source.name}"!')
        self.constants[var] = rlz[data['sourceVar']]

    # specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport is not None:
      self.localInitialize(solutionExport=solutionExport)
    else:
      self.localInitialize()

    for distrib, params in self.NDSamplingParams.items():
      if distrib in self.distributions2variablesMapping:
        temp = utils.first(self.distributions2variablesMapping[distrib][0].keys())
        try:
          self.distDict[temp].updateRNGParam(params)
        except AttributeError as err:
          msg =f'Distribution with name {distrib} is not a valid N-Dimensional probability distribution!'
          err.msg = msg
          raise err
      else:
        self.raiseAnError(IOError, f'Distribution "{distrib}" specified in distInit block of sampler "{self.name}" does not exist!')

    # Store the transformation matrix in the metadata
    if self.variablesTransformationDict:
      self.entitiesToRemove = []
      for variable, data in self.variables2distributionsMapping.items():
        distName = data['name']
        dim      = data['dim']
        totDim   = data['totDim']
        if totDim > 1 and dim  == 1:
          transformDict = {}
          transformDict['type'] = self.distDict[variable.strip()].type
          transformDict['transformationMatrix'] = self.distDict[variable.strip()].transformationMatrix()
          self.samplerInfo[f'transformation-{distName}'] = transformDict
          self.entitiesToRemove.append(f'transformation-{distName}')

    # Register expected metadata
    meta = ['ProbabilityWeight','prefix','PointProbability']
    for var in self.toBeSampled:
      meta +=  ['ProbabilityWeight-'+ key for key in var.split(",")]
    if self.batch > 0:
      meta.append('batchID')
      meta.append('batchSize')
    self.addMetaKeys(meta)

  def getBatchSize(self):
    """
      Returns the size of batches to use for this Sampler. Default is 0.
      @ In, None
      @ Out, size, int, 0
    """
    return self.batch

  def localGetInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def localInitialize(self, **kwargs):
    """
      use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, None
      @ Out, None
    """

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Local method. Place here the additional reading, remember to add initial parameters in the method localGetInitParams
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """

  def readSamplerInit(self,xmlNode):
    """
      This method is responsible to read only the samplerInit block in the .xml file.
      This method has been moved from the base sampler class since the samplerInit block is needed only for the MC and stratified (LHS) samplers
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    # TODO, this is redundant and paramInput should be directly passed in.
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == "samplerInit":
        self.initSeed = randomUtils.randomIntegers(0,2**31,self)
        for childChild in child.subparts:
          if childChild.getName() == "limit":
            try:
              self.limits['samples'] = int(childChild.value)
            except ValueError:
              self.raiseAnError(IOError, f'reading the attribute for the sampler {self.name} it was not possible to perform the conversion to integer for the attribute limit with value {childChild.value}')
          if childChild.getName() == "initialSeed":
            try:
              self.initSeed = int(childChild.value)
            except ValueError:
              self.raiseAnError(IOError, f'reading the attribute for the sampler {self.name} it was not possible to perform the conversion to integer for the attribute initialSeed with value {childChild.value}')
          elif childChild.getName() == "reseedEachIteration":
            if utils.stringIsTrue(childChild.value):
              self.reseedAtEachIteration = True
          elif childChild.getName() == "distInit":
            for childChildChild in childChild.subparts:
              NDdistData = {}
              for childChildChildChild in childChildChild.subparts:
                if childChildChildChild.getName() == 'initialGridDisc':
                  NDdistData[childChildChildChild.getName()] = int(childChildChildChild.value)
                elif childChildChildChild.getName() == 'tolerance':
                  NDdistData[childChildChildChild.getName()] = float(childChildChildChild.value)
                else:
                  self.raiseAnError(IOError, f'Unknown tag {childChildChildChild.getName()}. Available are: initialGridDisc and tolerance!')
              self.NDSamplingParams[childChildChild.parameterValues['name']] = NDdistData

  def _checkNDVariables(self):
    """
      Provides an opportunity to check compatibility with and usage of N-dimensional variables.
      By default, errors and provides notification to users.
      @ In, None
      @ Out, None
    """
    # NOTE the base class Sampler will handle moving ND variables into individual variables
    #      using the self.toBeSampled dictionary mapping, so no specific action needs to be taken
    #      to enable ND variables for a sampler, aside from overriding this method in the sampler.
    if self.ndVariables:
      self.raiseAnError(IOError, f'"{self.type}" sampler named "{self.name}" is not compatible with ND-variables (using the "shape" parameter!)')

  #### GETTERS AND SETTERS ####
  def endJobRunnable(self):
    """
      Returns the maximum number of inputs allowed to be created by the sampler
      right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)
      @ In, None
      @ Out, endJobRunnable, int, number of runnable jobs at the end of each sample
    """
    return self._endJobRunnable

  def getCurrentSetting(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['counter'] = self.counters['samples']
    paramDict['initial seed'] = self.initSeed
    paramDict.update(self.samplerInfo)
    paramDict.update(self.localGetCurrentSetting())
    return paramDict

  def getJobsToEnd(self, clear=False):
    """
      Provides a list of jobs that should be terminated.
      @ In, clear, bool, optional, if True then clear list after returning.
      @ Out, ret, list, jobs to terminate
    """
    ret = set(self._jobsToEnd[:])
    if clear:
      self._jobsToEnd = []

    return ret

  def localGetCurrentSetting(self):
    """
      Returns a dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  #### SAMPLING METHODS ####
  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
      @ In, None
      @ Out, ready, bool, is this sampler ready to generate another sample?
    """
    if self.counters['samples'] < self.limits['samples']: # can use < since counter is 0-based
      ready = True
    else:
      ready = False
      self.raiseADebug('Sampling limit reached! No new samples ...')
    ready = self.localStillReady(ready)

    return ready

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    if self.counters['samples'] > self.limits['samples']:
      ready = False

    return ready

  def _checkRestartForEvaluation(self, rlz):
    """
      Checks restart data object (if any) for matching realization.
      @ In, rlz, Realization, realization to check for in restart
      @ In, None
      @ Out, index, int, index of matching realization in restart (None if not found)
      @ Out, inExisting, dict, matching realization (None if not found)
    """
    #check if point already exists
    if self.restartData is not None:
      index,inExisting = self.restartData.realization(matchDict=rlz, tol=self.restartTolerance, unpackXArray=True)
    else:
      index = None
      inExisting = None
    return index, inExisting

  def _constantVariables(self, rlzBatch):
    """
      Method to set the constant variables into the inputInfo dictionary
      @ In, rlzBatch, BatchRealization, batch of mapping of sampled vars to values
      @ Out, None
    """
    if len(self.constants) > 0:
      for rlz in rlzBatch:
        # we inject the constant variables into the SampledVars
        rlz.update(self.constants)
        # we consider that CDF of the constant variables is equal to 1 (same as its Pb Weight)
        rlz.inputInfo['SampledVarsPb'].update(dict.fromkeys(self.constants.keys(), 1.0))
        pbKey = ['ProbabilityWeight-'+key for key in self.constants]
        self.addMetaKeys(pbKey)
        rlz.inputInfo.update(dict.fromkeys(['ProbabilityWeight-'+key for key in self.constants], 1.0))

  def _formNDVariables(self, rlzBatch):
    """
      Formats ND variables to fit the requested shape.
      @ In, rlzBatch, BatchRealization, batch of mapping of sampled vars to values
      @ Out, None
    """
    for rlz in rlzBatch:
      for baseName, info in self.ndVariables.items():
        shape = info['shape']
        dims = info['dims']
        # collect all the values from the split variables
        values = []
        entries = np.zeros(shape).size
        for i in range(entries):
          var = baseName
          if entries > 1:
            var += _vectorPostfixFormat.format(ID=str(i))
          value = rlz.values.pop(var)
          values.append(value)
        # shape values into the requested format
        rlz[baseName] = np.asarray(values).reshape(shape)
        # update indexMap
        if entries > 1:
          # TODO do we need to add both to self.values and to rlz (inputInfo.sampledvars)?
          rlz.indexMap[baseName] = dims
          # check for missing index vars and add default values if needed
          for d,dim in enumerate(dims):
            if dim not in rlz:
              rlz[dim] = np.arange(shape[d])
              self.raiseAWarning(f'Values for index "{dim}" not provided in Sampler; ' +\
                                 f'using default values (0 to {rlz[dim][-1]}).')

  def _expandNDVariable(self, ndName, ndVals):
    """
      Turns a name-NDarray pair into individual name-value pairs
      @ In, ndName, name of (full ND array) variable
      @ In, ndVals, np.ndarray, ND array of values
      @ Out, expanded, dict, mapping of individual names to values
    """
    # defined above, but for reference:
    # _vectorPostfixFormat = '__RVEC__{ID}'
    expanded = {}
    for ID, val in enumerate(ndVals.flat):
      name = ndName + _vectorPostfixFormat.format(ID=ID)
      expanded[name] = val
    return expanded

  def _evaluateFunctionsOrder(self):
    """
      Method to evaluate the function execution order using graph theory
      The order is stored in self.variableFunctionExecutionList
      @ In, None
      @ Out, None
    """
    functionsToVariables = {}
    for var, inst in  self.funcDict.items():
      outputMatch = []
      functionInputs = inst.instance.parameterNames()
      for inpVar in functionInputs:
        # find functions that are linked to this inpVar
        if inpVar in self.funcDict:
          outputMatch.append(inpVar)
      outputMatch = list(set(outputMatch))
      functionsToVariables[var] =  outputMatch
    executionList, variableFunctionsGraph, errMsg = evaluateModelsOrder(functionsToVariables, acceptLoop=False, reverse=True)
    if errMsg is not None:
      self.raiseAnError(*errMsg)
    if  executionList:
      self.variableFunctionExecutionList = executionList
      self.raiseAMessage("Function Variables are interdependent")
      self.raiseAMessage("Variable Evaluation and Function Execution list: "+
                         ' -> '.join([f"variable:{var} | function: {self.funcDict[var].instance.name}"
                                      for var in self.variableFunctionExecutionList]))

  def _functionalVariables(self, rlzBatch):
    """
      Evaluates variables that are functions of other input variables.
      @ In, rlzBatch, BatchRealization, batch of mapping of sampled vars to values
      @ Out, None
    """
    for var in self.variableFunctionExecutionList:
      for rlz in rlzBatch:
        funcEval = self.funcDict[var].instance.evaluate(self.funcDict[var].methodName, rlz)
        for corrVar in var.split(","):
          rlz[corrVar.strip()] = funcEval

  def _incrementCounter(self, numAdded=1):
    """
      Increments counter and sets up prefix.
      @ In, None
      @ Out, None
    """
    #since we are creating the input for the next run we increase the counter and global counter
    self.counters['samples'] += numAdded
    self.counters['batches'] += 1
    self.counters['seeding'] += numAdded # TODO could be 1, but kept for consistency
    # prep to exit if over the limit
    if self.counters['samples'] >= self.limits['samples']:
      self.raiseADebug(f'Sampling limit reached! ({self.counters["samples"]} samples > {self.limits["samples"]} limit)')
      # TODO this is disjointed from readiness check!
    # FIXME, the following condition check is make sure that the require info is only printed once
    # when dump metadata to xml, this should be removed in the future when we have a better way to
    # dump the metadata
    if self.counters['samples'] > 1:
      for key in self.entitiesToRemove:
        self.samplerInfo.pop(key,None)
    if self.reseedAtEachIteration:
      randomUtils.randomSeed(self.counters['seeding'] - 1)
    # FIXME this may be setting the BATCH prefix, not the SAMPLE prefix
    # -> so let's move it out of this method -> TODO REMOVE
    # self.samplerInfo['prefix'] = str(self.counters['batches']) #FIXME is this useful, or should we be using the counters?

  def _performVariableTransform(self, rlzBatch):
    """
      Performs variable transformations if existing.
      @ In, rlzBatch, BatchRealization, batch of maps for vars to values
      @ Out, None
    """
    if self.variablesTransformationDict:
      for rlz in rlzBatch:
        # add latent variables and original variables to rlz.inputInfo
        for dist, var in self.variablesTransformationDict.items():
          if self.transformationMethod[dist] == 'pca':
            self.pcaTransform(rlz, var, dist)
          else:
            self.raiseAnError(NotImplementedError, f'transformation method is not yet implemented for {self.transformationMethod[dist]} method')

  def _reassignSampledVarsPbToFullyCorrVars(self, rlzBatch):
    """
      Method to reassign sampledVarsPb to the fully correlated variables
      @ In, rlzBatch, BatchRealization, batch of maps for vars to values
      @ Out, None
    """
    for rlz in rlzBatch:
      #Need keys as list because modifying rlz.inputInfo['SampledVarsPb']
      keys = list(rlz.inputInfo['SampledVarsPb'].keys())
      fullyCorrVars = {s: rlz.inputInfo['SampledVarsPb'].pop(s) for s in keys if "," in s}
      # assign the SampledVarsPb to the fully correlated vars
      for key in fullyCorrVars:
        for kkey in key.split(","):
          rlz.inputInfo['SampledVarsPb'][kkey] = fullyCorrVars[key]

  def _reassignPbWeightToCorrelatedVars(self, rlzBatch):
    """
      Method to reassign probability weight to the correlated variables
      @ In, rlzBatch, BatchRealization, batch of maps for vars to values
      @ Out, None
    """
    for rlz in rlzBatch:
      # collect initial weights
      pbWeights = {key:value for key, value in rlz.inputInfo.items() if 'ProbabilityWeight' in key}
      for varName, varInfo in self.variables2distributionsMapping.items():
        # Handle ND Case
        if varInfo['totDim'] > 1:
          distName = self.variables2distributionsMapping[varName]['name']
          pbWeights[f'ProbabilityWeight-{varName}'] = rlz.inputInfo[f'ProbabilityWeight-{distName}']
        if "," in varName:
          for subVarName in varName.split(","):
            pbWeights[f'ProbabilityWeight-{subVarName.strip()}'] = pbWeights[f'ProbabilityWeight-{varName}']
      # update pbWeights
      rlz.inputInfo.update(pbWeights)

  def generateInput(self, model, modelInput):
    """
      This method has to be overwritten to provide the specialization for the specific sampler
      The model instance in might be needed since, especially for external codes,
      only the code interface possesses the dictionary for reading the variable definition syntax
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, modelInput, list, a list of the original Step inputs for the model (e.g. files)
      @ Out, rlzBatch, RealizationBatch, list of mappings from variables to values for sample
      @ Out, modelInput, potentially perturbed? original inputs for model, or None if taken from restart
    """
    if model is not None:
      model.getAdditionalInputEdits(self.samplerInfo)
    ##### GENERATE SAMPLE #####
    # instantiate a batch of data carrier realizations
    batchSize = self.getBatchSize()
    rlzBatch = RealizationBatch(batchSize)
    rlzBatch.ID = self.counters['batches']
    if batchSize == 0:
      # this means the current sampler does not know how to handle batching, so fill the batch one-at-a-time
      for rlz in rlzBatch:
        self._incrementCounter()
        rlz.inputInfo['prefix'] = str(self.counters['samples'])
        self.localGenerateInput(rlz, model, modelInput)
    else:
      # this sampler knows how to handle batching, so we do it all at once
      # since the counter incrementer adds the whole batch at once, grab the initial counter value
      # so we can use it to number the samples correctly
      startPrefix = self.counters['samples']
      self._incrementCounter(numAdded=batchSize)
      for r, rlz in enumerate(rlzBatch):
        rlz.inputInfo['prefix'] = str(startPrefix + r + 1)
      self.localGenerateInput(rlzBatch, model, modelInput)
    # correlated variables
    self._reassignSampledVarsPbToFullyCorrVars(rlzBatch)
    self._reassignPbWeightToCorrelatedVars(rlzBatch)
    # variable transforms
    self._performVariableTransform(rlzBatch)
    # constants and functioned values
    self._constantVariables(rlzBatch)
    self._functionalVariables(rlzBatch)
    # ND variables
    self._formNDVariables(rlzBatch)
    # merge sampler metadata
    for rlz in rlzBatch:
      rlz.inputInfo.update(self.samplerInfo)
    # reset distribution memory
    for _, dist in self.distDict.items():
      if dist.getMemory():
        dist.reset()
    ##### CHECK RESTART #####
    # check each rlz for restart, and if so, fill its values and submit it as complete
    for r, rlz in enumerate(rlzBatch):
      _, inExisting = self._checkRestartForEvaluation(rlz)
      if inExisting is None:
        # we have a new evaluation, so check its contents for consistency
        self._checkSample(rlz)
        self.raiseADebug(f' ... Batch Sample point {r}, prefix {rlz.inputInfo["prefix"]}:')
        for var, val in rlz.items():
          self.raiseADebug(f' ... - "{var}": "{val}"')
      else:
        self.raiseADebug(f'Batch Point {r} found in restart!')
        # TODO method for getting Realization object out of DataObjects?
        restartRlz = dict((var, np.atleast_1d(inExisting[var])) for var in self.restartData.getVars() + self.restartData.getVars('indexes'))
        rlz.setRestart(restartRlz)
      # END if restart
    # END loop over rlz for restart checking
    return rlzBatch, modelInput

  def generateInputBatch(self, myInput, model, batchSize, projector=None):
    """
      this function provide a mask to create several inputs at the same time
      It call the generateInput function as many time as needed
      @ In, myInput, list, list containing one input set
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, batchSize, int, the number of input sets required
      @ In, projector, object, optional, used for adaptive sampling to provide the projection of the solution on the success metric
      @ Out, newInputs, list of list, list of the list of input sets
    """
    DEPRECATE # used? -> should be moved to using batch system!
    newInputs = []
    while self.amIreadyToProvideAnInput() and (self.counters['samples'] < batchSize):
      if projector is None:
        newInputs.append(self.generateInput(model, myInput))
      else:
        newInputs.append(self.generateInput(model ,myInput, projector))

    return newInputs

  @abc.abstractmethod
  def localGenerateInput(self, batch, model, modelInput):
    """
      This class need to be overwritten since it is here that the magic of the sampler happens.
      After this method call the self.inputInfo should be ready to be sent to the model
      @ In, batch, RealizationBatch (or Realization if not compatible), mapping of variables to values
      @ In, model, model instance, Model instance
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """

  def pcaTransform(self, rlz, varsDict, dist):
    """
      This method is used to map latent variables with respect to the model input variables
      both the latent variables and the model input variables will be stored in the realization
      @ In, varsDict, dict, dictionary contains latent and manifest variables {'latentVariables':[latentVar1,latentVar2,...], 'manifestVariables':[var1,var2,...]}
      @ In, dist, string, the distribution name associated with given variable set
      @ Out, None
    """
    latentVariablesValues = []
    listIndex = []
    manifestVariablesValues = [None] * len(varsDict['manifestVariables'])
    for index,lvar in enumerate(varsDict['latentVariables']):
      value = rlz.get(lvar)
      if lvar is not None:
        latentVariablesValues.append(value)
        listIndex.append(varsDict['latentVariablesIndex'][index])
    varName = utils.first(utils.first(self.distributions2variablesMapping[dist]).keys())
    varsValues = self.distDict[varName].pcaInverseTransform(latentVariablesValues,listIndex)
    for index1,index2 in enumerate(varsDict['manifestVariablesIndex']):
      manifestVariablesValues[index2] = varsValues[index1]
    manifestVariablesDict = dict(zip(varsDict['manifestVariables'],manifestVariablesValues))
    rlz.update(manifestVariablesDict)

  def _checkSample(self, rlz):
    """
      Checks the current sample for consistency with expected contents.
      @ In, rlz, Realization, dict-like object to fill with sample
      @ In, None
      @ Out, None
    """

  ### FINALIZING METHODS ####
  def finalizeActualSampling(self, jobObject, model, myInput):
    """
      This function is used by samplers that need to collect information from a
      finished run.
      Provides a generic interface that all samplers will use, for specifically
      handling any sub-class, the localFinalizeActualSampling should be overridden
      instead, as finalizeActualSampling provides only generic functionality
      shared by all Samplers and will in turn call the localFinalizeActualSampling
      before returning.
      @ In, jobObject, instance, an instance of a JobRunner
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
    self.localFinalizeActualSampling(jobObject, model, myInput)

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by samplers that need to collect information from the just ended run
      For example, for a Dynamic Event Tree case, this function can be used to retrieve
      the information from the just finished run of a branch in order to retrieve, for example,
      the distribution name that caused the trigger, etc.
      It is a essentially a place-holder for most of the sampler to remain compatible with the StepsCR structure
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """

  def finalizeSampler(self, failedRuns):
    """
      Method called at the end of the Step when no more samples will be taken.  Closes out sampler for step.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    self.handleFailedRuns(failedRuns)

  def handleFailedRuns(self, failedRuns):
    """
      Collects the failed runs from the Step and allows samples to handle them individually if need be.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    self.raiseADebug('===============')
    self.raiseADebug('| RUN SUMMARY |')
    self.raiseADebug('===============')
    if len(failedRuns) > 0:
      self.raiseAWarning(f'There were {len(failedRuns)} failed runs!  Run with verbosity = debug for more details.')
      for run in failedRuns:
        # FIXME: run.command no longer exists, so I am removing the printing
        # of it and the metadata for the time being, please let me know if this
        # information is critical, as it is debug info, I cannot imagine it is
        # important to keep.
        self.raiseADebug(f'  Run number {run.identifier} FAILED:')
        self.raiseADebug('      return code : ', run.getReturnCode())
    else:
      self.raiseADebug('All runs completed without returning errors.')
    self._localHandleFailedRuns(failedRuns)
    self.raiseADebug('===============')
    self.raiseADebug('  END SUMMARY  ')
    self.raiseADebug('===============')

  def _localHandleFailedRuns(self, failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns) > 0:
      self.raiseAnError(IOError, 'There were failed runs; aborting RAVEN.')

  def flush(self):
    """
      Reset Sampler attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    self.metadataKeys = set()
    self.assemblerDict = {}
    for key in self.counters:
      self.counters[key] = 0
    self.distDict = {}
    self.funcDict = {}
    self.variableFunctionExecutionList = []
