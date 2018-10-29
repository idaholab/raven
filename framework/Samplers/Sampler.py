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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import abc
import json
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils,randomUtils,InputData
from BaseClasses import BaseType
from Assembler import Assembler
#Internal Modules End--------------------------------------------------------------------------------

class Sampler(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
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
    inputSpecification = super(Sampler, cls).getInputSpecification()
    # FIXME the DET HybridSampler doesn't use the "name" param for the samples it creates,
    #      so we can't require the name yet
    inputSpecification.addParam("name", InputData.StringType)

    outerDistributionInput = InputData.parameterInputFactory("Distribution")
    outerDistributionInput.addParam("name", InputData.StringType)
    outerDistributionInput.addSub(InputData.parameterInputFactory("distribution", contentType=InputData.StringType))
    inputSpecification.addSub(outerDistributionInput)

    variableInput = InputData.parameterInputFactory("variable")
    variableInput.addParam("name", InputData.StringType)
    variableInput.addParam("shape", InputData.IntegerListType, required=False)
    distributionInput = InputData.parameterInputFactory("distribution", contentType=InputData.StringType)
    distributionInput.addParam("dim", InputData.IntegerType)

    variableInput.addSub(distributionInput)

    functionInput = InputData.parameterInputFactory("function", contentType=InputData.StringType)

    variableInput.addSub(functionInput)

    inputSpecification.addSub(variableInput)

    variablesTransformationInput = InputData.parameterInputFactory("variablesTransformation")
    variablesTransformationInput.addParam('distribution', InputData.StringType)

    variablesTransformationInput.addSub(InputData.parameterInputFactory("latentVariables", contentType=InputData.StringListType))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("manifestVariables", contentType=InputData.StringListType))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("manifestVariablesIndex", contentType=InputData.StringListType))
    variablesTransformationInput.addSub(InputData.parameterInputFactory("method", contentType=InputData.StringType))

    inputSpecification.addSub(variablesTransformationInput)

    constantInput = InputData.parameterInputFactory("constant", contentType=InputData.InterpretedListType)
    constantInput.addParam("name", InputData.StringType, True)
    constantInput.addParam("shape", InputData.IntegerListType, required=False)

    inputSpecification.addSub(constantInput)

    restartToleranceInput = InputData.parameterInputFactory("restartTolerance", contentType=InputData.FloatType)
    inputSpecification.addSub(restartToleranceInput)

    restartInput = InputData.parameterInputFactory("Restart", contentType=InputData.StringType)
    restartInput.addParam("type", InputData.StringType)
    restartInput.addParam("class", InputData.StringType)
    inputSpecification.addSub(restartInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)
    Assembler.__init__(self)
    self.ableToHandelFailedRuns        = False                     # is this sampler able to handle failed runs?
    self.counter                       = 0                         # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.auxcnt                        = 0                         # Aux counter of samples performed (for its usage check initialize method)
    self.limit                         = sys.maxsize               # maximum number of Samples (for example, Monte Carlo = Number of HistorySet to run, DET = Unlimited)
    self.toBeSampled                   = {}                        # Sampling mapping dictionary {'Variable Name':'name of the distribution'}
    self.dependentSample               = {}                        # Sampling mapping dictionary for dependent variables {'Variable Name':'name of the external function'}
    self.distDict                      = {}                        # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.funcDict                      = {}                        # Contains the instance of the function     to be used, it is created every time the sampler is initialized. keys are the variable names
    self.values                        = {}                        # for each variable the current value {'var name':value}
    self.variableShapes                = {}                        # stores the dimensionality of each variable by name, as tuple e.g. (2,3) for [[#,#,#],[#,#,#]]
    self.inputInfo                     = {}                        # depending on the sampler several different type of keywarded information could be present only one is mandatory, see below
    self.initSeed                      = None                      # if not provided the seed is randomly generated at the istanciation of the sampler, the step can override the seed by sending in another seed
    self.inputInfo['SampledVars'     ] = self.values               # this is the location where to get the values of the sampled variables
    self.inputInfo['SampledVarsPb'   ] = {}                        # this is the location where to get the probability of the sampled variables
    #self.inputInfo['PointProbability'] = None                      # this is the location where the point wise probability is stored (probability associated to a sampled point)
    self.inputInfo['crowDist']         = {}                        # Stores a dictionary that contains the information to create a crow distribution.  Stored as a json object
    self.constants                     = {}                        # In this dictionary
    self.reseedAtEachIteration         = False                     # Logical flag. True if every newer evaluation is performed after a new reseeding
    self.FIXME                         = False                     # FIXME flag
    self.printTag                      = self.type                 # prefix for all prints (sampler type)
    self.restartData                   = None                      # presampled points to restart from
    self.restartTolerance              = 1e-15                     # strictness with which to find matches in the restart data
    self.restartIsCompatible           = None                      # flags restart as compatible with the sampling scheme (used to speed up checking)

    self._endJobRunnable               = sys.maxsize               # max number of inputs creatable by the sampler right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)

    ######
    self.variables2distributionsMapping = {}                       # for each variable 'varName'  , the following informations are included:  'varName': {'dim': 1, 'reducedDim': 1,'totDim': 2, 'name': 'distName'} ; dim = dimension of the variable; reducedDim = dimension of the variable in the transformed space; totDim = total dimensionality of its associated distribution
    self.distributions2variablesMapping = {}                       # for each variable 'distName' , the following informations are included: 'distName': [{'var1': 1}, {'var2': 2}]} where for each var it is indicated the var dimension
    self.NDSamplingParams               = {}                       # this dictionary contains a dictionary for each ND distribution (key). This latter dictionary contains the initialization parameters of the ND inverseCDF ('initialGridDisc' and 'tolerance')
    ######
    self.addAssemblerObject('Restart' ,'-n',True)

    #used for PCA analysis
    self.variablesTransformationDict    = {}                       # for each variable 'modelName', the following informations are included: {'modelName': {latentVariables:[latentVar1, latentVar2, ...], manifestVariables:[manifestVar1,manifestVar2,...]}}
    self.transformationMethod           = {}                       # transformation method used in variablesTransformation node {'modelName':method}
    self.entitiesToRemove               = []                       # This variable is used in order to make sure the transformation info is printed once in the output xml file.

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distributions and functions.
      @ In, availableDist, dict, dict of distributions
      @ In, availableFunc, dict, dict of functions
      @ Out, None
    """
    if self.initSeed != None:
      randomUtils.randomSeed(self.initSeed)
    for key in self.toBeSampled.keys():
      if self.toBeSampled[key] not in availableDist.keys():
        self.raiseAnError(IOError,'Distribution '+self.toBeSampled[key]+' not found among available distributions (check input)!')
      self.distDict[key] = availableDist[self.toBeSampled[key]]
      self.inputInfo['crowDist'][key] = json.dumps(self.distDict[key].getCrowDistDict())
    for key,val in self.dependentSample.items():
      if val not in availableFunc.keys():
        self.raiseAnError('Function',val,'was not found among the available functions:',availableFunc.keys())
      self.funcDict[key] = availableFunc[val]
      # check if the correct method is present
      if "evaluate" not in self.funcDict[key].availableMethods():
        self.raiseAnError(IOError,'Function '+self.funcDict[key].name+' does not contain a method named "evaluate". It must be present if this needs to be used in a Sampler!')

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    self._generateDistributions(availableDist,availableFunc)

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

  def _readMoreXML(self,xmlNode):
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

  def _readMoreXMLbase(self,xmlNode):
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
        # variable for tracking if distributions or functions have been declared
        foundDistOrFunc = False
        # store variable name for re-use
        varName = child.parameterValues['name']
        # set shape if present
        if 'shape' in child.parameterValues:
          self.variableShapes[varName] = child.parameterValues['shape']
        # read subnodes
        for childChild in child.subparts:
          if childChild.getName() =='distribution':
            # can only have a distribution if doesn't already have a distribution or function
            if not foundDistOrFunc:
              foundDistOrFunc = True
            else:
              self.raiseAnError(IOError,'A sampled variable cannot have both a distribution and a function, or more than one of either!')
            # name of the distribution to sample
            toBeSampled = childChild.value
            varData={}
            varData['name']=childChild.value
            # variable dimensionality
            if 'dim' not in childChild.parameterValues:
              dim=1
            else:
              dim=childChild.parameterValues['dim']
            varData['dim']=dim
            # set up mapping for variable to distribution
            self.variables2distributionsMapping[varName] = varData
            # flag distribution as needing to be sampled
            self.toBeSampled[prefix+varName] = toBeSampled
          elif childChild.getName() == 'function':
            # can only have a function if doesn't already have a distribution or function
            if not foundDistOrFunc:
              foundDistOrFunc = True
            else:
              self.raiseAnError(IOError,'A sampled variable cannot have both a distribution and a function!')
            # function name
            toBeSampled = childChild.value
            # track variable as a functional sample
            self.dependentSample[prefix+varName] = toBeSampled

        if not foundDistOrFunc:
          self.raiseAnError(IOError,'Sampled variable',varName,'has neither a <distribution> nor <function> node specified!')

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
        if listIndex == None:
          self.raiseAWarning('Index is not provided for manifestVariables, default index will be used instead!')
          listIndex = range(len(transformationDict["manifestVariables"]))
        transformationDict["manifestVariablesIndex"] = listIndex
        self.variablesTransformationDict[child.parameterValues['distribution']] = transformationDict

      elif child.getName() == "constant":
        name,value = self._readInConstant(child)
        self.constants[name] = value

      elif child.getName() == "restartTolerance":
        self.restartTolerance = child.value

    if len(self.constants) > 0:
      # check if constant variables are also part of the sampled space. In case, error out
      if not set(self.toBeSampled.keys()).isdisjoint(self.constants.keys()):
        self.raiseAnError(IOError,"Some constant variables are also in the sampling space:" +
                                  ' '.join([i if i in self.toBeSampled.keys() else "" for i in self.constants.keys()])  )

    if self.initSeed == None:
      self.initSeed = randomUtils.randomIntegers(0,2**31,self)
    # Creation of the self.distributions2variablesMapping dictionary: {'distName': [{'variable_name1': dim1}, {'variable_name2': dim2}]}
    for variable in self.variables2distributionsMapping.keys():
      distName = self.variables2distributionsMapping[variable]['name']
      dim      = self.variables2distributionsMapping[variable]['dim']
      listElement={}
      listElement[variable] = dim
      if (distName in self.distributions2variablesMapping.keys()):
        self.distributions2variablesMapping[distName].append(listElement)
      else:
        self.distributions2variablesMapping[distName]=[listElement]

    # creation of the self.distributions2variablesIndexList dictionary:{'distName':[dim1,dim2,...,dimN]}
    self.distributions2variablesIndexList = {}
    for distName in self.distributions2variablesMapping.keys():
      positionList = []
      for var in self.distributions2variablesMapping[distName]:
        position = utils.first(var.values())
        positionList.append(position)
      if sum(set(positionList)) > 1 and len(positionList) != len(set(positionList)):
        dups = set(str(var) for var in positionList if positionList.count(var) > 1)
        self.raiseAnError(IOError,'Each of the following dimensions are assigned to multiple variables in Samplers: "{}"'.format(', '.join(dups)),
                ' associated to ND distribution ', distName, '. This is currently not allowed!')
      positionList = list(set(positionList))
      positionList.sort()
      self.distributions2variablesIndexList[distName] = positionList

    for key in self.variables2distributionsMapping.keys():
      distName = self.variables2distributionsMapping[key]['name']
      dim      = self.variables2distributionsMapping[key]['dim']
      reducedDim = self.distributions2variablesIndexList[distName].index(dim) + 1
      self.variables2distributionsMapping[key]['reducedDim'] = reducedDim  # the dimension of variable in the transformed space
      self.variables2distributionsMapping[key]['totDim'] = max(self.distributions2variablesIndexList[distName]) # We will reset the value if the node <variablesTransformation> exist in the raven input file
      if not self.variablesTransformationDict and self.variables2distributionsMapping[key]['totDim'] > 1:
        if self.variables2distributionsMapping[key]['totDim'] != len(self.distributions2variablesIndexList[distName]):
          self.raiseAnError(IOError,'The "dim" assigned to the variables insider Sampler are not correct! the "dim" should start from 1, and end with the full dimension of given distribution')

    #Checking the variables transformation
    if self.variablesTransformationDict:
      for dist,varsDict in self.variablesTransformationDict.items():
        maxDim = len(varsDict['manifestVariables'])
        listLatentElement = varsDict['latentVariables']
        if len(set(listLatentElement)) != len(listLatentElement):
          dups = set(var for var in listLatentElement if listLatentElement.count(var) > 1)
          self.raiseAnError(IOError,'The following are duplicated variables listed in the latentVariables: ' + str(dups))
        if len(set(varsDict['manifestVariables'])) != len(varsDict['manifestVariables']):
          dups = set(var for var in varsDict['manifestVariables'] if varsDict['manifestVariables'].count(var) > 1)
          self.raiseAnError(IOError,'The following are duplicated variables listed in the manifestVariables: ' + str(dups))
        if len(set(varsDict['manifestVariablesIndex'])) != len(varsDict['manifestVariablesIndex']):
          dups = set(var+1 for var in varsDict['manifestVariablesIndex'] if varsDict['manifestVariablesIndex'].count(var) > 1)
          self.raiseAnError(IOError,'The following are duplicated variables indices listed in the manifestVariablesIndex: ' + str(dups))
        listElement = self.distributions2variablesMapping[dist]
        for var in listElement:
          self.variables2distributionsMapping[utils.first(var.keys())]['totDim'] = maxDim #reset the totDim to reflect the totDim of original input space
        tempListElement = {k.strip():v for x in listElement for ks,v in x.items() for k in list(ks.strip().split(','))}
        listIndex = []
        for var in listLatentElement:
          if var not in set(tempListElement.keys()):
            self.raiseAnError(IOError, 'The variable listed in latentVariables ' + var + ' is not listed in the given distribution: ' + dist)
          listIndex.append(tempListElement[var]-1)
        if max(listIndex) > maxDim:
          self.raiseAnError(IOError,'The maximum dim = ' + str(max(listIndex)) + ' defined for latent variables is exceeded the dimension of the problem ' + str(maxDim))
        if len(set(listIndex)) != len(listIndex):
          dups = set(var+1 for var in listIndex if listIndex.count(var) > 1)
          self.raiseAnError(IOError,'Each of the following dimensions are assigned to multiple latent variables in Samplers: ' + str(dups))
        # update the index for latentVariables according to the 'dim' assigned for given var defined in Sampler
        self.variablesTransformationDict[dist]['latentVariablesIndex'] = listIndex
    return paramInput

  def _readInConstant(self,inp):
    """
      Reads in a "constant" input parameter node.
      @ In, inp, utils.InputParameter.ParameterInput, input parameter node to read from
      @ Out, name, string, name of constant
      @ Out, value, float or np.array,
    """
    value = inp.value
    name = inp.parameterValues['name']
    shape = inp.parameterValues.get('shape',None)
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
            ('Requested shape "{}" ({} entries) for constant "{}"' +\
            ' is not consistent with the provided values ({} entries)!')
            .format(shape,np.prod(shape),name,len(value)))
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
    paramDict['limit' ]        = self.limit
    paramDict['initial seed' ] = self.initSeed
    paramDict.update(self.localGetInitParams())
    return paramDict

  def initialize(self,externalSeeding=None,solutionExport=None):
    """
      This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, in goal oriented sampling (a.k.a. adaptive sampling this is where the space/point satisfying the constrains)
      @ Out, None
    """
    if self.initSeed == None:
      self.initSeed = randomUtils.randomIntegers(0,2**31,self)
    self.counter = 0
    if not externalSeeding:
      randomUtils.randomSeed(self.initSeed)       #use the sampler initialization seed
      self.auxcnt = self.initSeed
    elif externalSeeding=='continue':
      pass        #in this case the random sequence needs to be preserved
    else                              :
      randomUtils.randomSeed(externalSeeding)     #the external seeding is used
      self.auxcnt = externalSeeding
    #grab restart dataobject if it's available, then in localInitialize the sampler can deal with it.
    if 'Restart' in self.assemblerDict.keys():
      self.raiseADebug('Restart object: '+str(self.assemblerDict['Restart']))
      self.restartData = self.assemblerDict['Restart'][0][3]
      # check the right variables are in the restart
      need = set(itertools.chain(self.toBeSampled.keys(),self.dependentSample.keys()))
      if not need.issubset(set(self.restartData.getVars())):
        missing = need - set(self.restartData.getVars())
        #TODO this could be a warning, instead, but user wouldn't see it until the run was deep in
        self.raiseAnError(KeyError,'Restart data object "{}" is missing the following variables: "{}". No restart can be performed.'.format(self.restartData.name,', '.join(missing)))
      else:
        self.raiseAMessage('Restarting from '+self.restartData.name)
      # we used to check distribution consistency here, but we want to give more flexibility to using
      #   restart data, so do NOT check distributions of restart data.
    else:
      self.raiseAMessage('No restart for '+self.printTag)

    #load restart data into existing points
    # TODO do not copy data!  Read directly from restart.
    #if self.restartData is not None:
    #  if len(self.restartData) > 0:
    #    inps = self.restartData.getInpParametersValues()
    #    outs = self.restartData.getOutParametersValues()
    #    #FIXME there is no guarantee ordering is accurate between restart data and sampler
    #    inputs = list(v for v in inps.values())
    #    existingInps = zip(*inputs)
    #    outVals = zip(*list(v for v in outs.values()))
    #    self.existing = dict(zip(existingInps,outVals))

    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport != None:
      self.localInitialize(solutionExport=solutionExport)
    else:
      self.localInitialize()

    for distrib in self.NDSamplingParams:
      if distrib in self.distributions2variablesMapping:
        params = self.NDSamplingParams[distrib]
        temp = utils.first(self.distributions2variablesMapping[distrib][0].keys())
        self.distDict[temp].updateRNGParam(params)
      else:
        self.raiseAnError(IOError,'Distribution "%s" specified in distInit block of sampler "%s" does not exist!' %(distrib,self.name))

    # Store the transformation matrix in the metadata
    if self.variablesTransformationDict:
      self.entitiesToRemove = []
      for variable in self.variables2distributionsMapping.keys():
        distName = self.variables2distributionsMapping[variable]['name']
        dim      = self.variables2distributionsMapping[variable]['dim']
        totDim   = self.variables2distributionsMapping[variable]['totDim']
        if totDim > 1 and dim  == 1:
          transformDict = {}
          transformDict['type'] = self.distDict[variable.strip()].type
          transformDict['transformationMatrix'] = self.distDict[variable.strip()].transformationMatrix()
          self.inputInfo['transformation-'+distName] = transformDict
          self.entitiesToRemove.append('transformation-'+distName)

    # Register expected metadata
    meta = ['ProbabilityWeight','prefix','PointProbability']
    for var in self.toBeSampled.keys():
      meta +=  ['ProbabilityWeight-'+ key for key in var.split(",")]
    self.addMetaKeys(*meta)

  def localGetInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def localInitialize(self):
    """
      use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, None
      @ Out, None
    """
    pass

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Local method. Place here the additional reading, remember to add initial parameters in the method localGetInitParams
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    pass

  def readSamplerInit(self,xmlNode):
    """
      This method is responsible to read only the samplerInit block in the .xml file.
      This method has been moved from the base sampler class since the samplerInit block is needed only for the MC and stratified (LHS) samplers
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    #TODO, this is redundant and paramInput should be directly passed in.
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == "samplerInit":
        self.initSeed = randomUtils.randomIntegers(0,2**31,self)
        for childChild in child.subparts:
          if childChild.getName() == "limit":
            try:
              self.limit = int(childChild.value)
            except ValueError:
              self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value ' + str(childChild.value))
          if childChild.getName() == "initialSeed":
            try:
              self.initSeed = int(childChild.value)
            except ValueError:
              self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute initialSeed with value ' + str(childChild.value))
          elif childChild.getName() == "reseedEachIteration":
            if childChild.value.lower() in utils.stringsThatMeanTrue():
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
                  self.raiseAnError(IOError,'Unknown tag '+childChildChildChild.getName()+' .Available are: initialGridDisc and tolerance!')
              self.NDSamplingParams[childChildChild.parameterValues['name']] = NDdistData


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
    paramDict['counter'       ] = self.counter
    paramDict['initial seed'  ] = self.initSeed
    for key in self.inputInfo:
      if key!='SampledVars':
        paramDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys():
          paramDict['Variable: '+var+' has value'] = paramDict[key][var]
    paramDict.update(self.localGetCurrentSetting())
    return paramDict

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
  def amIreadyToProvideAnInput(self): #inLastOutput=None):
    """
      This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
      @ In, None
      @ Out, ready, bool, is this sampler ready to generate another sample?
    """
    ready = True if self.counter < self.limit else False
    ready = self.localStillReady(ready)
    return ready

  def localStillReady(self,ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    return ready

  def _checkRestartForEvaluation(self):
    """
      Checks restart data object (if any) for matching realization.
      @ In, None
      @ Out, index, int, index of matching realization in restart (None if not found)
      @ Out, inExisting, dict, matching realization (None if not found)
    """
    #check if point already exists
    if self.restartData is not None:
      index,inExisting = self.restartData.realization(matchDict=self.values,tol=self.restartTolerance,unpackXArray=True)
    else:
      index = None
      inExisting = None
    return index,inExisting

  def _constantVariables(self):
    """
      Method to set the constant variables into the inputInfo dictionary
      @ In, None
      @ Out, None
    """
    if len(self.constants) > 0:
      # we inject the constant variables into the SampledVars
      self.inputInfo['SampledVars'  ].update(self.constants)
      # we consider that CDF of the constant variables is equal to 1 (same as its Pb Weight)
      self.inputInfo['SampledVarsPb'].update(dict.fromkeys(self.constants.keys(),1.0))
      pbKey = ['ProbabilityWeight-'+key for key in self.constants.keys()]
      self.addMetaKeys(*pbKey)
      self.inputInfo.update(dict.fromkeys(['ProbabilityWeight-'+key for key in self.constants.keys()],1.0))

  def _expandVectorVariables(self):
    """
      Expands vector variables to fit the requested shape.
      @ In, None
      @ Out, None
    """
    # by default, just repeat this value into the desired shape.  May be overloaded by other samplers.
    for var,shape in self.variableShapes.items():
      baseVal = self.inputInfo['SampledVars'][var]
      self.inputInfo['SampledVars'][var] = np.ones(shape)*baseVal

  def _functionalVariables(self):
    """
      Evaluates variables that are functions of other input variables.
      @ In, None
      @ Out, None
    """
    # generate the function variable values
    for var in self.dependentSample.keys():
      test=self.funcDict[var].evaluate("evaluate",self.values)
      for corrVar in var.split(","):
        self.values[corrVar.strip()] = test

  def _incrementCounter(self):
    """
      Incrementes counter and sets up prefix.
      @ In, None
      @ Out, None
    """
    #since we are creating the input for the next run we increase the counter and global counter
    self.counter +=1
    self.auxcnt  +=1
    #exit if over the limit
    if self.counter > self.limit:
      self.raiseADebug('Exceeded number of points requested in sampling!  Moving on...')
    #FIXME, the following condition check is make sure that the require info is only printed once when dump metadata to xml, this should be removed in the future when we have a better way to dump the metadata
    if self.counter >1:
      for key in self.entitiesToRemove:
        self.inputInfo.pop(key,None)
    if self.reseedAtEachIteration:
      randomUtils.randomSeed(self.auxcnt-1)
    self.inputInfo['prefix'] = str(self.counter)

  def _performVariableTransform(self):
    """
      Performs variable transformations if existing.
      @ In, None
      @ Out, None
    """
    # add latent variables and original variables to self.inputInfo
    if self.variablesTransformationDict:
      for dist,var in self.variablesTransformationDict.items():
        if self.transformationMethod[dist] == 'pca':
          self.pcaTransform(var,dist)
        else:
          self.raiseAnError(NotImplementedError,'transformation method is not yet implemented for ' + self.transformationMethod[dist] + ' method')

  def _reassignSampledVarsPbToFullyCorrVars(self):
    """
      Method to reassign sampledVarsPb to the fully correlated variables
      @ In, None
      @ Out, None
    """
    #Need keys as list because modifying self.inputInfo['SampledVarsPb']
    keys = list(self.inputInfo['SampledVarsPb'].keys())
    fullyCorrVars = {s: self.inputInfo['SampledVarsPb'].pop(s) for s in keys if "," in s}
    # assign the SampledVarsPb to the fully correlated vars
    for key in fullyCorrVars:
      for kkey in key.split(","):
        self.inputInfo['SampledVarsPb'][kkey] = fullyCorrVars[key]

  def _reassignPbWeightToCorrelatedVars(self):
    """
      Method to reassign probability weight to the correlated variables
      @ In, None
      @ Out, None
    """
    for varName, varInfo in self.variables2distributionsMapping.items():
      # Handle ND Case
      if varInfo['totDim'] > 1:
        distName = self.variables2distributionsMapping[varName]['name']
        self.inputInfo['ProbabilityWeight-' + varName] = self.inputInfo['ProbabilityWeight-' + distName]
      if "," in varName:
        for subVarName in varName.split(","):
          self.inputInfo['ProbabilityWeight-' + subVarName.strip()] = self.inputInfo['ProbabilityWeight-' + varName]

  def generateInput(self,model,oldInput):
    """
      This method has to be overwritten to provide the specialization for the specific sampler
      The model instance in might be needed since, especially for external codes,
      only the code interface possesses the dictionary for reading the variable definition syntax
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, generateInput, tuple(0,list), list contains the new inputs -in reality it is the model that returns this; the Sampler generates the value to be placed in the input of the model.
      The Out parameter depends on the results of generateInput
        If a new point is found, the default Out above is correct.
        If a restart point is found:
          @ Out, generateInput, tuple(int,dict), (1,realization dictionary)
    """
    self._incrementCounter()
    model.getAdditionalInputEdits(self.inputInfo)
    self.localGenerateInput(model,oldInput)
    # split the sampled vars Pb among the different correlated variables
    self._reassignSampledVarsPbToFullyCorrVars()
    self._reassignPbWeightToCorrelatedVars()
    ##### TRANSFORMATION #####
    self._performVariableTransform()
    ##### CONSTANT VALUES ######
    self._constantVariables()
    ##### REDUNDANT FUNCTIONALS #####
    self._functionalVariables()
    ##### VECTOR VARS #####
    self._expandVectorVariables()
    ##### RESTART #####
    index,inExisting = self._checkRestartForEvaluation()
    # reformat metadata into acceptable format for dataojbect
    # DO NOT format here, let that happen when a realization is made in collectOutput for each Model.  Sampler doesn't care about this.
    # self.inputInfo['ProbabilityWeight'] = np.atleast_1d(self.inputInfo['ProbabilityWeight'])
    # self.inputInfo['prefix'] = np.atleast_1d(self.inputInfo['prefix'])
    #if not found or not restarting, we have a new point!
    if inExisting is None:
      self.raiseADebug('Found new point to sample:',self.values)
      ## The new info for the perturbed run will be stored in the sampler's
      ## inputInfo (I don't particularly like this, I think it should be
      ## returned here, but let's get this working and then we can decide how
      ## to best pass this information around. My reasoning is that returning
      ## it here means the sampler does not need to store it, and we can return
      ## a copy of the information, otherwise we have to be careful to create a
      ## deep copy of this information when we submit it to a job).
      ## -- DPM 4/18/17
      return 0,oldInput
    #otherwise, return the restart point
    else:
      # TODO use realization format as per new data object (no subspaces)
      self.raiseADebug('Point found in restart!')
      rlz = {}
      # we've fixed it so the input and output space don't really matter, so use restartData's own definition
      # DO format the data as atleast_1d so it's consistent in the ExternalModel for users (right?)
      rlz['inputs'] = dict((var,np.atleast_1d(inExisting[var])) for var in self.restartData.getVars('input'))
      rlz['outputs'] = dict((var,np.atleast_1d(inExisting[var])) for var in self.restartData.getVars('output')+self.restartData.getVars('indexes'))
      rlz['metadata'] = copy.deepcopy(self.inputInfo) # TODO need deepcopy only because inputInfo is on self
      return 1,rlz

  def generateInputBatch(self,myInput,model,batchSize,projector=None):
    """
      this function provide a mask to create several inputs at the same time
      It call the generateInput function as many time as needed
      @ In, myInput, list, list containing one input set
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, batchSize, int, the number of input sets required
      @ In, projector, object, optional, used for adaptive sampling to provide the projection of the solution on the success metric
      @ Out, newInputs, list of list, list of the list of input sets
    """
    newInputs = []
    while self.amIreadyToProvideAnInput() and (self.counter < batchSize):
      if projector==None:
        newInputs.append(self.generateInput(model,myInput))
      else:
        newInputs.append(self.generateInput(model,myInput,projector))
    return newInputs

  @abc.abstractmethod
  def localGenerateInput(self,model,oldInput):
    """
      This class need to be overwritten since it is here that the magic of the sampler happens.
      After this method call the self.inputInfo should be ready to be sent to the model
      @ In, model, model instance, Model instance
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    pass

  def pcaTransform(self,varsDict,dist):
    """
      This method is used to map latent variables with respect to the model input variables
      both the latent variables and the model input variables will be stored in the dict: self.inputInfo['SampledVars']
      @ In, varsDict, dict, dictionary contains latent and manifest variables {'latentVariables':[latentVar1,latentVar2,...], 'manifestVariables':[var1,var2,...]}
      @ In, dist, string, the distribution name associated with given variable set
      @ Out, None
    """
    latentVariablesValues = []
    listIndex = []
    manifestVariablesValues = [None] * len(varsDict['manifestVariables'])
    for index,lvar in enumerate(varsDict['latentVariables']):
      for var,value in self.values.items():
        if lvar == var:
          latentVariablesValues.append(value)
          listIndex.append(varsDict['latentVariablesIndex'][index])
    varName = utils.first(utils.first(self.distributions2variablesMapping[dist]).keys())
    varsValues = self.distDict[varName].pcaInverseTransform(latentVariablesValues,listIndex)
    for index1,index2 in enumerate(varsDict['manifestVariablesIndex']):
      manifestVariablesValues[index2] = varsValues[index1]
    manifestVariablesDict = dict(zip(varsDict['manifestVariables'],manifestVariablesValues))
    self.values.update(manifestVariablesDict)


  ### FINALIZING METHODS ####
  def finalizeActualSampling(self,jobObject,model,myInput):
    """
      This function is used by samplers that need to collect information from a
      finished run.
      Provides a generic interface that all samplers will use, for specifically
      handling any sub-class, the localFinalizeActualSampling should be overridden
      instead, as finalizeActualSampling provides only generic functionality
      shared by all Samplers and will in turn call the localFinalizeActualSampling
      before returning.
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
    self.localFinalizeActualSampling(jobObject,model,myInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
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
    pass

  def finalizeSampler(self,failedRuns):
    """
      Method called at the end of the Step when no more samples will be taken.  Closes out sampler for step.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    self.handleFailedRuns(failedRuns)

  def handleFailedRuns(self,failedRuns):
    """
      Collects the failed runs from the Step and allows samples to handle them individually if need be.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    self.raiseADebug('===============')
    self.raiseADebug('| RUN SUMMARY |')
    self.raiseADebug('===============')
    if len(failedRuns)>0:
      self.raiseAWarning('There were %i failed runs!  Run with verbosity = debug for more details.' %(len(failedRuns)))
      for run in failedRuns:
        ## FIXME: run.command no longer exists, so I am removing the printing
        ## of it and the metadata for the time being, please let me know if this
        ## information is critical, as it is debug info, I cannot imagine it is
        ## important to keep.
        self.raiseADebug('  Run number %s FAILED:' %run.identifier)
        self.raiseADebug('      return code :',run.getReturnCode())
        # metadata = run.getMetadata()
        # if metadata is not None:
        #   self.raiseADebug('      sampled vars:')
        #   for v,k in metadata['SampledVars'].items():
        #     self.raiseADebug('         ',v,':',k)
    else:
      self.raiseADebug('All runs completed without returning errors.')
    self._localHandleFailedRuns(failedRuns)
    self.raiseADebug('===============')
    self.raiseADebug('  END SUMMARY  ')
    self.raiseADebug('===============')

  def _localHandleFailedRuns(self,failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns)>0:
      self.raiseAnError(IOError,'There were failed runs; aborting RAVEN.')
