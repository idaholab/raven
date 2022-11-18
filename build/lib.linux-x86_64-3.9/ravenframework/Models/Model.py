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
Module where the base class and the specialization of different type of Model are

@author crisrab, alfoa

"""
#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import abc
import sys
import importlib
import pickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..BaseClasses import BaseEntity, Assembler, InputDataUser
from ..utils import utils
from ..utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class Model(utils.metaclass_insert(abc.ABCMeta, BaseEntity, Assembler, InputDataUser)):
  """
    A model is something that given an input will return an output reproducing some physical model
    it could as complex as a stand alone code, a reduced order model trained somehow or something
    externally build and imported by the user
  """
  @classmethod
  def loadFromPlugins(cls):
    """
      Loads plugins from factory.
      @ In, cls, uninstantiated object, class to load for
      @ Out, None
    """
    cls.plugins = importlib.import_module(".ModelPlugInFactory","ravenframework.Models")

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Model, cls).getInputSpecification()
    inputSpecification.addParam("subType", InputTypes.StringType, True)

    ## Begin alias tag
    AliasInput = InputData.parameterInputFactory("alias", contentType=InputTypes.StringType)
    AliasInput.addParam("variable", InputTypes.StringType, True)
    AliasTypeInput = InputTypes.makeEnumType("aliasType","aliasTypeType",["input","output"])
    AliasInput.addParam("type", AliasTypeInput, True)
    inputSpecification.addSub(AliasInput)
    ## End alias tag

    return inputSpecification

  validateDict                  = {}
  validateDict['Input'  ]       = []
  validateDict['Output' ]       = []
  validateDict['Sampler']       = []
  validateDict['Optimizer']     = []
  testDict                      = {}
  testDict                      = {'class':'','type':[''],'multiplicity':0,'required':False}
  #FIXME: a multiplicity value is needed to control role that can have different class
  #the possible inputs
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][0]['class'       ] = 'DataObjects'
  validateDict['Input'  ][0]['type'        ] = ['PointSet','HistorySet','DataSet']
  validateDict['Input'  ][0]['required'    ] = False
  validateDict['Input'  ][0]['multiplicity'] = 'n'
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][1]['class'       ] = 'Files'
  # FIXME there's lots of types that Files can be, so until XSD replaces this, commenting this out
  #validateDict['Input'  ][1]['type'        ] = ['']
  validateDict['Input'  ][1]['required'    ] = False
  validateDict['Input'  ][1]['multiplicity'] = 'n'
  #the possible outputs
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][0]['class'       ] = 'DataObjects'
  validateDict['Output' ][0]['type'        ] = ['PointSet','HistorySet','DataSet']
  validateDict['Output' ][0]['required'    ] = False
  validateDict['Output' ][0]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][1]['class'       ] = 'Databases'
  validateDict['Output' ][1]['type'        ] = ['NetCDF', 'HDF5']
  validateDict['Output' ][1]['required'    ] = False
  validateDict['Output' ][1]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][2]['class'       ] = 'OutStreams'
  validateDict['Output' ][2]['type'        ] = ['Plot','Print']
  validateDict['Output' ][2]['required'    ] = False
  validateDict['Output' ][2]['multiplicity'] = 'n'
  #the possible samplers
  validateDict['Sampler'].append(testDict.copy())
  validateDict['Sampler'][0]['class'       ] ='Samplers'
  validateDict['Sampler'][0]['required'    ] = False
  validateDict['Sampler'][0]['multiplicity'] = 1
  validateDict['Sampler'][0]['type']         = ['MonteCarlo',
                                                'DynamicEventTree',
                                                'Stratified',
                                                'Grid',
                                                'LimitSurfaceSearch',
                                                'AdaptiveDynamicEventTree',
                                                'FactorialDesign',
                                                'ResponseSurfaceDesign',
                                                'SparseGridCollocation',
                                                'AdaptiveSparseGrid',
                                                'Sobol',
                                                'AdaptiveSobol',
                                                'EnsembleForward',
                                                'CustomSampler',
                                                'AdaptiveMonteCarlo',
                                                'Metropolis',
                                                'AdaptiveMetropolis']
  validateDict['Optimizer'].append(testDict.copy())
  validateDict['Optimizer'][0]['class'       ] ='Optimizers'
  validateDict['Optimizer'][0]['required'    ] = False
  validateDict['Optimizer'][0]['multiplicity'] = 1
  validateDict['Optimizer'][0]['type'] = ['SPSA',
                                          'FiniteDifference',
                                          'ConjugateGradient',
                                          'SimulatedAnnealing',
                                          'GeneticAlgorithm']

  @classmethod
  def generateValidateDict(cls):
    """
      This method generate a independent copy of validateDict for the calling class
      @ In, None
      @ Out, None
    """
    cls.validateDict = copy.deepcopy(Model.validateDict)

  @classmethod
  def specializeValidateDict(cls):
    """
      This method should be overridden to describe the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    raise NotImplementedError('The class '+str(cls.__name__)+' has not implemented the method specializeValidateDict')

  @classmethod
  def localValidateMethod(cls,who,what):
    """
      This class method is called to test the compatibility of the class with its possible usage
      @ In, who, string, a string identifying the what is the role of what we are going to test (i.e. input, output etc)
      @ In, what, string, a list (or a general iterable) that will be playing the 'who' role
      @ Out, None
    """
    #counting successful matches
    if who not in cls.validateDict.keys():
      raise IOError('The role "{}" is not recognized for the entity "{}"'.format(who,cls))
    for myItemDict in cls.validateDict[who]:
      myItemDict['tempCounter'] = 0
    for anItem in what:
      anItem['found'] = False
      for tester in cls.validateDict[who]:
        if anItem['class'] == tester['class']:
          if anItem['class']=='Files':
            #FIXME Files can accept any type, including None.
            tester['tempCounter']+=1
            anItem['found']=True
            break
          else:
            if anItem['type'] in tester['type']:
              tester['tempCounter'] +=1
              anItem['found']        = True
              break
    #testing if the multiplicity of the argument is correct
    for tester in cls.validateDict[who]:
      if tester['required']==True:
        if tester['multiplicity']=='n' and tester['tempCounter']<1:
          raise IOError('The number of times class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper. At least one object must be present!')
      if tester['multiplicity']!='n' and tester['tempCounter']!=tester['multiplicity']:
        raise IOError('The number of times class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper. Number of allowable times is '+str(tester['multiplicity'])+'.Got '+str(tester['tempCounter']))
    #testing if all argument to be tested have been found
    for anItem in what:
      if anItem['found']==False:
        raise IOError('It is not possible to use '+anItem['class']+' type = ' +anItem['type']+' as '+who)
    return True

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    #if alias are defined in the input it defines a mapping between the variable names in the framework and the one for the generation of the input
    #self.alias[framework variable name] = [input code name]. For Example, for a MooseBasedApp, the alias would be self.alias['internal_variable_name'] = 'Material|Fuel|thermal_conductivity'
    self.alias    = {'input':{},'output':{}}
    # optional specification of the input, output, aux  variables  (needed in case of FMI/FMU export)
    self.__vars   = {'input': [],'output': [], 'aux': []}
    self.subType  = ''
    self.runQueue = []
    self.printTag = 'MODEL'
    self.createWorkingDir = False


  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Assembler._readMoreXML(self,xmlNode)
    try:
      self.subType = xmlNode.attrib['subType']
    except KeyError:
      self.raiseADebug("Failed in Node: "+str(xmlNode),verbostiy='silent')
      self.raiseAnError(IOError,'missed subType for the model '+self.name)
    for child in xmlNode:
      if child.tag =='alias':
        # the input would be <alias variable='internal_variable_name'>Material|Fuel|thermal_conductivity</alias>
        if 'variable' in child.attrib.keys():
          if 'type' in child.attrib.keys():
            if child.attrib['type'].lower() not in ['input','output']:
              self.raiseAnError(IOError,'the type of alias can be either "input" or "output". Got '+child.attrib['type'].lower())
            aliasType           = child.attrib['type'].lower().strip()
            complementAliasType = 'output' if aliasType == 'input' else 'input'
          else:
            self.raiseAnError(IOError,'not found the attribute "type" in the definition of one of the alias for model '+str(self.name) +' of type '+self.type)
          varFramework, varModel = child.attrib['variable'], child.text.strip()
          if varFramework in self.alias[aliasType].keys():
            self.raiseAnError(IOError,' The alias for variable ' +varFramework+' has been already inputted in model '+str(self.name) +' of type '+self.type)
          if varModel in self.alias[aliasType].values():
            self.raiseAnError(IOError,' The alias ' +varModel+' has been already used for another variable in model '+str(self.name) +' of type '+self.type)
          if varFramework in self.alias[complementAliasType].keys():
            self.raiseAnError(IOError,' The alias for variable ' +varFramework+' has been already inputted ('+complementAliasType+') in model '+str(self.name) +' of type '+self.type)
          if varModel in self.alias[complementAliasType].values():
            self.raiseAnError(IOError,' The alias ' +varModel+' has been already used ('+complementAliasType+') for another variable in model '+str(self.name) +' of type '+self.type)
          self.alias[aliasType][varFramework] = child.text.strip()
        else:
          self.raiseAnError(IOError,'not found the attribute "variable" in the definition of one of the alias for model '+str(self.name) +' of type '+self.type)
    # read local information
    self.localInputAndChecks(xmlNode)
    #################

  def _setVariableList(self, type, vars):
    """
      Method to set the variable list (input,output,aux)
      @ In, type, str, one of "input", "output", "aux"
      @ In, vars, list, the list of variables
      @ Out, None
    """
    assert(type in  self.__vars)
    self.__vars[type].extend(vars)
    self.__vars[type] = list(set(self.__vars[type]))
    # alias system
    if type in 'aux':
      return
    self._replaceVariablesNamesWithAliasSystem(self.__vars[type],type)

  def _getVariableList(self, type):
    """
      Method to get the variable list (input,output,aux)
      @ In, type, str, one of "input", "output", "aux"
      @ Out, vars, list, the list of variables
    """
    assert(type in  self.__vars)
    vars = self.__vars[type]
    return vars

  def _replaceVariablesNamesWithAliasSystem(self, sampledVars, aliasType='input', fromModelToFramework=False):
    """
      Method to convert kwargs Sampled vars with the alias system
      @ In, sampledVars, dict or list, dictionary or list that are going to be modified
      @ In, aliasType, str, optional, type of alias to be replaced
      @ In, fromModelToFramework, bool, optional, When we define aliases for some input variables, we need to be sure to convert the variable names
                                                  (if alias is of type input) coming from RAVEN (e.g. sampled variables) into the corresponding names
                                                  of the model (e.g. frameworkVariableName = "wolf", modelVariableName="thermal_conductivity").
                                                  Vice versa, when we define aliases for some model output variables, we need to convert the variable
                                                  names coming from the model into the one that are used in RAVEN (e.g. modelOutputName="00001111",
                                                  frameworkVariableName="clad_temperature"). The fromModelToFramework bool flag controls this action
                                                  (if True, we convert the name in the dictionary from the model names to the RAVEN names, False vice versa)
      @ Out, originalVariables, dict or list, dictionary (or list) of the original sampled variables
    """
    if aliasType =='inout':
      listAliasType = ['input','output']
    else:
      listAliasType = [aliasType]
    originalVariables = copy.deepcopy(sampledVars)
    for aliasTyp in listAliasType:
      for varFramework,varModel in self.alias[aliasTyp].items():
        whichVar =  varModel if fromModelToFramework else varFramework
        notFound = 2**62
        if type(originalVariables).__name__ != 'list':
          found = sampledVars.pop(whichVar,[notFound])
          if not np.array_equal(np.asarray(found), [notFound]):
            if fromModelToFramework:
              sampledVars[varFramework] = originalVariables[varModel]
            else:
              sampledVars[varModel]     = originalVariables[varFramework]
        else:
          if whichVar in sampledVars:
            sampledVars[sampledVars.index(whichVar)] = varFramework if fromModelToFramework else varModel
    return originalVariables

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    if "subType" in paramInput.parameterValues:
      self.subType = paramInput.parameterValues["subType"]
    else:
      self.raiseADebug(" Failed in Node: "+str(xmlNode),verbostiy='silent')
      self.raiseAnError(IOError,'missed subType for the model '+self.name)

  @abc.abstractmethod
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, returnValue, tuple(input,dict), This holds the output information of the evaluated sample.
    """
    pass

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    pass

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
    paramDict['subType'] = self.subType
    for key, value in self.alias['input'].items():
      paramDict['The model input variable '+str(value)+' is filled using the framework variable '] = key
    for key, value in self.alias['output'].items():
      paramDict['The model output variable '+str(value)+' is filled using the framework variable '] = key
    return paramDict

  def localGetInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    return paramDict

  def initialize(self,runInfo,inputs,initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    pass

  def serialize(self,fileObjIn,**kwargs):
    """
      This method is the base class method that is aimed to serialize the model (and derived) instances.
      @ In, fileObjIn, str or File object, the filename of the output serialized binary file or the RAVEN File instance
      @ In, kwargs, dict, dictionary of options that the derived class might require
      @ Out, None
    """
    import cloudpickle
    if isinstance(fileObjIn,str):
      fileObj = open(filename, mode='wb+')
    else:
      fileObj = fileObjIn # if issues occur add 'isintance(fileObjIn,Files)'.
      fileObj.open(mode='wb+')
    cloudpickle.dump(self,fileObj, protocol=pickle.HIGHEST_PROTOCOL)
    fileObj.flush()
    fileObj.close()

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, [(kwargs)], list, return the new input in a list form
    """
    return [(copy.copy(kwargs))]

  def submit(self, myInput, samplerType, jobHandler, **kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler. Note, some parameters are needed by createNewInput
        and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, None
    """
    nRuns = 1
    batchMode =  kwargs.get("batchMode", False)
    if batchMode:
      nRuns = kwargs["batchInfo"]['nRuns']

    for index in range(nRuns):
      if batchMode:
        kw =  kwargs['batchInfo']['batchRealizations'][index]
      else:
        kw = kwargs

      prefix = kw.get("prefix")
      uniqueHandler = kw.get("uniqueHandler",'any')
      forceThreads = kw.get("forceThreads",False)

      ## These kw are updated by createNewInput, so the job either should not
      ## have access to the metadata, or it needs to be updated from within the
      ## evaluateSample function, which currently is not possible since that
      ## function does not know about the job instance.
      metadata = kw

      ## This may look a little weird, but due to how the parallel python library
      ## works, we are unable to pass a member function as a job because the
      ## pp library loses track of what self is, so instead we call it from the
      ## class and pass self in as the first parameter
      jobHandler.addJob((self, myInput, samplerType, kw), self.__class__.evaluateSample, prefix, metadata=metadata,
                        uniqueHandler=uniqueHandler, forceUseThreads=forceThreads,
                        groupInfo={'id': kwargs['batchInfo']['batchId'], 'size': nRuns} if batchMode else None)

  def addOutputFromExportDictionary(self,exportDict,output,options,jobIdentifier):
    """
      Method that collects the outputs from them export dictionary
      @ In, exportDict, dict, dictionary containing the output/input values: {'inputSpaceParams':dict(sampled variables),
                                                                              'outputSpaceParams':dict(output variables),
                                                                              'metadata':dict(metadata)}
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ In, jobIdentifier, str, job identifier
      @ Out, None
    """
    if output.type == 'HDF5':
      optionsIn = {'group':self.name+str(jobIdentifier)}
      if options is not None:
        optionsIn.update(options)
      output.addGroupDataObjects(optionsIn,exportDict,False)
    else:
      self.collectOutputFromDict(exportDict,output,options)

  def collectOutput(self,collectFrom,storeTo,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, collectFrom, InternalRunner object, instance of the run just finished
      @ In, storeTo, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    #if a addOutput is present in nameSpace of storeTo it is used
    if 'addOutput' in dir(storeTo):
      storeTo.addOutput(collectFrom)
    else:
      self.raiseAnError(IOError,'The place where we want to store the output has no addOutput method!')

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input.  By default does nothing.
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    pass

  def getSerializationFiles(self):
    """
      Returns a list of any files that this needs if it is serialized
      @ In, None
      @ Out, serializationFiles, set, set of filenames that are needed
    """
    serializationFiles = set()
    return serializationFiles

