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
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import numpy as np
import abc
import sys
import importlib
import inspect
#import atexit
import time
import threading
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
import CustomCommandExecuter
from utils import utils
from utils import mathUtils
from utils import TreeStructure
from utils import graphStructure
from utils import InputData
from utils.cached_ndarray import c1darray
import Files
import PostProcessors
import LearningGate
#Internal Modules End--------------------------------------------------------------------------------


class Model(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
  """
    A model is something that given an input will return an output reproducing some physical model
    it could as complex as a stand alone code, a reduced order model trained somehow or something
    externally build and imported by the user
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
    inputSpecification = super(Model, cls).getInputSpecification()
    inputSpecification.addParam("subType", InputData.StringType, True)

    ## Begin alias tag
    AliasInput = InputData.parameterInputFactory("alias", contentType=InputData.StringType)
    AliasInput.addParam("variable", InputData.StringType, True)
    AliasTypeInput = InputData.makeEnumType("aliasType","aliasTypeType",["input","output"])
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
  validateDict['Input'  ][0]['type'        ] = ['PointSet','HistorySet']
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
  validateDict['Output' ][0]['type'        ] = ['PointSet','HistorySet']
  validateDict['Output' ][0]['required'    ] = False
  validateDict['Output' ][0]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][1]['class'       ] = 'Databases'
  validateDict['Output' ][1]['type'        ] = ['HDF5']
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
                                                'CustomSampler']
  validateDict['Optimizer'].append(testDict.copy())
  validateDict['Optimizer'][0]['class'       ] ='Optimizers'
  validateDict['Optimizer'][0]['required'    ] = False
  validateDict['Optimizer'][0]['multiplicity'] = 1
  validateDict['Optimizer'][0]['type']         = ['SPSA']

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
    if who not in cls.validateDict.keys(): raise IOError('The role '+str(who)+' does not exist in the class '+str(cls))
    for myItemDict in cls.validateDict[who]: myItemDict['tempCounter'] = 0
    for anItem in what:
      anItem['found'] = False
      for tester in cls.validateDict[who]:
        if anItem['class'] == tester['class']:
          if anItem['class']=='Files': #FIXME Files can accept any type, including None.
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
        raise IOError('It is not possible to use '+anItem['class']+' type= ' +anItem['type']+' as '+who)
    return True

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    BaseType.__init__(self)
    Assembler.__init__(self)
    #if alias are defined in the input it defines a mapping between the variable names in the framework and the one for the generation of the input
    #self.alias[framework variable name] = [input code name]. For Example, for a MooseBasedApp, the alias would be self.alias['internal_variable_name'] = 'Material|Fuel|thermal_conductivity'
    self.alias    = {'input':{},'output':{}}
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
    try: self.subType = xmlNode.attrib['subType']
    except KeyError:
      self.raiseADebug(" Failed in Node: "+str(xmlNode),verbostiy='silent')
      self.raiseAnError(IOError,'missed subType for the model '+self.name)
    for child in xmlNode:
      if child.tag =='alias':
        # the input would be <alias variable='internal_variable_name'>Material|Fuel|thermal_conductivity</alias>
        if 'variable' in child.attrib.keys():
          if 'type' in child.attrib.keys():
            if child.attrib['type'].lower() not in ['input','output']: self.raiseAnError(IOError,'the type of alias can be either "input" or "output". Got '+child.attrib['type'].lower())
            aliasType           = child.attrib['type'].lower().strip()
            complementAliasType = 'output' if aliasType == 'input' else 'input'
          else: self.raiseAnError(IOError,'not found the attribute "type" in the definition of one of the alias for model '+str(self.name) +' of type '+self.type)
          varFramework, varModel = child.attrib['variable'], child.text.strip()
          if varFramework in self.alias[aliasType].keys(): self.raiseAnError(IOError,' The alias for variable ' +varFramework+' has been already inputted in model '+str(self.name) +' of type '+self.type)
          if varModel in self.alias[aliasType].values()  : self.raiseAnError(IOError,' The alias ' +varModel+' has been already used for another variable in model '+str(self.name) +' of type '+self.type)
          if varFramework in self.alias[complementAliasType].keys(): self.raiseAnError(IOError,' The alias for variable ' +varFramework+' has been already inputted ('+complementAliasType+') in model '+str(self.name) +' of type '+self.type)
          if varModel in self.alias[complementAliasType].values()  : self.raiseAnError(IOError,' The alias ' +varModel+' has been already used ('+complementAliasType+') for another variable in model '+str(self.name) +' of type '+self.type)
          self.alias[aliasType][varFramework] = child.text.strip()
        else: self.raiseAnError(IOError,'not found the attribute "variable" in the definition of one of the alias for model '+str(self.name) +' of type '+self.type)
    # read local information
    self.localInputAndChecks(xmlNode)

  def _replaceVariablesNamesWithAliasSystem(self, sampledVars, aliasType='input', fromModelToFramework=False):
    """
      Method to convert kwargs Sampled vars with the alias system
      @ In , sampledVars, dict, dictionary that are going to be modified
      @ In, aliasType, str, optional, type of alias to be replaced
      @ In, fromModelToFramework, bool, optional, When we define aliases for some input variables, we need to be sure to convert the variable names
                                                  (if alias is of type input) coming from RAVEN (e.g. sampled variables) into the corresponding names
                                                  of the model (e.g. frameworkVariableName = "wolf", modelVariableName="thermal_conductivity").
                                                  Viceversa, when we define aliases for some model output variables, we need to convert the variable
                                                  names coming from the model into the one that are used in RAVEN (e.g. modelOutputName="00001111",
                                                  frameworkVariableName="clad_temperature"). The fromModelToFramework bool flag controls this action
                                                  (if True, we convert the name in the dictionary from the model names to the RAVEN names, False vice versa)
      @ Out, originalVariables, dict, dictionary of the original sampled variables
    """
    if aliasType =='inout': listAliasType = ['input','output']
    else                  : listAliasType = [aliasType]
    originalVariables = copy.deepcopy(sampledVars)
    for aliasTyp in listAliasType:
      if len(self.alias[aliasTyp].keys()) != 0:
        for varFramework,varModel in self.alias[aliasTyp].items():
          whichVar =  varModel if fromModelToFramework else varFramework
          found = sampledVars.pop(whichVar,[sys.maxint])
          if not np.array_equal(np.asarray(found), [sys.maxint]):
            if fromModelToFramework: sampledVars[varFramework] = originalVariables[varModel]
            else                   : sampledVars[varModel]     = originalVariables[varFramework]
    return originalVariables

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    if "subType" in paramInput.parameterValues:
      self.subType = paraminput.parameterValues["subType"]
    else:
      self.raiseADebug(" Failed in Node: "+str(xmlNode),verbostiy='silent')
      self.raiseAnError(IOError,'missed subType for the model '+self.name)

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

  def finalizeModelOutput(self,finishedJob):
    """
      Method that is aimed to finalize (preprocess) the output of a model before the results get collected
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ Out, None
    """
    pass

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

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    pass

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, [(Kwargs)], list, return the new input in a list form
    """
    return [(copy.copy(Kwargs))]

  @abc.abstractmethod
  def run(self,Input,jobHandler):
    """
      Method that performs the actual run of the Code model
      @ In,  Input, object, object contained the data to process. (inputToInternal output)
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    pass

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

class Dummy(Model):
  """
    This is a dummy model that just return the effect of the sampler. The values reported as input in the output
    are the output of the sampler and the output is the counter of the performed sampling
  """
  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.admittedData = self.__class__.validateDict['Input' ][0]['type'] #the list of admitted data is saved also here for run time checks
    #the following variable are reset at each call of the initialize method
    self.printTag = 'DUMMY MODEL'

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['type'        ] = ['PointSet']
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet']

  def _manipulateInput(self,dataIn):
    """
      Method that is aimed to manipulate the input in order to return a common input understandable by this class
      @ In, dataIn, object, the object that needs to be manipulated
      @ Out, inRun, dict, the manipulated input
    """
    if len(dataIn)>1: self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name '+self.name)
    if type(dataIn[0])!=tuple: inRun = self._inputToInternal(dataIn[0]) #this might happen when a single run is used and the input it does not come from self.createNewInput
    else:                      inRun = dataIn[0][0]
    return inRun

  def _inputToInternal(self,dataIN,full=False):
    """
      Transform it in the internal format the provided input. dataIN could be either a dictionary (then nothing to do) or one of the admitted data
      @ In, dataIn, object, the object that needs to be manipulated
      @ In, full, bool, optional, does the full input needs to be retrieved or just the last element?
      @ Out, localInput, dict, the manipulated input
    """
    #self.raiseADebug('wondering if a dictionary compatibility should be kept','FIXME')
    if  type(dataIN).__name__ !='dict':
      if dataIN.type not in self.admittedData: self.raiseAnError(IOError,self,'type "'+dataIN.type+'" is not compatible with the model "' + self.type + '" named "' + self.name+'"!')
    if type(dataIN)!=dict:
      localInput = dict.fromkeys(dataIN.getParaKeys('inputs' )+dataIN.getParaKeys('outputs' ),None)
      if not dataIN.isItEmpty():
        if dataIN.type == 'PointSet':
          for entries in dataIN.getParaKeys('inputs' ): localInput[entries] = copy.copy(np.array(dataIN.getParam('input' ,entries))[0 if full else -1:])
          for entries in dataIN.getParaKeys('outputs'): localInput[entries] = copy.copy(np.array(dataIN.getParam('output',entries))[0 if full else -1:])
        else:
          if full:
            for hist in range(len(dataIN)):
              realization = dataIN.getRealization(hist)
              for entries in dataIN.getParaKeys('inputs' ):
                if localInput[entries] is None: localInput[entries] = c1darray(shape=(1,))
                localInput[entries].append(realization['inputs'][entries])
              for entries in dataIN.getParaKeys('outputs' ):
                if localInput[entries] is None: localInput[entries] = []
                localInput[entries].append(realization['outputs'][entries])
          else:
            realization = dataIn.getRealization(len(dataIn)-1)
            for entries in dataIN.getParaKeys('inputs' ):  localInput[entries] = [realization['inputs'][entries]]
            for entries in dataIN.getParaKeys('outputs' ): localInput[entries] = [realization['outputs'][entries]]

      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in dataIN.getParaKeys('outputs'): localInput.pop('OutputPlaceHolder') # this remove the counter from the inputs to be placed among the outputs
    else: localInput = dataIN #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
    return localInput

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries are copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(Kwargs)), tuple, return the new input in a tuple form
    """
    if len(myInput)>1:
      self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name'+self.name)

    inputDict   = self._inputToInternal(myInput[0])
    self._replaceVariablesNamesWithAliasSystem(inputDict,'input',False)

    if 'SampledVars' in Kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(Kwargs['SampledVars'],'input',False)

    for key in Kwargs['SampledVars'].keys():
      inputDict[key] = np.atleast_1d(Kwargs['SampledVars'][key])
    for val in inputDict.values():
      if val is None: self.raiseAnError(IOError,'While preparing the input for the model '+self.type+' with name '+self.name+' found a None input variable '+ str(inputDict.items()))
    #the inputs/outputs should not be store locally since they might be used as a part of a list of input for the parallel runs
    #same reason why it should not be used the value of the counter inside the class but the one returned from outside as a part of the input
    if 'SampledVars' in Kwargs.keys() and len(self.alias['input'].keys()) != 0: Kwargs['SampledVars'] = sampledVars
    return [(inputDict)],copy.deepcopy(Kwargs)

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    inputOut = Input
    for key, value in externalDict.items():
      inputOut[0][0][key] =  externalDict[key]
      inputOut[1]["SampledVars"  ][key] =  externalDict[key]
      inputOut[1]["SampledVarsPb"][key] =  1.0    #FIXME it is a mistake (Andrea). The SampledVarsPb for this variable should be transfred from outside
      self._replaceVariablesNamesWithAliasSystem(inputOut[1]["SampledVars"  ],'input',False)
      self._replaceVariablesNamesWithAliasSystem(inputOut[1]["SampledVarsPb"],'input',False)
    return inputOut

  def run(self,Input,jobHandler):
    """
      This method executes the model .
      @ In,  Input, object, object contained the data to process. (inputToInternal output)
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    #this set of test is performed to avoid that if used in a single run we come in with the wrong input structure since the self.createNewInput is not called
    inRun = self._manipulateInput(Input[0])

    def lambdaReturnOut(inRun,prefix):
      """
        This method is the one is going to be submitted through the jobHandler
        @ In, inRun, dict, the input
        @ In, prefix, string, the string identifying this job
        @ Out, lambdaReturnOut, dict, the return dictionary
      """
      return {'OutputPlaceHolder':np.atleast_1d(np.float(prefix))}

    uniqueHandler = Input[1]['uniqueHandler'] if 'uniqueHandler' in Input[1].keys() else 'any'
    jobHandler.addInternal((inRun,Input[1]['prefix']),lambdaReturnOut,str(Input[1]['prefix']),metadata=Input[1], modulesToImport = self.mods, uniqueHandler=uniqueHandler)

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      self.raiseAnError(AttributeError,"No available Output to collect")
    evaluation = finishedJob.getEvaluation()
    if type(evaluation[1]).__name__ == "tuple":
      outputeval = evaluation[1][0]
    else:
      outputeval = evaluation[1]
    exportDict = copy.deepcopy({'inputSpaceParams':evaluation[0],'outputSpaceParams':outputeval,'metadata':finishedJob.getMetadata()})
    self._replaceVariablesNamesWithAliasSystem(exportDict['inputSpaceParams'], 'input',True)
    if output.type == 'HDF5':
      optionsIn = {'group':self.name+str(finishedJob.identifier)}
      if options is not None: optionsIn.update(options)
      self._replaceVariablesNamesWithAliasSystem(exportDict['inputSpaceParams'], 'input',True)
      output.addGroupDataObjects(optionsIn,exportDict,False)
    else:
      self.collectOutputFromDict(exportDict,output,options)

  def collectOutputFromDict(self,exportDict,output,options=None):
    """
      Collect results from a dictionary
      @ In, exportDict, dict, contains 'inputSpaceParams','outputSpaceParams','metadata'
      @ In, output, DataObject, to whom we write the data
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    #prefix is not generally useful for dummy-related models, so we remove it but store it
    if 'prefix' in exportDict.keys():
      prefix = exportDict.pop('prefix')
    #check for name usage, depends on where it comes from
    if 'inputSpaceParams' in exportDict.keys():
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
    else:
      inKey = 'inputs'
      outKey = 'outputs'
    if not set(output.getParaKeys('inputs') + output.getParaKeys('outputs')).issubset(set(list(exportDict[inKey].keys()) + list(exportDict[outKey].keys()))):
      missingParameters = set(output.getParaKeys('inputs') + output.getParaKeys('outputs')) - set(list(exportDict[inKey].keys()) + list(exportDict[outKey].keys()))
      self.raiseAnError(RuntimeError,"the model "+ self.name+" does not generate all the outputs requested in output object "+ output.name +". Missing parameters are: " + ','.join(list(missingParameters)) +".")
    for key in exportDict[inKey ]:
      if key in output.getParaKeys('inputs'):
        output.updateInputValue (key,exportDict[inKey][key],options)
    for key in exportDict[outKey]:
      if key in output.getParaKeys('outputs'):
        output.updateOutputValue(key,exportDict[outKey][key])
    for key in exportDict['metadata']:
      output.updateMetadata(key,exportDict['metadata'][key])

class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls. This one seems a bit excessive, are all of these for this class?
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ROM, cls).getInputSpecification()

    IndexSetInputType = InputData.makeEnumType("indexSet","indexSetType",["TensorProduct","TotalDegree","HyperbolicCross","Custom"])
    CriterionInputType = InputData.makeEnumType("criterion", "criterionType", ["bic","aic","gini","entropy","mse"])

    InterpolationInput = InputData.parameterInputFactory('Interpolation', contentType=InputData.StringType)
    InterpolationInput.addParam("quad", InputData.StringType, False)
    InterpolationInput.addParam("poly", InputData.StringType, False)
    InterpolationInput.addParam("weight", InputData.FloatType, False)

    inputSpecification.addSub(InputData.parameterInputFactory('Features',contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory('Target',contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("IndexPoints", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("IndexSet",IndexSetInputType))
    inputSpecification.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputData.StringType))
    inputSpecification.addSub(InterpolationInput)
    inputSpecification.addSub(InputData.parameterInputFactory("PolynomialOrder", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SobolOrder", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SparseGrid", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("persistence", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gradient", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("simplification", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("graph", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("beta", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("knn", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("partitionPredictor", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("smooth", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("bandwidth", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("p", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("SKLtype", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_iter", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("tol", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_1", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("alpha_2", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_1", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("lambda_2", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("compute_score", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("threshold_lambda", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_intercept", InputData.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("normalize", InputData.StringType))  #bool
    inputSpecification.addSub(InputData.parameterInputFactory("verbose", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("alpha", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("l1_ratio", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_iter", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("warm_start", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("positive", InputData.StringType)) #bool?
    inputSpecification.addSub(InputData.parameterInputFactory("eps", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_alphas", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("precompute", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_nonzero_coefs", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_path", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("max_n_alphas", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("criterion", CriterionInputType))
    inputSpecification.addSub(InputData.parameterInputFactory("penalty", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("dual", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("C", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("intercept_scaling", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("class_weight", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("random_state", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("cv", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shuffle", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("loss", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("epsilon", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("eta0", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("solver", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("alphas", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("scoring", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("gcv_mode", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("store_cv_values", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("learning_rate", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("power_t", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("multi_class", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("kernel", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("degree", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("gamma", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("coef0", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("probability", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("shrinking", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("cache_size", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("nu", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("code_size", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("fit_prior", InputData.StringType)) #bool
    inputSpecification.addSub(InputData.parameterInputFactory("class_prior", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("binarize", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("n_neighbors", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("weights", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("algorithm", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("leaf_size", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("metric", InputData.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("radius", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("outlier_label", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("shrink_threshold", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("priors", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("reg_param", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("splitter", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("max_features", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_depth", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_split", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("min_samples_leaf", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("max_leaf_nodes", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("regr", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("corr", InputData.StringType)) #enum?
    inputSpecification.addSub(InputData.parameterInputFactory("beta0", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("storage_mode", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("theta0", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaL", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("thetaU", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("nugget", InputData.FloatType))
    inputSpecification.addSub(InputData.parameterInputFactory("optimizer", InputData.StringType)) #enum
    inputSpecification.addSub(InputData.parameterInputFactory("random_start", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Pmax", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Pmin", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Qmax", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("Qmin", InputData.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("outTruncation", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("Fourier", InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("FourierOrder", InputData.StringType))

    #Estimators can include ROMs, and so because baseNode does a copy, this
    #needs to be after the rest of ROMInput is defined.
    EstimatorInput = InputData.parameterInputFactory('estimator', contentType=InputData.StringType, baseNode=inputSpecification)
    EstimatorInput.addParam("estimatorType", InputData.StringType, True)
    inputSpecification.addSub(EstimatorInput)

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet','HistorySet']

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.initializationOptionDict = {}          # ROM initialization options
    self.amITrained                = False      # boolean flag, is the ROM trained?
    self.supervisedEngine          = None       # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    return Dummy.updateInputFromOutside(self, Input, externalDict)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy._readMoreXML(self, xmlNode)
    self.initializationOptionDict['name'] = self.name
    paramInput = ROM.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    def tryStrParse(s):
      """
        Trys to parse if it is stringish
        @ In, s, string, possible string
        @ Out, s, string, original type, or possibly parsed string
      """
      return utils.tryParse(s) if type(s).__name__ in ['str','unicode'] else s

    for child in paramInput.subparts:
      if len(child.parameterValues) > 0:
        if child.getName() == 'alias': continue
        if child.getName() not in self.initializationOptionDict.keys(): self.initializationOptionDict[child.getName()]={}
        self.initializationOptionDict[child.getName()][child.value]=child.parameterValues
      else:
        if child.getName() == 'estimator':
          self.initializationOptionDict[child.getName()] = {}
          for node in child.subparts: self.initializationOptionDict[child.getName()][node.getName()] = tryStrParse(node.value)
        else: self.initializationOptionDict[child.getName()] = tryStrParse(child.value)
    # if working with a pickled ROM, send along that information
    if self.subType == 'pickledROM':
      self.initializationOptionDict['pickled'] = True
    self._initializeSupervisedGate(**self.initializationOptionDict)
    #the ROM is instanced and initialized
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(SupervisedLearning),True)) - set(self.mods))
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(LearningGate),True)) - set(self.mods))

  def _initializeSupervisedGate(self,**initializationOptions):
    """
      Method to initialize the supervisedGate class
      @ In, initializationOptions, dict, the initialization options
      @ Out, None
    """
    self.supervisedEngine = LearningGate.returnInstance('SupervisedGate', self.subType, self,**initializationOptions)

  def printXML(self,options={}):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself to file.
      @ In, options, dict, optional, the options to use in printing, including filename, things to print, etc.
      @ Out, None
    """
    #determine dynamic or static
    dynamic          = self.supervisedEngine.isADynamicModel
    # get pivot parameter
    pivotParameterId = self.supervisedEngine.pivotParameterId
    # establish file
    if 'filenameroot' in options.keys():
      filenameLocal = options['filenameroot']
    else:
      filenameLocal = self.name + '_dump'
    if dynamic:
      outFile = Files.returnInstance('DynamicXMLOutput',self)
    else:
      outFile = Files.returnInstance('StaticXMLOutput',self)
    outFile.initialize(filenameLocal+'.xml',self.messageHandler)
    outFile.newTree('ROM',pivotParam=pivotParameterId)
    #get all the targets the ROMs have
    ROMtargets = self.supervisedEngine.initializationOptions['Target'].split(",")
    #establish targets
    targets = options['target'].split(',') if 'target' in options.keys() else ROMtargets
    #establish sets of engines to work from
    engines = self.supervisedEngine.supervisedContainer
    #handle 'all' case
    if 'all' in targets: targets = ROMtargets
    #this loop is only 1 entry long if not dynamic
    for s,rom in enumerate(engines):
      if dynamic:
        pivotValue = self.supervisedEngine.historySteps[s]
      else:
        pivotValue = 0
      for target in targets:
        #for key,target in step.items():
        #skip the pivot param
        if target == pivotParameterId: continue
        #otherwise, if this is one of the requested keys, call engine's print method
        if target in ROMtargets:
          options['Target'] = target
          self.raiseAMessage('Printing time-like',pivotValue,'target',target,'ROM XML')
          rom.printXML(outFile,pivotValue,options)
    self.raiseADebug('Writing to XML file...')
    outFile.writeFile()
    self.raiseAMessage('ROM XML printed to "'+filenameLocal+'.xml"')

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    self.supervisedEngine.reset()
    self.amITrained   = False

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedEngine.getInitParams()
    return paramDict

  def train(self,trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.initializationOptionDict = copy.deepcopy(trainingSet.initializationOptionDict)
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.supervisedEngine         = copy.deepcopy(trainingSet.supervisedEngine)
    else:
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet, full=True))
      self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      self.supervisedEngine.train(self.trainingSet)
      self.amITrained = self.supervisedEngine.amITrained

  def confidence(self,request,target = None):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, datatype, feature coordinates (request)
      @ Out, confidenceDict, dict, the dict containing the confidence on each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM = self._inputToInternal(request)
    confidenceDict = self.supervisedEngine.confidence(inputToROM)
    return confidenceDict

  def evaluate(self,request):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ Out, outputEvaluation, dict, the dict containing the outputs for each target ({'target1':np.array(size 1 or n_ts),'target2':np.array(...)}
    """
    inputToROM       = self._inputToInternal(request)
    outputEvaluation = self.supervisedEngine.evaluate(inputToROM)
    return outputEvaluation

  def __externalRun(self,inRun):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, inRun, datatype, feature coordinates
      @ Out, returnDict, dict, the return dictionary containing the results
    """
    returnDict = self.evaluate(inRun)
    self._replaceVariablesNamesWithAliasSystem(returnDict, 'output',True)
    self._replaceVariablesNamesWithAliasSystem(inRun, 'input',True)
    return returnDict

  def run(self,Input,jobHandler):
    """
      This method executes the model ROM.
      @ In,  Input, object, object contained the data to process. (inputToInternal output)
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    inRun = self._manipulateInput(Input[0])
    uniqueHandler = Input[1]['uniqueHandler'] if 'uniqueHandler' in Input[1].keys() else 'any'
    jobHandler.addInternal((inRun,), self.__externalRun, str(Input[1]['prefix']), metadata=Input[1], modulesToImport=self.mods, uniqueHandler=uniqueHandler)
#
#
#

class ExternalModel(Dummy):
  """
    External model class: this model allows to interface with an external python module
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
    inputSpecification = super(ExternalModel, cls).getInputSpecification()
    inputSpecification.addParam("ModuleToLoad", InputData.StringType, True)
    inputSpecification.addSub(InputData.parameterInputFactory("variables", contentType=InputData.StringType))

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    #one data is needed for the input
    #cls.raiseADebug('think about how to import the roles to allowed class for the external model. For the moment we have just all')
    pass

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.sim                      = None
    self.modelVariableValues      = {}                                          # dictionary of variable values for the external module imported at runtime
    self.modelVariableType        = {}                                          # dictionary of variable types, used for consistency checks
    self.listOfRavenAwareVars     = []                                          # list of variables RAVEN needs to be aware of
    self._availableVariableTypes = ['float','bool','int','ndarray',
                                    'c1darray','float16','float32','float64',
                                    'float128','int16','int32','int64','bool8'] # available data types
    self._availableVariableTypes = self._availableVariableTypes + ['numpy.'+item for item in self._availableVariableTypes]                   # as above
    self.printTag                 = 'EXTERNAL MODEL'
    self.initExtSelf              = utils.Object()
    self.workingDir = runInfoDict['WorkingDir']

  def initialize(self,runInfo,inputs,initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    for key in self.modelVariableType.keys(): self.modelVariableType[key] = None
    if 'initialize' in dir(self.sim): self.sim.initialize(self.initExtSelf,runInfo,inputs)
    Dummy.initialize(self, runInfo, inputs)
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(self.sim)))

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(Kwargs)), tuple, return the new input in a tuple form
    """
    modelVariableValues ={}
    if 'createNewInput' in dir(self.sim):
      if 'SampledVars' in Kwargs.keys(): sampledVars = self._replaceVariablesNamesWithAliasSystem(Kwargs['SampledVars'],'input',False)
      extCreateNewInput = self.sim.createNewInput(self,myInput,samplerType,**Kwargs)
      if extCreateNewInput== None: self.raiseAnError(AttributeError,'in external Model '+self.ModuleToLoad+' the method createNewInput must return something. Got: None')
      if 'SampledVars' in Kwargs.keys() and len(self.alias['input'].keys()) != 0: Kwargs['SampledVars'] = sampledVars
      newInput = ([(extCreateNewInput)],copy.deepcopy(Kwargs))
      #return ([(extCreateNewInput)],copy.deepcopy(Kwargs)),copy.copy(modelVariableValues)
    else:
      newInput =  Dummy.createNewInput(self, myInput,samplerType,**Kwargs)
    for key in Kwargs['SampledVars'].keys(): modelVariableValues[key] = Kwargs['SampledVars'][key]
    return newInput, copy.copy(modelVariableValues)

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    dummyReturn =  Dummy.updateInputFromOutside(self,Input[0], externalDict)
    self._replaceVariablesNamesWithAliasSystem(dummyReturn[0][0],'input',False)
    inputOut = (dummyReturn,Input[1])
    for key, value in externalDict.items(): inputOut[1][key] =  externalDict[key]
    self._replaceVariablesNamesWithAliasSystem(inputOut[1],'input',False)
    return inputOut

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    #Model._readMoreXML(self, xmlNode)
    paramInput = ExternalModel.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    if 'ModuleToLoad' in paramInput.parameterValues:
      self.ModuleToLoad = paramInput.parameterValues['ModuleToLoad']
      moduleToLoadString, self.ModuleToLoad = utils.identifyIfExternalModelExists(self, self.ModuleToLoad, self.workingDir)
    else: self.raiseAnError(IOError,'ModuleToLoad not provided for module externalModule')
    # load the external module and point it to self.sim
    self.sim = utils.importFromPath(moduleToLoadString,self.messageHandler.getDesiredVerbosity(self)>1)
    # check if there are variables and, in case, load them
    for child in paramInput.subparts:
      if child.getName() =='variable':
        self.raiseAnError(IOError,'"variable" node included but has been depreciated!  Please list variables in a "variables" node instead.  Remove this message by Dec 2016.')
      elif child.getName() == 'variables':
        if len(child.parameterValues) > 0: self.raiseAnError(IOError,'the block '+child.getName()+' named '+child.value+' should not have attributes!!!!!')
        for var in child.value.split(','):
          var = var.strip()
          self.modelVariableType[var] = None
    self.listOfRavenAwareVars.extend(self.modelVariableType.keys())
    # check if there are other information that the external module wants to load
    #TODO this needs to be converted to work with paramInput
    if '_readMoreXML' in dir(self.sim): self.sim._readMoreXML(self.initExtSelf,xmlNode)

  def __externalRun(self, Input, modelVariables):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, Input, list, list of the inputs needed for running the model
      @ In, modelVariables, dict, the dictionary containing all the External Model variables
      @ Out, (outcomes,self), tuple, tuple containing the dictionary of the results (pos 0) and the self (pos 1)
    """
    externalSelf        = utils.Object()
    #self.sim=__import__(self.ModuleToLoad)
    modelVariableValues = {}
    for key in self.modelVariableType.keys(): modelVariableValues[key] = None
    for key,value in self.initExtSelf.__dict__.items():
      CustomCommandExecuter.execCommand('self.'+ key +' = copy.copy(object)',self=externalSelf,object=value)  # exec('externalSelf.'+ key +' = copy.copy(value)')
      modelVariableValues[key] = copy.copy(value)
    for key in Input.keys():
      if key in modelVariableValues.keys():
        modelVariableValues[key] = copy.copy(Input[key])
    if 'createNewInput' not in dir(self.sim):
      for key in Input.keys():
        if key in modelVariables.keys():
          modelVariableValues[key] = copy.copy(Input[key])
      for key in self.modelVariableType.keys() : CustomCommandExecuter.execCommand('self.'+ key +' = copy.copy(object["'+key+'"])',self=externalSelf,object=modelVariableValues) #exec('externalSelf.'+ key +' = copy.copy(modelVariableValues[key])')  #self.__uploadSolution()
    # only pass the variables and their values according to the model itself.
    InputDict = {}
    for key in Input.keys():
      if key in self.modelVariableType.keys():
        InputDict[key] = Input[key]
    self.sim.run(externalSelf, InputDict)
    for key in self.modelVariableType.keys()   : CustomCommandExecuter.execCommand('object["'+key+'"]  = copy.copy(self.'+key+')',self=externalSelf,object=modelVariableValues) #exec('modelVariableValues[key]  = copy.copy(externalSelf.'+key+')') #self.__pointSolution()
    for key in self.initExtSelf.__dict__.keys(): CustomCommandExecuter.execCommand('self.' +key+' = copy.copy(object.'+key+')',self=self.initExtSelf,object=externalSelf) #exec('self.initExtSelf.' +key+' = copy.copy(externalSelf.'+key+')')
    if None in self.modelVariableType.values():
      errorFound = False
      for key in self.modelVariableType.keys():
        self.modelVariableType[key] = type(modelVariableValues[key]).__name__
        if self.modelVariableType[key] not in self._availableVariableTypes:
          if not errorFound: self.raiseADebug('Unsupported type found. Available ones are: '+ str(self._availableVariableTypes).replace('[','').replace(']', ''),verbosity='silent')
          errorFound = True
          self.raiseADebug('variable '+ key+' has an unsupported type -> '+ self.modelVariableType[key],verbosity='silent')
      if errorFound: self.raiseAnError(RuntimeError,'Errors detected. See above!!')
    outcomes = dict((k, modelVariableValues[k]) for k in self.listOfRavenAwareVars)
    # check type consistency... This is needed in order to keep under control the external model... In order to avoid problems in collecting the outputs in our internal structures
    for key in self.modelVariableType.keys():
      if not (utils.typeMatch(outcomes[key],self.modelVariableType[key])):
        self.raiseAnError(RuntimeError,'type of variable '+ key + ' is ' + str(type(outcomes[key]))+' and mismatches with respect to the input ones (' + self.modelVariableType[key] +')!!!')
    self._replaceVariablesNamesWithAliasSystem(outcomes,'inout',True)
    return outcomes,self

  def run(self,Input,jobHandler):
    """
       Method that performs the actual run of the imported external model
       @ In,  Input, object, object contained the data to process. (inputToInternal output)
       @ In,  jobHandler, JobHandler instance, the global job handler instance
       @ Out, None
    """
    inRun = copy.copy(self._manipulateInput(Input[0][0]))
    uniqueHandler = Input[0][1]['uniqueHandler'] if 'uniqueHandler' in Input[0][1].keys() else 'any'
    jobHandler.addInternal((inRun,Input[1],),self.__externalRun,str(Input[0][1]['prefix']),metadata=Input[0][1], modulesToImport = self.mods,uniqueHandler=uniqueHandler)

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError,"No available Output to collect")
    instanciatedSelf = finishedJob.getEvaluation()[1][1]
    outcomes         = finishedJob.getEvaluation()[1][0]
    if output.type in ['HistorySet']:
      outputSize = -1
      for key in output.getParaKeys('outputs'):
        if key in instanciatedSelf.modelVariableType.keys():
          if outputSize == -1: outputSize = len(np.atleast_1d(outcomes[key]))
          if not utils.sizeMatch(outcomes[key],outputSize):
            self.raiseAnError(Exception,"the time series size needs to be the same for the output space in a HistorySet! Variable:"+key+". Size in the HistorySet="+str(outputSize)+".Size outputed="+str(len(np.atleast_1d(outcomes[key]))))
    Dummy.collectOutput(self, finishedJob, output, options)
#
#
#
#
class Code(Model):
  """
    This is the generic class that import an external code into the framework
  """
  CodeInterfaces = importlib.import_module("CodeInterfaces")

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(Code, cls).getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("executable", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("preexec", contentType=InputData.StringType))

    ## Begin command line arguments tag
    ClargsInput = InputData.parameterInputFactory("clargs")

    ClargsTypeInput = InputData.makeEnumType("clargsType","clargsTypeType",["text","input","output","prepend","postpend"])
    ClargsInput.addParam("type", ClargsTypeInput, True)

    ClargsInput.addParam("arg", InputData.StringType, False)
    ClargsInput.addParam("extension", InputData.StringType, False)
    inputSpecification.addSub(ClargsInput)
    ## End command line arguments tag

    ## Begin file arguments tag
    FileargsInput = InputData.parameterInputFactory("fileargs")

    FileargsTypeInput = InputData.makeEnumType("fileargsType", "fileargsTypeType",["input","output","moosevpp"])
    FileargsInput.addParam("type", FileargsTypeInput, True)

    FileargsInput.addParam("arg", InputData.StringType, False)
    FileargsInput.addParam("extension", InputData.StringType, False)
    inputSpecification.addSub(FileargsInput)
    ## End file arguments tag

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    #FIXME think about how to import the roles to allowed class for the codes. For the moment they are not specialized by executable
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][1]['class'       ] = 'Files'
    # FIXME there's lots of types that Files can be, so until XSD replaces this, commenting this out
    #validateDict['Input'  ][1]['type'        ] = ['']
    cls.validateDict['Input'  ][1]['required'    ] = False
    cls.validateDict['Input'  ][1]['multiplicity'] = 'n'

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.executable         = ''   #name of the executable (abs path)
    self.preExec            = None   #name of the pre-executable, if any
    self.oriInputFiles      = []   #list of the original input files (abs path)
    self.workingDir         = ''   #location where the code is currently running
    self.outFileRoot        = ''   #root to be used to generate the sequence of output files
    self.currentInputFiles  = []   #list of the modified (possibly) input files (abs path)
    self.codeFlags          = None #flags that need to be passed into code interfaces(if present)
    self.printTag           = 'CODE MODEL'
    self.createWorkingDir   = True

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Model._readMoreXML(self, xmlNode)
    paramInput = Code.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self.clargs={'text':'', 'input':{'noarg':[]}, 'pre':'', 'post':''} #output:''
    self.fargs={'input':{}, 'output':'', 'moosevpp':''}
    for child in paramInput.subparts:
      if child.getName() =='executable':
        self.executable = str(child.value)
      if child.getName() =='preexec':
        self.preExec = child.value
      elif child.getName() == 'clargs':
        argtype = child.parameterValues['type']      if 'type'      in child.parameterValues else None
        arg     = child.parameterValues['arg']       if 'arg'       in child.parameterValues else None
        ext     = child.parameterValues['extension'] if 'extension' in child.parameterValues else None
        if argtype == None: self.raiseAnError(IOError,'"type" for clarg not specified!')
        elif argtype == 'text':
          if ext != None: self.raiseAWarning('"text" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None: self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['text']=arg
        elif argtype == 'input':
          if ext == None: self.raiseAnError(IOError,'"extension" for clarg '+argtype+' not specified! Enter filetype to be listed for this flag.')
          if arg == None: self.clargs['input']['noarg'].append(ext)
          else:
            if arg not in self.clargs['input'].keys(): self.clargs['input'][arg]=[]
            self.clargs['input'][arg].append(ext)
        elif argtype == 'output':
          if arg == None: self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter flag for output file specification.')
          self.clargs['output'] = arg
        elif argtype == 'prepend':
          if ext != None: self.raiseAWarning('"prepend" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None: self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['pre'] = arg
        elif argtype == 'postpend':
          if ext != None: self.raiseAWarning('"postpend" nodes only accept "type" and "arg" attributes! Ignoring "extension"...')
          if arg == None: self.raiseAnError(IOError,'"arg" for clarg '+argtype+' not specified! Enter text to be used.')
          self.clargs['post'] = arg
        else: self.raiseAnError(IOError,'clarg type '+argtype+' not recognized!')
      elif child.getName() == 'fileargs':
        argtype = child.parameterValues['type']      if 'type'      in child.parameterValues else None
        arg     = child.parameterValues['arg']       if 'arg'       in child.parameterValues else None
        ext     = child.parameterValues['extension'] if 'extension' in child.parameterValues else None
        if argtype == None: self.raiseAnError(IOError,'"type" for filearg not specified!')
        elif argtype == 'input':
          if arg == None: self.raiseAnError(IOError,'filearg type "input" requires the template variable be specified in "arg" attribute!')
          if ext == None: self.raiseAnError(IOError,'filearg type "input" requires the auxiliary file extension be specified in "ext" attribute!')
          self.fargs['input'][arg]=[ext]
        elif argtype == 'output':
          if self.fargs['output']!='': self.raiseAnError(IOError,'output fileargs already specified!  You can only specify one output fileargs node.')
          if arg == None: self.raiseAnError(IOError,'filearg type "output" requires the template variable be specified in "arg" attribute!')
          self.fargs['output']=arg
        elif argtype.lower() == 'moosevpp':
          if self.fargs['moosevpp'] != '': self.raiseAnError(IOError,'moosevpp fileargs already specified!  You can only specify one moosevpp fileargs node.')
          if arg == None: self.raiseAnError(IOError,'filearg type "moosevpp" requires the template variable be specified in "arg" attribute!')
          self.fargs['moosevpp']=arg
        else: self.raiseAnError(IOError,'filearg type '+argtype+' not recognized!')
    if self.executable == '':
      self.raiseAWarning('The node "<executable>" was not found in the body of the code model '+str(self.name)+' so no code will be run...')
    else:
      if '~' in self.executable: self.executable = os.path.expanduser(self.executable)
      abspath = os.path.abspath(str(self.executable))
      if os.path.exists(abspath):
        self.executable = abspath
      else: self.raiseAMessage('not found executable '+self.executable,'ExceptedError')
    if self.preExec is not None:
      if '~' in self.preExec: self.preExec = os.path.expanduser(self.preExec)
      abspath = os.path.abspath(self.preExec)
      if os.path.exists(abspath):
        self.preExec = abspath
      else: self.raiseAMessage('not found preexec '+self.preExec,'ExceptedError')
    self.code = Code.CodeInterfaces.returnCodeInterface(self.subType,self)
    self.code.readMoreXML(xmlNode) #TODO figure out how to handle this with InputData
    self.code.setInputExtension(list(a.strip('.') for b in (c for c in self.clargs['input'].values()) for a in b))
    self.code.addInputExtension(list(a.strip('.') for b in (c for c in self.fargs ['input'].values()) for a in b))
    self.code.addDefaultExtension()

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Model.getInitParams(self)
    paramDict['executable']=self.executable
    return paramDict

  def getCurrentSetting(self):
    """
      This can be seen as an extension of getInitParams for the Code(model)
      that will return some information regarding the current settings of the
      code.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['current working directory'] = self.workingDir
    paramDict['current output file root' ] = self.outFileRoot
    paramDict['current input file'       ] = self.currentInputFiles
    paramDict['original input file'      ] = self.oriInputFiles
    return paramDict

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input.  By default does nothing.
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    inputInfo['additionalEdits']=self.fargs

  def initialize(self,runInfoDict,inputFiles,initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    for inputFile in inputFiles:
      shutil.copy(inputFile.getAbsFile(),self.workingDir)
    self.oriInputFiles = []
    for i in range(len(inputFiles)):
      self.oriInputFiles.append(copy.deepcopy(inputFiles[i]))
      self.oriInputFiles[-1].setPath(self.workingDir)
    self.currentInputFiles        = None
    self.outFileRoot              = None

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet only the last set of entries are copied.
      The copied values are returned as a dictionary back
      @ In, currentInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the SampledVars'that contains a dictionary {'name variable':value}
           also 'additionalEdits', similar dictionary for non-variables
      @ Out, createNewInput, tuple, return the new input in a tuple form
    """
    Kwargs['executable'] = self.executable
    found = False
    newInputSet = copy.deepcopy(currentInput)
    #TODO FIXME I don't think the extensions are the right way to classify files anymore, with the new Files
    #  objects.  However, this might require some updating of many Code Interfaces as well.
    for index, inputFile in enumerate(newInputSet):
      if inputFile.getExt() in self.code.getInputExtension():
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.code.getInputExtension()))
    Kwargs['outfile'] = 'out~'+newInputSet[index].getBase()
    subDirectory = os.path.join(self.workingDir,Kwargs['prefix'] if 'prefix' in Kwargs.keys() else '1')

    if not os.path.exists(subDirectory):
      os.mkdir(subDirectory)
    for index in range(len(newInputSet)):
      newInputSet[index].setPath(subDirectory)
      shutil.copy(self.oriInputFiles[index].getAbsFile(),subDirectory)
    Kwargs['subDirectory'] = subDirectory
    if 'SampledVars' in Kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(Kwargs['SampledVars'],'input',False)
    newInput    = self.code.createNewInput(newInputSet,self.oriInputFiles,samplerType,**copy.deepcopy(Kwargs))
    if 'SampledVars' in Kwargs.keys() and len(self.alias['input'].keys()) != 0: Kwargs['SampledVars'] = sampledVars
    return (newInput,Kwargs)

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    newKwargs = Input[1]
    newKwargs['SampledVars'].update(externalDict)
    # the following update should be done with the Pb value coming from the previous (in the model chain) model
    newKwargs['SampledVarsPb'].update(dict.fromkeys(externalDict.keys(),1.0))
    self._replaceVariablesNamesWithAliasSystem(newKwargs['SampledVars'  ],'input',False)
    self._replaceVariablesNamesWithAliasSystem(newKwargs['SampledVarsPb'],'input',False)
    inputOut = self.createNewInput(Input[1]['originalInput'], Input[1]['SamplerType'], **newKwargs)
    return inputOut

  def run(self,inputFiles,jobHandler):
    """
      Method that performs the actual run of the Code model
      @ In,  Input, object, object contained the data to process. (inputToInternal output)
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    self.currentInputFiles, metaData = (copy.deepcopy(inputFiles[0]),inputFiles[1]) if type(inputFiles).__name__ == 'tuple' else (inputFiles, None)
    returnedCommand = self.code.genCommand(self.currentInputFiles,self.executable, flags=self.clargs, fileArgs=self.fargs, preExec=self.preExec)
    if type(returnedCommand).__name__ != 'tuple'  : self.raiseAnError(IOError, "the generateCommand method in code interface must return a tuple")
    if type(returnedCommand[0]).__name__ != 'list': self.raiseAnError(IOError, "the first entry in tuple returned by generateCommand method needs to be a list of tuples!")
    executeCommand, self.outFileRoot = returnedCommand
    uniqueHandler = inputFiles[1]['uniqueHandler'] if 'uniqueHandler' in inputFiles[1].keys() else 'any'
    identifier    = inputFiles[1]['prefix'] if 'prefix' in inputFiles[1].keys() else None
    jobHandler.addExternal(executeCommand,self.outFileRoot,metaData.pop('subDirectory'),identifier=identifier,metadata=metaData,codePointer=self.code,uniqueHandler = uniqueHandler)
    found = False
    for index, inputFile in enumerate(self.currentInputFiles):
      if inputFile.getExt() in self.code.getInputExtension():
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.getInputExtension()))
    self.raiseAMessage('job "'+ str(identifier)  +'" submitted!')

  def finalizeModelOutput(self,finishedJob):
    """
      Method that is aimed to finalize (preprocess) the output of a model before the results get collected
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ Out, None
    """
    if 'finalizeCodeOutput' in dir(self.code):
      out = self.code.finalizeCodeOutput(finishedJob.command,finishedJob.output,finishedJob.getWorkingDir())
      if out: finishedJob.output = out

  def collectOutput(self,finishedjob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    outputFilelocation = finishedjob.getWorkingDir()
    attributes={"inputFile":self.currentInputFiles,"type":"csv","name":os.path.join(outputFilelocation,finishedjob.output+'.csv')}
    attributes['alias'] = self.alias
    metadata = finishedjob.getMetadata()
    if metadata: attributes['metadata'] = metadata
    if output.type == "HDF5"        : output.addGroup(attributes,attributes)
    elif output.type in ['PointSet','HistorySet']:
      outfile = Files.returnInstance('CSV',self)
      outfile.initialize(finishedjob.output+'.csv',self.messageHandler,path=outputFilelocation)
      output.addOutput(outfile,attributes)
      if metadata:
        for key,value in metadata.items(): output.updateMetadata(key,value,attributes)
    else: self.raiseAnError(ValueError,"output type "+ output.type + " unknown for Model Code "+self.name)

  def collectOutputFromDict(self,exportDict,output,options=None):
    """
      Collect results from dictionary
      @ In, exportDict, dict, contains 'inputs','outputs','metadata'
      @ In, output, instance, the place to write to
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    prefix = exportDict.pop('prefix')
    #convert to *spaceParams instead of inputs,outputs
    if 'inputs' in exportDict.keys():
      inp = exportDict.pop('inputs')
      exportDict['inputSpaceParams'] = inp
    if 'outputs' in exportDict.keys():
      out = exportDict.pop('outputs')
      exportDict['outputSpaceParams'] = out
    if output.type == 'HDF5':
      output.addGroupDataObjects({'group':self.name+str(prefix)},exportDict,False)
    else: #point set
      for key in exportDict['inputSpaceParams']:
        if key in output.getParaKeys('inputs'):
          output.updateInputValue(key,exportDict['inputSpaceParams'][key],options)
      for key in exportDict['outputSpaceParams']:
        if key in output.getParaKeys('outputs'):
          output.updateOutputValue(key,exportDict['outputSpaceParams'][key])
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key])
      output.numAdditionalLoadPoints += 1 #prevents consistency problems for entries from restart

class PostProcessor(Model, Assembler):
  """
    PostProcessor is an Action System. All the models here, take an input and perform an action
  """
  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input']                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input'][0]['required'    ] = False
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][1]['class'       ] = 'Databases'
    cls.validateDict['Input'  ][1]['type'        ] = ['HDF5']
    cls.validateDict['Input'  ][1]['required'    ] = False
    cls.validateDict['Input'  ][1]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][2]['class'       ] = 'DataObjects'
    cls.validateDict['Input'  ][2]['type'        ] = ['PointSet','HistorySet']
    cls.validateDict['Input'  ][2]['required'    ] = False
    cls.validateDict['Input'  ][2]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][3]['class'       ] = 'Files'
    # FIXME there's lots of types that Files can be, so until XSD replaces this, commenting this out
    #cls.validateDict['Input'  ][3]['type'        ] = ['']
    cls.validateDict['Input'  ][3]['required'    ] = False
    cls.validateDict['Input'  ][3]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][0]['class'       ] = 'Files'
    cls.validateDict['Output' ][0]['type'        ] = ['']
    cls.validateDict['Output' ][0]['required'    ] = False
    cls.validateDict['Output' ][0]['multiplicity'] = 'n'
    cls.validateDict['Output' ][1]['class'       ] = 'DataObjects'
    cls.validateDict['Output' ][1]['type'        ] = ['PointSet','HistorySet']
    cls.validateDict['Output' ][1]['required'    ] = False
    cls.validateDict['Output' ][1]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][2]['class'       ] = 'Databases'
    cls.validateDict['Output' ][2]['type'        ] = ['HDF5']
    cls.validateDict['Output' ][2]['required'    ] = False
    cls.validateDict['Output' ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][3]['class'       ] = 'OutStreams'
    cls.validateDict['Output' ][3]['type'        ] = ['Plot','Print']
    cls.validateDict['Output' ][3]['required'    ] = False
    cls.validateDict['Output' ][3]['multiplicity'] = 'n'
    cls.validateDict['Function'] = [cls.testDict.copy()]
    cls.validateDict['Function'  ][0]['class'       ] = 'Functions'
    cls.validateDict['Function'  ][0]['type'        ] = ['External','Internal']
    cls.validateDict['Function'  ][0]['required'    ] = False
    cls.validateDict['Function'  ][0]['multiplicity'] = 1
    cls.validateDict['ROM'] = [cls.testDict.copy()]
    cls.validateDict['ROM'       ][0]['class'       ] = 'Models'
    cls.validateDict['ROM'       ][0]['type'        ] = ['ROM']
    cls.validateDict['ROM'       ][0]['required'    ] = False
    cls.validateDict['ROM'       ][0]['multiplicity'] = 1
    cls.validateDict['KDD'] = [cls.testDict.copy()]
    cls.validateDict['KDD'       ][0]['class'       ] = 'Models'
    cls.validateDict['KDD'       ][0]['type'        ] = ['KDD']
    cls.validateDict['KDD'       ][0]['required'    ] = False
    cls.validateDict['KDD'       ][0]['multiplicity'] = 'n'

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.input  = {}     # input source
    self.action = None   # action
    self.workingDir = ''
    self.printTag = 'POSTPROCESSOR MODEL'

  def whatDoINeed(self):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for inquiring the class, which is implementing the method, about the kind of objects the class needs to
      be initialize. It is an abstract method -> It must be implemented in the derived class!
      NB. In this implementation, the method only calls the self.interface.whatDoINeed() method
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed (class:tuple(object type{if None, Simulation does not check the type}, object name))
    """
    return self.interface.whatDoINeed()

  def generateAssembler(self,initDict):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      NB. In this implementation, the method only calls the self.interface.generateAssembler(initDict) method
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.interface.generateAssembler(initDict)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Model._readMoreXML(self, xmlNode)
    self.interface = PostProcessors.returnInstance(self.subType,self)
    self.interface._readMoreXML(xmlNode)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Model.getInitParams(self)
    return paramDict

  def initialize(self,runInfo,inputs, initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    self.interface.initialize(runInfo, inputs, initDict)
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(PostProcessors),True)) - set(self.mods))

  def run(self,Input,jobHandler):
    """
      Method that performs the actual run of the Post-Processor model
      @ In,  Input, object, object contained the data to process. (inputToInternal output)
      @ In,  jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    if len(Input) > 0:
      jobHandler.addInternal((Input,),self.interface.run,str(0),modulesToImport = self.mods, forceUseThreads = True)
    else:
      jobHandler.addInternal((None,),self.interface.run,str(0),modulesToImport = self.mods, forceUseThreads = True)

  def collectOutput(self,finishedjob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    self.interface.collectOutput(finishedjob,output)

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries is copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, createNewInput, tuple, return the new input in a tuple form
    """
    return self.interface.inputToInternal(myInput)

class EnsembleModel(Dummy, Assembler):
  """
    EnsembleModel class. This class is aimed to create a comunication 'pipe' among different models in terms of Input/Output relation
  """
  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      Being this class an essembler class, all the Inputs
      @ In, None
      @ Out, None
    """
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][1]['class'       ] = 'DataObjects'
    cls.validateDict['Output' ][1]['type'        ] = ['PointSet']
    cls.validateDict['Output' ][1]['required'    ] = False
    cls.validateDict['Output' ][1]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][2]['class'       ] = 'Databases'
    cls.validateDict['Output' ][2]['type'        ] = ['HDF5']
    cls.validateDict['Output' ][2]['required'    ] = False
    cls.validateDict['Output' ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][3]['class'       ] = 'OutStreams'
    cls.validateDict['Output' ][3]['type'        ] = ['Plot','Print']
    cls.validateDict['Output' ][3]['required'    ] = False
    cls.validateDict['Output' ][3]['multiplicity'] = 'n'

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.modelsDictionary         = {}           # dictionary of models that are going to be assembled {'modelName':{'Input':[in1,in2,..,inN],'Output':[out1,out2,..,outN],'Instance':Instance}}
    self.activatePicard           = False
    self.printTag = 'EnsembleModel MODEL'
    self.addAssemblerObject('Model','n',True)
    self.addAssemblerObject('TargetEvaluation','n')
    self.addAssemblerObject('Input','n')
    self.tempTargetEvaluations = {}
    self.maxIterations         = 30
    self.convergenceTol        = 1.e-3
    self.initialConditions     = {}
    self.ensembleModelGraph    = None
    self.lockSystem = threading.RLock()

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy.localInputAndChecks(self, xmlNode)
    for child in xmlNode:
      if child.tag not in  ["Model","settings"]: self.raiseAnError(IOError, "Expected <Model> or <settings> tag. Got "+child.tag)
      if child.tag == 'Model':
        if 'type' not in child.attrib.keys() or 'class' not in child.attrib.keys(): self.raiseAnError(IOError, 'Tag Model must have attributes "class" and "type"')
        # get model name
        modelName = child.text.strip()
        # create space of the allowed entries
        self.modelsDictionary[modelName] = {'TargetEvaluation':None,'Instance':None,'Input':[],'metadataToTransfer':[]}
        # number of allower entries
        allowedEntriesLen = len(self.modelsDictionary[modelName].keys())
        for childChild in child:
          if childChild.tag.strip() == 'metadataToTransfer':
            # metadata that needs to be transfered from a source model into this model
            # list(metadataToTranfer, ModelSource,Alias (optional))
            if 'source' not in childChild.attrib.keys(): self.raiseAnError(IOError, 'when metadataToTransfer XML block is defined, the "source" attribute must be inputted!')
            self.modelsDictionary[modelName][childChild.tag].append([childChild.text.strip(),childChild.attrib['source'],childChild.attrib.get("alias",None)])
          else:
            try                  : self.modelsDictionary[modelName][childChild.tag].append(childChild.text.strip())
            except AttributeError: self.modelsDictionary[modelName][childChild.tag] = childChild.text.strip()
        if self.modelsDictionary[modelName].values().count(None) != 1         : self.raiseAnError(IOError, "TargetEvaluation xml block needs to be inputted!")
        if len(self.modelsDictionary[modelName]['Input']) == 0                : self.raiseAnError(IOError, "Input XML node for Model" + modelName +" has not been inputted!")
        if len(self.modelsDictionary[modelName].values()) > allowedEntriesLen : self.raiseAnError(IOError, "TargetEvaluation, Input and metadataToTransfer XML blocks are the only XML sub-blocks allowed!")
        if child.attrib['type'].strip() == "Code"                             : self.createWorkingDir = True
      if child.tag == 'settings': self.__readSettings(child)
    if len(self.modelsDictionary.keys()) < 2: self.raiseAnError(IOError, "The EnsembleModel needs at least 2 models to be constructed!")

  def __readSettings(self, xmlNode):
    """
      Method to read the ensemble model settings from XML input files
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'maxIterations'    : self.maxIterations  = int(child.text)
      if child.tag == 'tolerance'        : self.convergenceTol = float(child.text)
      if child.tag == 'initialConditions':
        for var in child:
          if "repeat" in var.attrib.keys():
            self.initialConditions[var.tag] = np.repeat([float(var.text.split()[0])], int(var.attrib['repeat'])) #np.array([float(var.text.split()[0]) for _ in range(int(var.attrib['repeat']))])
          else:
            try:
              values = var.text.split()
              self.initialConditions[var.tag] = float(values[0]) if len(values) == 1 else np.asarray([float(varValue) for varValue in values])
            except: self.raiseAnError(IOError,"unable to read text from XML node "+var.tag)

  def __findMatchingModel(self,what,subWhat):
    """
      Method to find the matching models with respect a some input/output. If not found, return None
      @ In, what, string, "Input" or "Output"
      @ In, subWhat, string, a keyword that needs to be contained in "what" for the mathching model
      @ Out, models, list, list of model names that match the key subWhat
    """
    models = []
    for key, value in self.modelsDictionary.items():
      if subWhat in value[what]: models.append(key)
    if len(models) == 0: models = None
    return models

#######################################################################################
#  To be uncommented when the execution list can be handled
#  def __getExecutionList(self, orderedNodes, allPath):
#    """
#      Method to get the execution list
#      @ In, orderedNodes, list, list of models ordered based
#                       on the input/output relationships
#      @ In, allPath, list, list of lists containing all the
#                       path from orderedNodes[0] to orderedNodes[-1]
#      @ Out, executionList, list, list of lists with the execution
#                       order ([[model1],[model2.1,model2.2],[model3], etc.]
#    """
#    numberPath    = len(allPath)
#    maxComponents = max([len(path) for path in allPath])
#
#    executionList = [ [] for _ in range(maxComponents)]
#    executionCounter = -1
#    for node in orderedNodes:
#      nodeCtn = 0
#      for path in allPath:
#        if node in path: nodeCtn +=1
#      if nodeCtn == numberPath:
#        executionCounter+=1
#        executionList[executionCounter] = [node]
#      else:
#        previousNodesInPath = []
#        for path in allPath:
#          if path.count(node) > 0: previousNodesInPath.append(path[path.index(node)-1])
#        for previousNode in previousNodesInPath:
#          if previousNode in executionList[executionCounter]:
#            executionCounter+=1
#            break
#        executionList[executionCounter].append(node)
#    return executionList
#######################################################################################

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize the EnsembleModel
      @ In, runInfo is the run info from the jobHandler
      @ In, inputs is a list containing whatever is passed with an input role in the step
      @ In, initDict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    for modelIn in self.assemblerDict['Model']:
      self.modelsDictionary[modelIn[2]]['Instance'] = modelIn[3]
      inputInstancesForModel = []
      for input in self.modelsDictionary[modelIn[2]]['Input']: inputInstancesForModel.append( self.retrieveObjectFromAssemblerDict('Input',input))
      self.modelsDictionary[modelIn[2]]['InputObject'] = inputInstancesForModel
      self.modelsDictionary[modelIn[2]]['Instance'].initialize(runInfo,inputInstancesForModel,initDict)
      for mm in self.modelsDictionary[modelIn[2]]['Instance'].mods:
        if mm not in self.mods: self.mods.append(mm)
      self.modelsDictionary[modelIn[2]]['TargetEvaluation'] = self.retrieveObjectFromAssemblerDict('TargetEvaluation',self.modelsDictionary[modelIn[2]]['TargetEvaluation'])
      self.tempTargetEvaluations[modelIn[2]]                 = copy.deepcopy(self.modelsDictionary[modelIn[2]]['TargetEvaluation'])
      self.modelsDictionary[modelIn[2]]['Input' ] = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("inputs")
      self.modelsDictionary[modelIn[2]]['Output'] = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("outputs")
    # construct chain connections
    modelsToOutputModels  = dict.fromkeys(self.modelsDictionary.keys(),None)
    # find matching models
    for modelIn in self.modelsDictionary.keys():
      outputMatch = []
      for i in range(len(self.modelsDictionary[modelIn]['Output'])):
        match = self.__findMatchingModel('Input',self.modelsDictionary[modelIn]['Output'][i])
        outputMatch.extend(match if match is not None else [])
      outputMatch = list(set(outputMatch))
      modelsToOutputModels[modelIn] = outputMatch
    # construct the ensemble model directed graph
    self.ensembleModelGraph = graphStructure.graphObject(modelsToOutputModels)
    # make some checks
    if not self.ensembleModelGraph.isConnectedNet():
      isolatedModels = self.ensembleModelGraph.findIsolatedVertices()
      self.raiseAnError(IOError, "Some models are not connected. Possible candidates are: "+' '.join(isolatedModels))
    # get all paths
    allPath = self.ensembleModelGraph.findAllUniquePaths()
    ###################################################
    # to be removed once executionList can be handled #
    self.orderList = self.ensembleModelGraph.createSingleListOfVertices(allPath)
    self.raiseAMessage("Model Execution list: "+' -> '.join(self.orderList))
    ###################################################
    ###########################################################################################
    # To be uncommented when the execution list can be handled                                #
    # if len(allPath) > 1: self.executionList = self.__getExecutionList(self.orderList,allPath) #
    # else               : self.executionList = allPath[-1]                                     #
    ###########################################################################################
    # check if Picard needs to be activated
    self.activatePicard = self.ensembleModelGraph.isALoop()
    if self.activatePicard:
      self.raiseAMessage("EnsembleModel connections determined a non-linear system. Picard's iterations activated!")
      if len(self.initialConditions.keys()) == 0: self.raiseAnError(IOError,"Picard's iterations mode activated but no intial conditions provided!")
    else                  : self.raiseAMessage("EnsembleModel connections determined a linear system. Picard's iterations not activated!")

    self.allOutputs = []
    for modelIn in self.modelsDictionary.keys():
      for modelInOut in self.modelsDictionary[modelIn]['Output']:
        if modelInOut not in self.allOutputs: self.allOutputs.append(modelInOut)
      # in case there are metadataToTransfer, let's check if the source model is executed before the one that requests info
      if self.modelsDictionary[modelIn]['metadataToTransfer']:
        indexModelIn = self.orderList.index(modelIn)
        for metadataToGet, source, _ in self.modelsDictionary[modelIn]['metadataToTransfer']:
          if self.orderList.index(source) >= indexModelIn:
            self.raiseAnError(IOError, 'In model "'+modelIn+'" the "metadataToTransfer" named "'+metadataToGet+
                                       '" is linked to the source"'+source+'" that will be executed after this model.')
    self.needToCheckInputs = True
    # write debug statements
    self.raiseAMessage("Specs of Graph Network represented by EnsembleModel:")
    self.raiseAMessage("Graph Degree Sequence is    : "+str(self.ensembleModelGraph.degreeSequence()))
    self.raiseAMessage("Graph Minimum/Maximum degree: "+str( (self.ensembleModelGraph.minDelta(), self.ensembleModelGraph.maxDelta())))
    self.raiseAMessage("Graph density/diameter      : "+str( (self.ensembleModelGraph.density(),  self.ensembleModelGraph.diameter())))

  def getInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, tempDict, dict, dictionary to be updated. {'attribute name':value}
    """
    tempDict = OrderedDict()
    tempDict['Models contained in EnsembleModel are '] = self.modelsDictionary.keys()
    for modelIn in self.modelsDictionary.keys():
      tempDict['Model '+modelIn+' TargetEvaluation is '] = self.modelsDictionary[modelIn]['TargetEvaluation']
      tempDict['Model '+modelIn+' Inputs are '] = self.modelsDictionary[modelIn]['Input']
    return tempDict

  def getCurrentSetting(self):
    """
      Function to inject the name and values of the parameters that might change during the simulation
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
    """
    paramDict = self.getInitParams()
    return paramDict

  def __selectInputSubset(self,modelName, kwargs ):
    """
      Method aimed to select the input subset for a certain model
      @ In, modelName, string, the model name
      @ In, kwargs , dict, the kwarded dictionary where the sampled vars are stored
      @ Out, selectedKwargs , dict, the subset of variables (in a swallow copy of the kwargs  dict)
    """
    selectedKwargs = copy.copy(kwargs)
    selectedKwargs['SampledVars'], selectedKwargs['SampledVarsPb'] = {}, {}
    for key in kwargs["SampledVars"].keys():
      if key in self.modelsDictionary[modelName]['Input']:
        selectedKwargs['SampledVars'][key], selectedKwargs['SampledVarsPb'][key] =  kwargs["SampledVars"][key],  kwargs["SampledVarsPb"][key] if 'SampledVarsPb' in kwargs.keys() else 1.0
    return copy.deepcopy(selectedKwargs)

  def _inputToInternal(self, myInput, sampledVarsKeys, full=False):
    """
      Transform it in the internal format the provided input. myInput could be either a dictionary (then nothing to do) or one of the admitted data
      This method is used only for the sub-models that are INTERNAL (not for Code models)
      @ In, myInput, object, the object that needs to be manipulated
      @ In, sampledVarsKeys, list, list of variables that partecipate to the sampling
      @ In, full, bool, optional, does the full input needs to be retrieved or just the last element?
      @ Out, initialConversion, dict, the manipulated input
    """
    initialConversion = Dummy._inputToInternal(self, myInput, full)
    for key in initialConversion.keys():
      if key not in sampledVarsKeys: initialConversion.pop(key)
    return initialConversion

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    # check if all the inputs of the submodule are covered by the sampled vars and Outputs of the other sub-models
    if self.needToCheckInputs: allCoveredVariables = list(set(self.allOutputs + Kwargs['SampledVars'].keys()))
    identifier                    = Kwargs['prefix']
    # global prefix
    newInputs                     = {'prefix':identifier}
    for modelIn, specs in self.modelsDictionary.items():
      if self.needToCheckInputs:
        for inp in specs['Input']:
          if inp not in allCoveredVariables: self.raiseAnError(RuntimeError,"for sub-model "+ modelIn + " the input "+inp+" has not been found among other models' outputs and sampled variables!")
      newKwargs = self.__selectInputSubset(modelIn,Kwargs)
      inputDict = [self._inputToInternal(self.modelsDictionary[modelIn]['InputObject'][0],newKwargs['SampledVars'].keys())] if specs['Instance'].type != 'Code' else  self.modelsDictionary[modelIn]['InputObject']
      # local prefix
      newKwargs['prefix'] = modelIn+utils.returnIdSeparator()+identifier
      newInputs[modelIn]  = specs['Instance'].createNewInput(inputDict,samplerType,**newKwargs)
      if specs['Instance'].type == 'Code': newInputs[modelIn][1]['originalInput'] = inputDict
    self.needToCheckInputs = False
    return copy.deepcopy(newInputs)

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError,"Job " + finishedJob.identifier +" failed!")
    inputs, out = finishedJob.getEvaluation()[:2]
    exportDict = {'inputSpaceParams':{},'outputSpaceParams':{},'metadata':{}}
    outcomes, targetEvaluations = out
    for modelIn in self.modelsDictionary.keys():
      # update TargetEvaluation
      inputsValues               = targetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
      unstructuredInputsValues   = targetEvaluations[modelIn].getParametersValues('unstructuredInputs', nodeId = 'RecontructEnding')
      outputsValues              = targetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
      metadataValues             = targetEvaluations[modelIn].getAllMetadata(nodeId = 'RecontructEnding')
      inputsValues  = inputsValues if targetEvaluations[modelIn].type != 'HistorySet' else inputsValues.values()[-1]

      if len(unstructuredInputsValues.keys()) > 0:
        if targetEvaluations[modelIn].type != 'HistorySet':
          castedUnstructuredInputsValues = {}
          for key in unstructuredInputsValues.keys():
            castedUnstructuredInputsValues[key] = copy.copy(unstructuredInputsValues[key][-1])
        else: castedUnstructuredInputsValues  =  unstructuredInputsValues.values()[-1]
        inputsValues.update(castedUnstructuredInputsValues)
      outputsValues  = outputsValues if targetEvaluations[modelIn].type != 'HistorySet' else outputsValues.values()[-1]
      for key in targetEvaluations[modelIn].getParaKeys('inputs'):
        if key not in inputsValues.keys(): self.raiseAnError(Exception,"the variable "+key+" is not in the input space of the model! Vars are:"+' '.join(inputsValues.keys()))
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateInputValue (key,inputsValues[key])
      for key in targetEvaluations[modelIn].getParaKeys('outputs'):
        if key not in outputsValues.keys(): self.raiseAnError(Exception,"the variable "+key+" is not in the output space of the model! Vars are:"+' '.join(outputsValues.keys()))
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateOutputValue (key,outputsValues[key])
      for key in metadataValues.keys():
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateMetadata(key,metadataValues[key])
      # end of update of TargetEvaluation
      for typeInfo,values in outcomes[modelIn].items():
        for key in values.keys(): exportDict[typeInfo][key] = np.asarray(values[key])
      if output.name == self.modelsDictionary[modelIn]['TargetEvaluation'].name: self.raiseAnError(RuntimeError, "The Step output can not be one of the target evaluation outputs!")
    if output.type == 'HDF5': output.addGroupDataObjects({'group':self.name+str(finishedJob.identifier)},exportDict,False)
    else:
      for key in exportDict['inputSpaceParams' ] :
        if key in output.getParaKeys('inputs') : output.updateInputValue (key,exportDict['inputSpaceParams' ][key])
      for key in exportDict['outputSpaceParams'] :
        if key in output.getParaKeys('outputs'): output.updateOutputValue(key,exportDict['outputSpaceParams'][key])
      for key in exportDict['metadata'] :  output.updateMetadata(key,exportDict['metadata'][key][-1])

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input. In this case, it calls all the getAdditionalInputEdits methods
      of the sub-models
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    for modelIn in self.modelsDictionary.keys(): self.modelsDictionary[modelIn]['Instance'].getAdditionalInputEdits(inputInfo)

  def run(self,Input,jobHandler):
    """
      Method to run the essembled model
      @ In, Input, object, object contained the data to process. (inputToInternal output)
      @ In, jobHandler, JobHandler instance, the global job handler instance
      @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods: self.mods.append(mm)
    jobHandler.addInternalClient(((copy.deepcopy(Input),jobHandler),), self.__externalRun,str(Input['prefix']))

  def __retrieveDependentOutput(self,modelIn,listOfOutputs, typeOutputs):
    """
      This method is aimed to retrieve the values of the output of the models on which the modelIn depends on
      @ In, modelIn, string, name of the model for which the dependent outputs need to be
      @ In, listOfOutputs, list, list of dictionary outputs ({modelName:dictOfOutputs})
      @ Out, dependentOutputs, dict, the dictionary of outputs the modelIn needs
    """
    dependentOutputs = {}
    for previousOutputs, outputType in zip(listOfOutputs,typeOutputs):
      if len(previousOutputs.values()) > 0:
        for input in self.modelsDictionary[modelIn]['Input']:
          if input in previousOutputs.keys():
            dependentOutputs[input] =  previousOutputs[input][-1] if outputType != 'HistorySet' else np.asarray(previousOutputs[input])
          #if input in previousOutputs.keys(): dependentOutputs[input] =  previousOutputs[input] if outputType != 'HistorySet' else np.asarray(previousOutputs[input])
    return dependentOutputs

  def __externalRun(self,inRun):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] jobHandler instance )
      @ Out, returnEvaluation, tuple, the results of the essembled model:
                               - returnEvaluation[0] dict of results from each sub-model,
                               - returnEvaluation[1] the dataObjects where the projection of each model is stored
    """
    Input, jobHandler = inRun[0], inRun[1]
    identifier = Input.pop('prefix')
    for modelIn in self.orderList: self.tempTargetEvaluations[modelIn].resetData()
    tempTargetEvaluations = copy.deepcopy(self.tempTargetEvaluations)
    residueContainer = dict.fromkeys(self.modelsDictionary.keys())
    gotOutputs       = [{}]*len(self.orderList)
    typeOutputs      = ['']*len(self.orderList)
    # if nonlinear system, initialize residue container
    if self.activatePicard:
      for modelIn in self.orderList:
        residueContainer[modelIn] = {'residue':{},'iterValues':[{}]*2}
        for out in self.modelsDictionary[modelIn]['Output']:
          residueContainer[modelIn]['residue'][out], residueContainer[modelIn]['iterValues'][0][out], residueContainer[modelIn]['iterValues'][1][out] = np.zeros(1), np.zeros(1), np.zeros(1)
    maxIterations, iterationCount = (self.maxIterations, 0) if self.activatePicard else (1 , 0)
    while iterationCount < maxIterations:
      returnDict     = {}
      iterationCount += 1
      if self.activatePicard: self.raiseAMessage("Picard's Iteration "+ str(iterationCount))
      for modelCnt, modelIn in enumerate(self.orderList):
        tempTargetEvaluations[modelIn].resetData()
        # in case there are metadataToTransfer, let's collect them from the source
        metadataToTransfer = None
        if self.modelsDictionary[modelIn]['metadataToTransfer']: metadataToTransfer = {}
        for metadataToGet, source, alias in self.modelsDictionary[modelIn]['metadataToTransfer']:
          if metadataToGet not in returnDict[source]['metadata'].keys():
            self.raiseAnError(RuntimeError,'metadata "'+metadataToGet+'" is not present among the ones available in source "'+source+'"!')
          metadataToTransfer[metadataToGet if alias is None else alias] = returnDict[source]['metadata'][metadataToGet][-1]
        # get dependent outputs
        dependentOutput = self.__retrieveDependentOutput(modelIn, gotOutputs, typeOutputs)
        # if nonlinear system, check for initial coditions
        if iterationCount == 1  and self.activatePicard:
          try:
            sampledVars = Input[modelIn][0][1]['SampledVars'].keys()
          except (IndexError,TypeError):
            sampledVars = Input[modelIn][1]['SampledVars'].keys()
          for initCondToSet in [x for x in self.modelsDictionary[modelIn]['Input'] if x not in set(dependentOutput.keys()+sampledVars)]:
            if initCondToSet in self.initialConditions.keys(): dependentOutput[initCondToSet] = self.initialConditions[initCondToSet]
            else                                             : self.raiseAnError(IOError,"No initial conditions provided for variable "+ initCondToSet)
        # set new identifiers
        try:
          Input[modelIn][0][1]['prefix']        = modelIn+utils.returnIdSeparator()+identifier
          Input[modelIn][0][1]['uniqueHandler'] = self.name+identifier
          if metadataToTransfer is not None: Input[modelIn][0][1]['metadataToTransfer'] = metadataToTransfer
        except (IndexError,TypeError):
          Input[modelIn][1]['prefix']           = modelIn+utils.returnIdSeparator()+identifier
          Input[modelIn][1]['uniqueHandler']    = self.name+identifier
          if metadataToTransfer is not None: Input[modelIn][1]['metadataToTransfer'] = metadataToTransfer
        # update input with dependent outputs
        Input[modelIn]  = self.modelsDictionary[modelIn]['Instance'].updateInputFromOutside(Input[modelIn], dependentOutput)
        nextModel = False
        while not nextModel:
          moveOn = False
          while not moveOn:
            if jobHandler.availability() > 0:
              # run the model
              self.modelsDictionary[modelIn]['Instance'].run(copy.deepcopy(Input[modelIn]),jobHandler)
              # wait until the model finishes, in order to get ready to run the subsequential one
              while not jobHandler.isThisJobFinished(modelIn+utils.returnIdSeparator()+identifier): time.sleep(1.e-3)
              nextModel, moveOn = True, True
            else: time.sleep(1.e-3)
          # get job that just finished to gather the results
          finishedRun = jobHandler.getFinished(jobIdentifier = modelIn+utils.returnIdSeparator()+identifier, uniqueHandler=self.name+identifier)
          if finishedRun[0].getEvaluation() == -1:
            # the model failed
            for modelToRemove in self.orderList:
              if modelToRemove != modelIn: jobHandler.getFinished(jobIdentifier = modelToRemove + utils.returnIdSeparator() + identifier, uniqueHandler = self.name + identifier)
            self.raiseAnError(RuntimeError,"The Model "+modelIn + " failed!")
          # get back the output in a general format
          # finalize the model (e.g. convert the output into a RAVEN understandable one)
          self.modelsDictionary[modelIn]['Instance'].finalizeModelOutput(finishedRun[0])
          # collect output in the temporary data object
          self.modelsDictionary[modelIn]['Instance'].collectOutput(finishedRun[0],tempTargetEvaluations[modelIn])
          # store the results in the working dictionaries
          returnDict[modelIn]   = {}
          responseSpace         = tempTargetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
          inputSpace            = tempTargetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
          typeOutputs[modelCnt] = tempTargetEvaluations[modelIn].type
          gotOutputs[modelCnt]  = responseSpace if typeOutputs[modelCnt] != 'HistorySet' else responseSpace.values()[-1]
          #store the results in return dictionary
          returnDict[modelIn]['outputSpaceParams'] = gotOutputs[modelCnt]
          returnDict[modelIn]['inputSpaceParams' ] = inputSpace if typeOutputs[modelCnt] != 'HistorySet' else inputSpace.values()[-1]
          returnDict[modelIn]['metadata'         ] = tempTargetEvaluations[modelIn].getAllMetadata()
          # if nonlinear system, compute the residue
          if self.activatePicard:
            residueContainer[modelIn]['iterValues'][1] = copy.copy(residueContainer[modelIn]['iterValues'][0])
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['iterValues'][0][out] = copy.copy(gotOutputs[modelCnt][out])
              if iterationCount == 1: residueContainer[modelIn]['iterValues'][1][out] = np.zeros(len(residueContainer[modelIn]['iterValues'][0][out]))
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['residue'][out] = abs(np.asarray(residueContainer[modelIn]['iterValues'][0][out]) - np.asarray(residueContainer[modelIn]['iterValues'][1][out]))
            residueContainer[modelIn]['Norm'] =  np.linalg.norm(np.asarray(residueContainer[modelIn]['iterValues'][1].values())-np.asarray(residueContainer[modelIn]['iterValues'][0].values()))
      # if nonlinear system, check the total residue and convergence
      if self.activatePicard:
        iterZero, iterOne = [],[]
        for modelIn in self.orderList:
          iterZero += residueContainer[modelIn]['iterValues'][0].values()
          iterOne  += residueContainer[modelIn]['iterValues'][1].values()
        residueContainer['TotalResidue'] = np.linalg.norm(np.asarray(iterOne)-np.asarray(iterZero))
        self.raiseAMessage("Picard's Iteration Norm: "+ str(residueContainer['TotalResidue']))
        if residueContainer['TotalResidue'] <= self.convergenceTol:
          self.raiseAMessage("Picard's Iteration converged. Norm: "+ str(residueContainer['TotalResidue']))
          break
    returnEvaluation = returnDict, tempTargetEvaluations
    return returnEvaluation

"""
 Factory
"""
__base = 'model'
__interFaceDict = {}
__interFaceDict['Dummy'         ] = Dummy
__interFaceDict['ROM'           ] = ROM
__interFaceDict['ExternalModel' ] = ExternalModel
__interFaceDict['Code'          ] = Code
__interFaceDict['PostProcessor' ] = PostProcessor
__interFaceDict['EnsembleModel' ] = EnsembleModel
#__interFaceDict                   = (__interFaceDict.items()+CodeInterfaces.__interFaceDict.items()) #try to use this and remove the code interface
__knownTypes                      = list(__interFaceDict.keys())

#here the class methods are called to fill the information about the usage of the classes
for classType in __interFaceDict.values():
  classType.generateValidateDict()
  classType.specializeValidateDict()

def addKnownTypes(newDict):
  """
    Function to add in the module dictionaries the known types
    @ In, newDict, dict, the dict of known types
    @ Out, None
  """
  for name,value in newDict.items():
    __interFaceDict[name]=value
    __knownTypes.append(name)

def knownTypes():
  """
    Return the known types
    @ In, None
    @ Out, knownTypes, list, list of known types
  """
  return __knownTypes

needsRunInfo = True

def returnInstance(Type,runInfoDict,caller):
  """
    function used to generate a Model class
    @ In, Type, string, Model type
    @ Out, returnInstance, instance, Instance of the Specialized Model class
  """
  try: return __interFaceDict[Type](runInfoDict)
  except KeyError: caller.raiseAnError(NameError,'MODELS','not known '+__base+' type '+Type)

def validate(className,role,what,caller):
  """
    This is the general interface for the validation of a model usage
    @ In, className, string, the name of the class
    @ In, role, string, the role assumed in the Step
    @ In, what, string, type of object
    @ In, caller, instance, the instance of the caller
    @ Out, None
  """
  if className in __knownTypes: return __interFaceDict[className].localValidateMethod(role,what)
  else : caller.raiseAnError(IOError,'MODELS','the class '+str(className)+' it is not a registered model')
