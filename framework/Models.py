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
import importlib
import inspect
import atexit
import time
import threading
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
import PostProcessors
import CustomCommandExecuter
import utils
import mathUtils
import TreeStructure
import Files
#Internal Modules End--------------------------------------------------------------------------------

class Model(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
  """
    A model is something that given an input will return an output reproducing some physical model
    it could as complex as a stand alone code, a reduced order model trained somehow or something
    externally build and imported by the user
  """
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
    self.subType  = ''
    self.runQueue = []
    self.printTag = 'MODEL'


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

    del(xmlNode.attrib['subType'])
    # read local information
    self.localInputAndChecks(xmlNode)

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
      this function have to return a new input that will be submitted to the model, it is called by the sampler
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
#
#
#
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
    if full==True:  length = 0
    if full==False: length = -1
    localInput = {}
    if type(dataIN)!=dict:
      for entries in dataIN.getParaKeys('inputs' ):
        if not dataIN.isItEmpty(): localInput[entries] = copy.copy(np.array(dataIN.getParam('input' ,entries))[length:])
        else:                      localInput[entries] = None
      for entries in dataIN.getParaKeys('outputs'):
        if not dataIN.isItEmpty(): localInput[entries] = copy.copy(np.array(dataIN.getParam('output',entries))[length:])
        else:                      localInput[entries] = None
      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in dataIN.getParaKeys('outputs'): localInput.pop('OutputPlaceHolder') # this remove the counter from the inputs to be placed among the outputs
    else: localInput = dataIN #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
    return localInput

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
      this function have to return a new input that will be submitted to the model, it is called by the sampler
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries are copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(Kwargs)), tuple, return the new input in a tuple form
    """
    if len(myInput)>1: self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name'+self.name)
    inputDict = self._inputToInternal(myInput[0])
    #test if all sampled variables are in the inputs category of the data
    # fixme? -congjian
    #if set(list(Kwargs['SampledVars'].keys())+list(inputDict.keys())) != set(list(inputDict.keys())):
    #  self.raiseAnError(IOError,'When trying to sample the input for the model '+self.name+' of type '+self.type+' the sampled variable are '+str(Kwargs['SampledVars'].keys())+' while the variable in the input are'+str(inputDict.keys()))
    for key in Kwargs['SampledVars'].keys(): inputDict[key] = np.atleast_1d(Kwargs['SampledVars'][key])
    for val in inputDict.values():
      if val is None: self.raiseAnError(IOError,'While preparing the input for the model '+self.type+' with name '+self.name+' found a None input variable '+ str(inputDict.items()))
    #the inputs/outputs should not be store locally since they might be used as a part of a list of input for the parallel runs
    #same reason why it should not be used the value of the counter inside the class but the one returned from outside as a part of the input
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
    jobHandler.submitDict['Internal']((inRun,Input[1]['prefix']),lambdaReturnOut,str(Input[1]['prefix']),metadata=Input[1], modulesToImport = self.mods, uniqueHandler=uniqueHandler)

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
    if output.type == 'HDF5':
      optionsIn = {'group':self.name+str(finishedJob.identifier)}
      if options is not None: optionsIn.update(options)
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
#
#
#
class ROM(Dummy):
  """
    ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome
  """
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
    self.howManyTargets            = 0          # how many targets?
    self.SupervisedEngine          = {}         # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'
    self.numberOfTimeStep          = 1
    self.historyPivotParameter     = 'none'     #time-like pivot parameter for data object on which ROM was trained
    self.historySteps              = []

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    return Dummy.updateInputFromOutside(self, Input, externalDict)

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    if not self.amITrained:
      a = state.pop("SupervisedEngine")
      del a
    return state

  def __setstate__(self, newstate):
    """
      Initialize the ROM with the data contained in newstate
      @ In, newstate, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(newstate)
    if not self.amITrained:
      if self.numberOfTimeStep > 1:
        targets = self.initializationOptionDict['Target'].split(',')
        self.howManyTargets = len(targets)
        self.SupervisedEngine = []
        for t in range(self.numberOfTimeStep):
          tempSupervisedEngine = {}
          for target in targets:
            self.initializationOptionDict['Target'] = target
            tempSupervisedEngine[target] =  SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptionDict)
            self.initializationOptionDict['Target'] = ','.join(targets)
          self.SupervisedEngine.append(tempSupervisedEngine)
      else:
        #this can't be accurate, since in readXML the 'Target' keyword is set to a single target
        targets = self.initializationOptionDict['Target'].split(',')
        self.howManyTargets = len(targets)
        self.SupervisedEngine = {}

        for target in targets:
          self.initializationOptionDict['Target'] = target
          self.SupervisedEngine[target] =  SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptionDict)
          #restore targets to initialization option dict
          self.initializationOptionDict['Target'] = ','.join(targets)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy._readMoreXML(self, xmlNode)
    self.initializationOptionDict['name'] = self.name
    for child in xmlNode:
      if child.attrib:
        if child.tag not in self.initializationOptionDict.keys():
          self.initializationOptionDict[child.tag]={}
        self.initializationOptionDict[child.tag][child.text]=child.attrib
      else:
        if child.tag == 'estimator':
          self.initializationOptionDict[child.tag] = {}
          for node in child:
            self.initializationOptionDict[child.tag][node.tag] = utils.tryParse(node.text)
        else:
          self.initializationOptionDict[child.tag] = utils.tryParse(child.text)
    #the ROM is instanced and initialized
    # check how many targets
    if not 'Target' in self.initializationOptionDict.keys(): self.raiseAnError(IOError,'No Targets specified!!!')
    targets = self.initializationOptionDict['Target'].split(',')
    self.howManyTargets = len(targets)

    if 'SKLtype' in self.initializationOptionDict and 'MultiTask' in self.initializationOptionDict['SKLtype']:
      self.initializationOptionDict['Target'] = targets
      model = SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptionDict)
      for target in targets:
        self.SupervisedEngine[target] = model
    else:
      for target in targets:
        self.initializationOptionDict['Target'] = target
        self.SupervisedEngine[target] =  SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptionDict)
    # extend the list of modules this ROM depen on
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(utils.first(self.SupervisedEngine.values())),True)) - set(self.mods))
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(SupervisedLearning),True)) - set(self.mods))
    #restore targets to initialization option dict
    self.initializationOptionDict['Target'] = ','.join(targets)

  def printXML(self,options={}):
    """
      Called by the OutStreamPrint object to cause the ROM to print itself to file.
      @ In, options, dict, optional, the options to use in printing, including filename, things to print, etc.
      @ Out, None
    """
    #determine dynamic or static
    if type(self.SupervisedEngine) == list:
      dynamic = True
    elif type(self.SupervisedEngine) == dict:
      dynamic = False
    else:
      self.raiseAnError(RuntimeError,'Unrecognized structure for self.SupervisedEngine:',type(self.SupervisedEnging))
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
    outFile.newTree('ROM',pivotParam=self.historyPivotParameter)
    #establish targets
    if 'target' in options.keys():
      targets = options['target'].split(',')
    else:
      targets = ['all']
    #establish sets of engines to work from
    if dynamic:
      engines = self.SupervisedEngine
    else:
      engines = [self.SupervisedEngine]
    #handle 'all' case
    if 'all' in targets:
      targets = engines[0].keys()
    #this loop is only 1 entry long if not dynamic
    for s,step in enumerate(engines):
      if dynamic:
        pivotValue = self.historySteps[s]
      else:
        pivotValue = 0
      for key,target in step.items():
        #skip the pivot param
        if key == self.historyPivotParameter:
          continue
        #otherwise, if this is one of the requested keys, call engine's print method
        if key in targets:
          self.raiseAMessage('Printing time',pivotValue,'target',key,'ROM XML')
          target.printXML(outFile,pivotValue,options)
    self.raiseADebug('Writing to XML file...')
    outFile.writeFile()
    self.raiseAMessage('ROM XML printed to "'+filenameLocal+'.xml"')

  def reset(self):
    """
      Reset the ROM
      @ In,  None
      @ Out, None
    """
    if type(self.SupervisedEngine).__name__ == 'list':
      for ts in self.SupervisedEngine:
        for instrom in ts.values():
          instrom.reset()
    else:
      for instrom in self.SupervisedEngine.values():
        instrom.reset()
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
    paramDict = {}
    for target, instrom in self.SupervisedEngine.items():
      paramDict[self.name + '|' + target] = instrom.returnInitialParameters()
    return paramDict

  def train(self,trainingSet):
    """
      This function train the ROM
      @ In, trainingSet, dict or PointSet or HistorySet, data used to train the ROM; if an HistorySet is provided the a list of ROM is created in order to create a temporal-ROM
      @ Out, None
    """
    if type(trainingSet).__name__ == 'ROM':
      self.howManyTargets           = copy.deepcopy(trainingSet.howManyTargets)
      self.initializationOptionDict = copy.deepcopy(trainingSet.initializationOptionDict)
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.SupervisedEngine         = copy.deepcopy(trainingSet.SupervisedEngine)
      self.historyPivotParameter    = copy.deepcopy(getattr(trainingSet,self.historyPivotParameter,'time'))
      self.historySteps             = copy.deepcopy(trainingSet.historySteps)
    else:
      if 'HistorySet' in type(trainingSet).__name__:
        #get the pivot parameter if specified
        self.historyPivotParameter = trainingSet._dataParameters.get('pivotParameter','time')
        #get the list of history steps if specified
        self.historySteps = trainingSet.getParametersValues('outputs').values()[0].get(self.historyPivotParameter,[])
        #store originals for future copying
        origRomCopies = {}
        for target,engine in self.SupervisedEngine.items():
          origRomCopies[target] = copy.deepcopy(engine)
        #clear engines for time-based storage
        self.SupervisedEngine = []
        outKeys = trainingSet.getParaKeys('outputs')
        targets = origRomCopies.keys()
        # check that all histories have the same length
        tmp = trainingSet.getParametersValues('outputs')
        for t in tmp:
          if t==1:
            self.numberOfTimeStep = len(tmp[t][outKeys[0]])
          else:
            if self.numberOfTimeStep != len(tmp[t][outKeys[0]]):
              self.raiseAnError(IOError,'DataObject can not be used to train a ROM: length of HistorySet is not consistent')
        # train the ROM
        self.trainingSet = mathUtils.historySetWindow(trainingSet,self.numberOfTimeStep)
        for ts in range(self.numberOfTimeStep):
          newRom = {}
          for target in targets:
            newRom[target] =  copy.deepcopy(origRomCopies[target])
          for target,instrom in newRom.items():
            # train the ROM
            instrom.train(self.trainingSet[ts])
            self.amITrained = self.amITrained and instrom.amITrained
          self.SupervisedEngine.append(newRom)
        self.amITrained = True
      else:
        self.trainingSet = copy.copy(self._inputToInternal(trainingSet,full=True))
        if type(self.trainingSet) is dict:
          self.amITrained = True
          for instrom in self.SupervisedEngine.values():
            instrom.train(self.trainingSet)
            self.amITrained = self.amITrained and instrom.amITrained
          self.raiseADebug('add self.amITrained to currentParamters','FIXME')

  def confidence(self,request,target = None):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, datatype, feature coordinates (request)
      @ In, target, string, optional, target name (by default the first target entered in the input file)
    """
    inputToROM = self._inputToInternal(request)
    if target != None:
      if type(self.SupervisedEngine) is list:
        confDic = {}
        for ts in self.SupervisedEngine:
          confDic[ts] = ts[target].confidence(inputToROM)
        return confDic
      else:
        return self.SupervisedEngine[target].confidence(inputToROM)
    else:
      if type(self.SupervisedEngine) is list:
        confDic = {}
        for ts in self.SupervisedEngine:
          confDic[ts] = ts.values()[0].confidence(inputToROM)
        return confDic
      else:
        return self.SupervisedEngine.values()[0].confidence(inputToROM)

  def evaluate(self,request, target = None, timeInst = None):
    """
      When the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
      @ In, request, datatype, feature coordinates (request)
      @ In, target, string, optional, target name (by default the first target entered in the input file)
      @ In, timeInst, int, element of the temporal ROM to evaluate
    """
    inputToROM = self._inputToInternal(request)
    if target != None:
      if timeInst == None:
        return self.SupervisedEngine[target].evaluate(inputToROM)
      else:
        return self.SupervisedEngine[timeInst][target].evaluate(inputToROM)
    else:
      if timeInst == None:
        return utils.first(self.SupervisedEngine.values()).evaluate(inputToROM)
      else:
        return self.SupervisedEngine[timeInst].values()[0].evaluate(inputToROM)

  def __externalRun(self,inRun):
    """
      Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
      @ In, inRun, datatype, feature coordinates
      @ Out, returnDict, dict, the return dictionary containing the results
    """
    returnDict = {}
    if type(self.SupervisedEngine).__name__ == 'list':
      targets = self.SupervisedEngine[0].keys()
      for target in targets:
        returnDict[target] = np.zeros(0)
      for ts in range(len(self.SupervisedEngine)):
        for target in targets:
          returnDict[target] = np.append(returnDict[target],self.evaluate(inRun,target,ts))
    else:
      for target in self.SupervisedEngine.keys():
        returnDict[target] = self.evaluate(inRun,target)
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
    jobHandler.submitDict['Internal']((inRun,), self.__externalRun, str(Input[1]['prefix']), metadata=Input[1], modulesToImport=self.mods, uniqueHandler=uniqueHandler)
#
#
#
class ExternalModel(Dummy):
  """
    External model class: this model allows to interface with an external python module
  """
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
      this function have to return a new input that will be submitted to the model, it is called by the sampler
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(Kwargs)), tuple, return the new input in a tuple form
    """
    modelVariableValues ={}
    for key in Kwargs['SampledVars'].keys(): modelVariableValues[key] = Kwargs['SampledVars'][key]
    if 'createNewInput' in dir(self.sim):
      extCreateNewInput = self.sim.createNewInput(self,myInput,samplerType,**Kwargs)
      if extCreateNewInput== None: self.raiseAnError(AttributeError,'in external Model '+self.ModuleToLoad+' the method createNewInput must return something. Got: None')
      return ([(extCreateNewInput)],copy.deepcopy(Kwargs)),copy.copy(modelVariableValues)
    else: return Dummy.createNewInput(self, myInput,samplerType,**Kwargs),copy.copy(modelVariableValues)

  def updateInputFromOutside(self, Input, externalDict):
    """
      Method to update an input from outside
      @ In, Input, list, list of inputs that needs to be updated
      @ In, externalDict, dict, dictionary of new values that need to be added or updated
      @ Out, inputOut, list, updated list of inputs
    """
    dummyReturn =  Dummy.updateInputFromOutside(self,Input[0], externalDict)
    inputOut = (dummyReturn,Input[1])
    for key, value in externalDict.items(): inputOut[1][key] =  externalDict[key]
    return inputOut

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    #Model._readMoreXML(self, xmlNode)
    if 'ModuleToLoad' in xmlNode.attrib.keys():
      self.ModuleToLoad = str(xmlNode.attrib['ModuleToLoad'])
      moduleToLoadString, self.ModuleToLoad = utils.identifyIfExternalModelExists(self, self.ModuleToLoad, self.workingDir)
    else: self.raiseAnError(IOError,'ModuleToLoad not provided for module externalModule')
    # load the external module and point it to self.sim
    self.sim = utils.importFromPath(moduleToLoadString,self.messageHandler.getDesiredVerbosity(self)>1)
    # check if there are variables and, in case, load them
    for son in xmlNode:
      if son.tag=='variable':
        self.raiseAnError(IOError,'"variable" node included but has been depreciated!  Please list variables in a "variables" node instead.  Remove this message by Dec 2016.')
      elif son.tag=='variables':
        if len(son.attrib.keys()) > 0: self.raiseAnError(IOError,'the block '+son.tag+' named '+son.text+' should not have attributes!!!!!')
        for var in son.text.split(','):
          var = var.strip()
          self.modelVariableType[var] = None
          self.listOfRavenAwareVars.append(var)
    # check if there are other information that the external module wants to load
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
    jobHandler.submitDict['Internal']((inRun,Input[1],),self.__externalRun,str(Input[0][1]['prefix']),metadata=Input[0][1], modulesToImport = self.mods,uniqueHandler=uniqueHandler)

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      #is it still possible for the run to not be finished yet?  Should we be erroring out if so?
      self.raiseAnError(RuntimeError,"No available Output to collect")
    def typeMatch(var,varTypeStr):
      """
        This method is aimed to check if a variable changed datatype
        @ In, var, python datatype, the first variable to compare
        @ In, varTypeStr, string, the type that this variable should have
        @ Out, typeMatch, bool, is the datatype changed?
      """
      typeVar = type(var)
      return typeVar.__name__ == varTypeStr or \
        typeVar.__module__+"."+typeVar.__name__ == varTypeStr
    def sizeMatch(var,sizeToCheck):
      """
        This method is aimed to check if a variable has an expected size
        @ In, var, python datatype, the first variable to compare
        @ In, sizeToCheck, int, the size this variable should have
        @ Out, sizeMatched, bool, is the size ok?
      """
      sizeMatched = True
      if len(np.atleast_1d(var)) != sizeToCheck: sizeMatched = False
      return sizeMatched
    # check type consistency... This is needed in order to keep under control the external model... In order to avoid problems in collecting the outputs in our internal structures
    instanciatedSelf = finishedJob.getEvaluation()[1][1]
    outcomes         = finishedJob.getEvaluation()[1][0]
    for key in instanciatedSelf.modelVariableType.keys():
      if not (typeMatch(outcomes[key],instanciatedSelf.modelVariableType[key])):
        self.raiseAnError(RuntimeError,'type of variable '+ key + ' is ' + str(type(outcomes[key]))+' and mismatches with respect to the input ones (' + instanciatedSelf.modelVariableType[key] +')!!!')
    for key in instanciatedSelf.modelVariableType.keys():
      if not (typeMatch(outcomes[key],instanciatedSelf.modelVariableType[key])):
        self.raiseAnError(RuntimeError,'type of variable '+ key + ' is ' + str(type(outcomes[key]))+' and mismatches with respect to the input ones (' + instanciatedSelf.modelVariableType[key] +')!!!')

    if output.type in ['HistorySet']:
      outputSize = -1
      for key in output.getParaKeys('outputs'):
        if key in instanciatedSelf.modelVariableType.keys():
          if outputSize == -1: outputSize = len(np.atleast_1d(outcomes[key]))
          if not sizeMatch(outcomes[key],outputSize): self.raiseAnError(Exception,"the time series size needs to be the same for the output space in a HistorySet!")
    Dummy.collectOutput(self, finishedJob, output, options)
#
#
#
class Code(Model):
  """
    This is the generic class that import an external code into the framework
  """
  CodeInterfaces = importlib.import_module("CodeInterfaces")
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
    #if alias are defined in the input it defines a mapping between the variable names in the framework and the one for the generation of the input
    #self.alias[framework variable name] = [input code name]. For Example, for a MooseBasedApp, the alias would be self.alias['internal_variable_name'] = 'Material|Fuel|thermal_conductivity'
    self.alias              = {}
    self.printTag           = 'CODE MODEL'
    self.lockedFileName     = "ravenLocked.raven"

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Model._readMoreXML(self, xmlNode)
    self.clargs={'text':'', 'input':{'noarg':[]}, 'pre':'', 'post':''} #output:''
    self.fargs={'input':{}, 'output':'', 'moosevpp':''}
    for child in xmlNode:
      if child.tag =='executable':
        self.executable = str(child.text)
      if child.tag =='preexec':
        self.preExec = str(child.text)
      elif child.tag =='alias':
        # the input would be <alias variable='internal_variable_name'>Material|Fuel|thermal_conductivity</alias>
        if 'variable' in child.attrib.keys(): self.alias[child.attrib['variable']] = child.text
        else: self.raiseAnError(IOError,'not found the attribute variable in the definition of one of the alias for code model '+str(self.name))
      elif child.tag == 'clargs':
        argtype = child.attrib['type']      if 'type'      in child.attrib.keys() else None
        arg     = child.attrib['arg']       if 'arg'       in child.attrib.keys() else None
        ext     = child.attrib['extension'] if 'extension' in child.attrib.keys() else None
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
      elif child.tag == 'fileargs':
        argtype = child.attrib['type']      if 'type'      in child.attrib.keys() else None
        arg     = child.attrib['arg']       if 'arg'       in child.attrib.keys() else None
        ext     = child.attrib['extension'] if 'extension' in child.attrib.keys() else None
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
    if self.executable == '': self.raiseAnError(IOError,'not found the node <executable> in the body of the code model '+str(self.name))
    if '~' in self.executable: self.executable = os.path.expanduser(self.executable)
    abspath = os.path.abspath(self.executable)
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
    self.code.readMoreXML(xmlNode)
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
    for key, value in self.alias.items():
      paramDict['The code variable '+str(value)+' it is filled using the framework variable '] = key
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
    try: os.mkdir(self.workingDir)
    except OSError:
      self.raiseAWarning('current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
      if utils.checkIfPathAreAccessedByAnotherProgram(self.workingDir,3.0): self.raiseAWarning('directory '+ self.workingDir + ' is likely used by another program!!! ')
      if utils.checkIfLockedRavenFileIsPresent(self.workingDir,self.lockedFileName): self.raiseAnError(RuntimeError, self, "another instance of RAVEN is running in the working directory "+ self.workingDir+". Please check your input!")
      # register function to remove the locked file at the end of execution
      atexit.register(lambda filenamelocked: os.remove(filenamelocked),os.path.join(self.workingDir,self.lockedFileName))
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
      this function have to return a new input that will be submitted to the model, it is called by the sampler
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet only the last set of entries are copied.
      The copied values are returned as a dictionary back
      @ In, currentInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
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
    if len(self.alias.keys()) != 0: Kwargs['alias']   = self.alias
    return (self.code.createNewInput(newInputSet,self.oriInputFiles,samplerType,**Kwargs),Kwargs)

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
    newKwargs['SampledVarsPb'].update(dict.fromkeys(externalDict.keys(),0.0))
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
    jobHandler.submitDict['External'](executeCommand,self.outFileRoot,metaData.pop('subDirectory'),identifier=identifier,metadata=metaData,codePointer=self.code,uniqueHandler = uniqueHandler)
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


#
#
#
#
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
    if len(Input) > 0 : jobHandler.submitDict['Internal']((Input,),self.interface.run,str(0),modulesToImport = self.mods, forceUseThreads = True)
    else: jobHandler.submitDict['Internal']((None,),self.interface.run,str(0),modulesToImport = self.mods, forceUseThreads = True)

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
      this function have to return a new input that will be submitted to the model, it is called by the sampler
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries is copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, createNewInput, tuple, return the new input in a tuple form
    """
    return self.interface.inputToInternal(self,myInput)
#
#
#
#
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
        modelName = child.text.strip()
        self.modelsDictionary[modelName] = {'TargetEvaluation':None,'Instance':None,'Input':[]}
        for childChild in child:
          try                  : self.modelsDictionary[modelName][childChild.tag].append(childChild.text.strip())
          except AttributeError: self.modelsDictionary[modelName][childChild.tag] = childChild.text.strip()
        if self.modelsDictionary[modelName].values().count(None) != 1: self.raiseAnError(IOError, "TargetEvaluation xml block needs to be inputted!")
        #if len(self.modelsDictionary[child.text.strip()].values()) > 2: self.raiseAnError(IOError, "TargetEvaluation xml block is the only XML sub-block allowed!")
        #if 'inputNames' not in child.attrib.keys(): self.raiseAnError(IOError, "inputNames attribute for Model" + child.text.strip() +" has not been inputted!")
        #self.modelsDictionary[modelName]['inputNames'] = [utils.toStrish(inpName) for inpName in child.attrib["inputNames"].split(",")]
        if len(self.modelsDictionary[modelName]['Input']) == 0 : self.raiseAnError(IOError, "Input XML node for Model" + modelName +" has not been inputted!")
        if len(self.modelsDictionary[modelName].values()) > 3  : self.raiseAnError(IOError, "TargetEvaluation and Input XML blocks are the only XML sub-blocks allowed!")
      if child.tag == 'settings':
        self.__readSettings(child)
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
            self.initialConditions[var.tag] = np.array(var.text.split())

  def __createDictOfInputsAndModels(self):
    allInputs = {}
    for modelIn, modelOptions in self.modelsDictionary.items():
     for inp in modelOptions['Input']:
       if inp not in allInputs.keys(): allInputs[inp] = []
       allInputs[inp].append(modelIn)
    return allInputs

  def __findModelWithSameInput(self,inputList):
    """
      Method to find the models that have certain inputs. If not found, return None
      @ In, inputList, list, list of inputs that need to be checked
      @ Out, models, list, list of model names that match the input list
    """
    models = []
    for modelIn, modelOptions in self.modelsDictionary.items():
      if set(modelOptions['Input']) == set(inputList): models.append(modelIn)
    return models

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


    def findPath(self, graph, start_vertex, end_vertex, path=[]):
      """
      find a path from start_vertex to end_vertex
            in graph
      """
      graph = self.__graph_dict
      path = path + [start_vertex]
      if start_vertex == end_vertex:
          return path
      if start_vertex not in graph:
          return None
      for vertex in graph[start_vertex]:
        if vertex not in path:
          extended_path = self.findPath(graph,vertex,end_vertex,path)
          if extended_path:return extended_path
      return None


  def findAllPaths(self, graph, start, end, path=[]):
    path = path + [start]
    if start == end: return [path]
    if not graph.has_key(start): return []
    paths = []
    for node in graph[start]:
      if node not in path:
        newpaths = self.findAllPaths(graph, node, end, path)
        for newpath in newpaths: paths.append(newpath)
    return paths

  def isALoop(self,g):
      """
        Return True if the directed graph g has a cycle.
        g must be represented as a dictionary mapping vertices to
        iterables of neighbouring vertices. For example:

        >>> isALoop({1: (2,), 2: (3,), 3: (1,)})
        True
        >>> isALoop({1: (2,), 2: (3,), 3: (4,)})
        False

      """
      path = set()

      def visit(vertex):
          path.add(vertex)
          for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour): return True
          path.remove(vertex)
          return False

      return any(visit(v) for v in g)

  def __getExecutionList(self, orderedNodes, allPath):
    """
    """
    numberPath = len(allPath)
    maxComponents = 0
    for path in allPath:
      if len(path) > maxComponents: maxComponents = len(path)
    executionList = [ [] for _ in range(maxComponents)]
    executionCounter = -1
    for node in orderedNodes:
      nodeCtn = 0
      for path in allPath:
        if node in path: nodeCtn +=1
      if nodeCtn == numberPath:
        executionCounter+=1
        executionList[executionCounter] = [node]
      else:
        previousNodesInPath = []
        for path in allPath:
          if path.count(node) > 0: previousNodesInPath.append(path[path.index(node)-1])
        for previousNode in previousNodesInPath:
          if previousNode in executionList[executionCounter]:
            executionCounter+=1
            break
        executionList[executionCounter].append(node)
    return executionList


  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize the EnsembleModel
      @ In, runInfo is the run info from the jobHandler
      @ In, inputs is a list containing whatever is passed with an input role in the step
      @ In, initDict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    moldelNodes = {}
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
      #if type(self.tempTargetEvaluations[modelIn[2]]).__name__ != 'PointSet': self.raiseAnError(IOError, "The TargetEvaluation needs to be an instance of PointSet. Got "+type(self.tempTargetEvaluations[modelIn[2]]).__name__)
      self.modelsDictionary[modelIn[2]]['Input' ] = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("inputs")
      self.modelsDictionary[modelIn[2]]['Output'] = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("outputs")
      modelNode = TreeStructure.Node(modelIn[2])
      modelNode.add( 'inputs', self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("inputs"))
      modelNode.add('outputs', self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("outputs"))
      moldelNodes[modelIn[2]] = modelNode
      rootNode.appendBranch(modelNode)

    # construct chain connections
    modelsToOutputModels  = dict.fromkeys(self.modelsDictionary.keys(),None)

    for modelIn in self.modelsDictionary.keys():
      outputMatch = []
      for i in range(len(self.modelsDictionary[modelIn]['Output'])):
        match = self.__findMatchingModel('Input',self.modelsDictionary[modelIn]['Output'][i])
        outputMatch.extend(match if match is not None else [])
      outputMatch = list(set(outputMatch))
      modelsToOutputModels[modelIn] = outputMatch
    orderList        = self.modelsDictionary.keys()
    for modelIn in self.modelsDictionary.keys():
      for i in range(len(self.modelsDictionary[modelIn]['Input'])):
        inputMatch   = self.__findMatchingModel('Output',self.modelsDictionary[modelIn]['Input'][i])
        if inputMatch is not None:
          for match in inputMatch:
            indexModelIn = orderList.index(modelIn)
            orderList.pop(indexModelIn)
            orderList.insert(int(max(orderList.index(match)+1,indexModelIn)), modelIn)
    allPath = self.findAllPaths(modelsToOutputModels,orderList[0],orderList[-1])
    ###################################################
    # to be removed once executionList can be handled #
    self.orderList = orderList                        #
    ###################################################
    if len(allPath) > 1: self.executionList = self.__getExecutionList(orderList,allPath)
    else               : self.executionList = allPath[-1]
    # check if Picard needs to be activated
    self.activatePicard = self.isALoop(modelsToOutputModels)
    if self.activatePicard:
      self.raiseAMessage("Multi-model connections determined a non-linear system. Picard's iterations activated!")
      if len(self.initialConditions.keys()) == 0: self.raiseAnError(IOError,"Picard's iterations mode activated but no intial conditions provided!")
    else                  : self.raiseAMessage("Multi-model connections determined a linear system. Picard's iterations not activated!")

    self.allOutputs = []
    self.needToCheckInputs = True

  def localAddInitParams(self,tempDict):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, tempDict, dict, dictionary to be updated. {'attribute name':value}
      @ Out, None
    """
    tempDict['Models contained in EnsembleModel are '] = self.modelsDictionary.keys()

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
      this function have to return a new input that will be submitted to the model, it is called by the sampler
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **Kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    # check if all the inputs of the submodule are covered by the sampled vars and Outputs of the other sub-models
    if self.needToCheckInputs: allCoveredVariables = list(set(self.allOutputs + Kwargs['SampledVars'].keys()))
    newInputs                     = {}
    identifier                    = Kwargs['prefix']
    newInputs['prefix']           = identifier
    for modelIn, specs in self.modelsDictionary.items():
      if self.needToCheckInputs:
        for inp in specs['Input']:
          if inp not in allCoveredVariables: self.raiseAnError(RuntimeError,"for sub-model "+ modelIn + "the input "+inp+" has not been found among other models' outputs and sampled variables!")
      newKwargs = self.__selectInputSubset(modelIn,Kwargs)
      #inputForModel = []
      #for input in myInput:
      #  if input.name in self.modelsDictionary[modelIn]['inputNames']: inputForModel.append(input)
      #if len(inputForModel) == 0: self.raiseAnError(IOError,"inputs " + " ".join(self.modelsDictionary[modelIn]['inputNames']) + " has not been found!")
      inputDict = [self._inputToInternal(self.modelsDictionary[modelIn]['InputObject'][0],newKwargs['SampledVars'].keys())] if specs['Instance'].type != 'Code' else  self.modelsDictionary[modelIn]['InputObject']
      newInputs[modelIn] = specs['Instance'].createNewInput(inputDict,samplerType,**newKwargs)
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
      inputsValues   = targetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
      outputsValues  = targetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
      metadataValues = targetEvaluations[modelIn].getAllMetadata(nodeId = 'RecontructEnding')
      inputsValues   = inputsValues if targetEvaluations[modelIn].type != 'HistorySet' else inputsValues.values()[-1]
      outputsValues  = outputsValues if targetEvaluations[modelIn].type != 'HistorySet' else outputsValues.values()[-1]

      for key in targetEvaluations[modelIn].getParaKeys('inputs'):
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateInputValue (key,inputsValues[key],options={'acceptArrayRealizations':True})
      for key in targetEvaluations[modelIn].getParaKeys('outputs'):
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateOutputValue (key,outputsValues[key],options={'acceptArrayRealizations':True})
      for key in metadataValues.keys():
        self.modelsDictionary[modelIn]['TargetEvaluation'].updateMetadata(key,metadataValues[key])
      # end of update of TargetEvaluation
      for typeInfo,values in outcomes[modelIn].items():
        for key in values.keys(): exportDict[typeInfo][key] = np.asarray(values[key])
      if output.name == self.modelsDictionary[modelIn]['TargetEvaluation'].name: self.raiseAnError(RuntimeError, "The Step output can not be one of the target evaluation outputs!")
    if output.type == 'HDF5': output.addGroupDataObjects({'group':self.name+str(finishedJob.identifier)},exportDict,False)
    else:
      for key in exportDict['inputSpaceParams' ] :
        if key in output.getParaKeys('inputs') : output.updateInputValue (key,exportDict['inputSpaceParams' ][key],options={'acceptArrayRealizations':True})
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
    jobHandler.submitDict['InternalClient'](((copy.deepcopy(Input),jobHandler),), self.__externalRun,str(Input['prefix']))

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
          if input in previousOutputs.keys(): dependentOutputs[input] =  previousOutputs[input][-1] if outputType != 'HistorySet' else np.asarray(previousOutputs[input])
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
    #with self.lockSystem:
    for modelIn in self.orderList:
      self.tempTargetEvaluations[modelIn].resetData()
    tempTargetEvaluations = copy.deepcopy(self.tempTargetEvaluations)
    #modelsTargetEvaluations[modelIn] = copy.deepcopy(self.modelsDictionary[modelIn]['TargetEvaluation'])
    residueContainer = dict.fromkeys(self.modelsDictionary.keys())
    gotOutputs  = [{}]*len(self.orderList)
    typeOutputs = ['']*len(self.orderList)
    if self.activatePicard:
      for modelIn in self.orderList:
        residueContainer[modelIn] = {'residue':{},'iterValues':[{}]*2}
        for out in self.modelsDictionary[modelIn]['Output']:
          residueContainer[modelIn]['residue'][out], residueContainer[modelIn]['iterValues'][0][out], residueContainer[modelIn]['iterValues'][1][out] = np.zeros(1), np.zeros(1), np.zeros(1)
    maxIterations, iterationCount = (self.maxIterations, 0) if self.activatePicard else (1 , 0)
    #if self.activatePicard: maxIterations, iterationCount = self.maxIterations, 0 if self.activatePicard else 1 , 0
    #else                  : maxIterations, iterationCount = 1 , 0
    while iterationCount < maxIterations:
      returnDict     = {}
      iterationCount += 1
      if self.activatePicard: self.raiseAMessage("Picard's Iteration "+ str(iterationCount))
      for modelCnt, modelIn in enumerate(self.orderList):
        #with self.lockSystem:
        dependentOutput = self.__retrieveDependentOutput(modelIn, gotOutputs, typeOutputs)
        if iterationCount == 1  and self.activatePicard:
          try              : sampledVars = Input[modelIn][0][1]['SampledVars'].keys()
          except IndexError: sampledVars = Input[modelIn][1]['SampledVars'].keys()
          for initCondToSet in [x for x in self.modelsDictionary[modelIn]['Input'] if x not in set(dependentOutput.keys()+sampledVars)]:
            if initCondToSet in self.initialConditions.keys(): dependentOutput[initCondToSet] = np.asarray(self.initialConditions[initCondToSet])
            else                                             : self.raiseAnError(IOError,"No initial conditions provided for variable "+ initCondToSet)
        #with self.lockSystem:
        Input[modelIn]  = self.modelsDictionary[modelIn]['Instance'].updateInputFromOutside(Input[modelIn], dependentOutput)
        try              : Input[modelIn][0][1]['prefix'], Input[modelIn][0][1]['uniqueHandler'] = modelIn+"|"+identifier, self.name+identifier
        except IndexError: Input[modelIn][1]['prefix'   ], Input[modelIn][1]['uniqueHandler'   ] = modelIn+"|"+identifier, self.name+identifier
        nextModel = False
        while not nextModel:
          moveOn = False
          while not moveOn:
            if jobHandler.howManyFreeSpots() > 0:
              #with self.lockSystem:
              self.modelsDictionary[modelIn]['Instance'].run(copy.deepcopy(Input[modelIn]),jobHandler)
              while not jobHandler.isThisJobFinished(modelIn+"|"+identifier): time.sleep(1.e-3)
              nextModel, moveOn = True, True
            else: time.sleep(1.e-3)
          # get job that just finished
          #with self.lockSystem:
          finishedRun = jobHandler.getFinished(jobIdentifier = modelIn+"|"+identifier, uniqueHandler=self.name+identifier)
          if finishedRun[0].getEvaluation() == -1:
            for modelToRemove in self.orderList:
              if modelToRemove != modelIn: jobHandler.getFinished(jobIdentifier = modelToRemove + "|" + identifier, uniqueHandler = self.name + identifier)
            self.raiseAnError(RuntimeError,"The Model "+modelIn + " failed!")
          # get back the output in a general format
          self.modelsDictionary[modelIn]['Instance'].collectOutput(finishedRun[0],tempTargetEvaluations[modelIn],options={'acceptArrayRealizations':True})
          returnDict[modelIn]  = {}
          responseSpace = tempTargetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
          inputSpace    = tempTargetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
          typeOutputs[modelCnt] = tempTargetEvaluations[modelIn].type
          gotOutputs[modelCnt]  = responseSpace if typeOutputs[modelCnt] != 'HistorySet' else responseSpace.values()[-1]
          #store the result in return dictionary
          returnDict[modelIn]['outputSpaceParams'] = gotOutputs[modelCnt]
          returnDict[modelIn]['inputSpaceParams' ] = inputSpace if typeOutputs[modelCnt] != 'HistorySet' else inputSpace.values()[-1]
          returnDict[modelIn]['metadata'         ] = tempTargetEvaluations[modelIn].getAllMetadata()
          #returnDict[modelIn] = {'outputSpaceParams':gotOutputs[modelCnt],'inputSpaceParams':tempTargetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding'),'metadata':tempTargetEvaluations[modelIn].getAllMetadata()}
          if self.activatePicard:
            # compute residue
            residueContainer[modelIn]['iterValues'][1] = copy.copy(residueContainer[modelIn]['iterValues'][0])
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['iterValues'][0][out] = copy.copy(gotOutputs[modelCnt][out])
              if iterationCount == 1: residueContainer[modelIn]['iterValues'][1][out] = np.zeros(len(residueContainer[modelIn]['iterValues'][0][out]))
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['residue'][out] = abs(np.asarray(residueContainer[modelIn]['iterValues'][0][out]) - np.asarray(residueContainer[modelIn]['iterValues'][1][out]))
            residueContainer[modelIn]['Norm'] =  np.linalg.norm(np.asarray(residueContainer[modelIn]['iterValues'][1].values())-np.asarray(residueContainer[modelIn]['iterValues'][0].values()))
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
#
#
#
#
"""
 Factory......
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
