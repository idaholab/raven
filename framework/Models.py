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
import sys
import atexit
# to be removed
from scipy import spatial
# to be removed
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from Assembler import Assembler
import SupervisedLearning
import PostProcessors #import returnFilterInterface
import CustomCommandExecuter
import utils
import TreeStructure
from FileObjects import FileObject
#Internal Modules End--------------------------------------------------------------------------------

#class Model(BaseType):
class Model(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
  A model is something that given an input will return an output reproducing some physical model
  it could as complex as a stand alone code, a reduced order model trained somehow or something
  externally build and imported by the user
  """
  validateDict                  = {}
  validateDict['Input'  ]       = []
  validateDict['Output' ]       = []
  validateDict['Sampler']       = []
  testDict                      = {}
  testDict                      = {'class':'','type':[''],'multiplicity':0,'required':False}
  #FIXME: a multiplicity value is needed to control role that can have different class
  #the possible inputs
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][0]['class'       ] = 'DataObjects'
  validateDict['Input'  ][0]['type'        ] = ['Point','PointSet','History','HistorySet']
  validateDict['Input'  ][0]['required'    ] = False
  validateDict['Input'  ][0]['multiplicity'] = 'n'
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][1]['class'       ] = 'Files'
  validateDict['Input'  ][1]['type'        ] = ['']
  validateDict['Input'  ][1]['required'    ] = False
  validateDict['Input'  ][1]['multiplicity'] = 'n'
  #the possible outputs
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][0]['class'       ] = 'DataObjects'
  validateDict['Output' ][0]['type'        ] = ['Point','PointSet','History','HistorySet']
  validateDict['Output' ][0]['required'    ] = False
  validateDict['Output' ][0]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][1]['class'       ] = 'Databases'
  validateDict['Output' ][1]['type'        ] = ['HDF5']
  validateDict['Output' ][1]['required'    ] = False
  validateDict['Output' ][1]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][2]['class'       ] = 'OutStreamManager'
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
                                                'Sobol']

  @classmethod
  def generateValidateDict(cls):
    """This method generate a independent copy of validateDict for the calling class"""
    cls.validateDict = copy.deepcopy(Model.validateDict)

  @classmethod
  def specializeValidateDict(cls):
    """ This method should be overridden to describe the types of input accepted with a certain role by the model class specialization"""
    raise NotImplementedError('The class '+str(cls.__name__)+' has not implemented the method specializeValidateDict')

  @classmethod
  def localValidateMethod(cls,who,what):
    """
    This class method is called to test the compatibility of the class with its possible usage
    @in who: a string identifying the what is the role of what we are going to test (i.e. input, output etc)
    @in what: a list (or a general iterable) that will be playing the 'who' role
    """
    #counting successful matches
    if who not in cls.validateDict.keys(): raise IOError('The role '+str(who)+' does not exist in the class '+str(cls))
    for myItemDict in cls.validateDict[who]: myItemDict['tempCounter'] = 0
    for anItem in what:
      anItem['found'] = False
      for tester in cls.validateDict[who]:
        if anItem['class'] == tester['class']:
          if anItem['type'] in tester['type']:
            tester['tempCounter'] +=1
            anItem['found']        = True
            break
    #testing if the multiplicity of the argument is correct
    for tester in cls.validateDict[who]:
      if tester['required']==True:
        if tester['multiplicity']=='n' and tester['tempCounter']<1:
          raise IOError('The number of time class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper')
        if tester['multiplicity']!='n' and tester['tempCounter']!=tester['multiplicity']:
          raise IOError('The number of time class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper')
    #testing if all argument to be tested have been found
    for anItem in what:
      if anItem['found']==False:
        raise IOError('It is not possible to use '+anItem['class']+' type= ' +anItem['type']+' as '+who)
    return True

  def __init__(self):
    BaseType.__init__(self)
    self.subType  = ''
    self.runQueue = []
    self.printTag = 'MODEL'
    self.mods     = utils.returnImportModuleString(inspect.getmodule(self),True)
    self.globs    = {}

  def _readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['subType']
    except KeyError:
      self.raiseADebug(" Failed in Node: "+str(xmlNode),verbostiy='silent')
      self.raiseAnError(IOError,'missed subType for the model '+self.name)
    del(xmlNode.attrib['subType'])

  def localInputAndChecks(self,xmlNode):
    """place here the additional reading, remember to add initial parameters in the method localAddInitParams"""

  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType

  def localAddInitParams(self,tempDict):
    """use this function to export to the printer in the base class the additional PERMANENT your local class have"""

  def initialize(self,runInfo,inputs,initDict=None):
    """ this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
    after this call the next one will be run
    @ In, runInfo is the run info from the jobHandler
    @ In, inputs is a list containing whatever is passed with an input role in the step
    @ In, initDict, optional, dictionary of all objects available in the step is using this model
    """
    pass

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
    this function have to return a new input that will be submitted to the model, it is called by the sampler
    @in myInput the inputs (list) to start from to generate the new one
    @in samplerType is the type of sampler that is calling to generate a new input
    @in **Kwargs is a dictionary that contains the information coming from the sampler,
         a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
    @return the new input in a list form
    """
    return [(copy.copy(Kwargs))]

  @abc.abstractmethod
  def run(self,Input,jobHandler):
    """
    This call should be over loaded and should not return any results,
    possible it places a run one of the jobhadler lists!!!
    @in inputs is a list containing whatever is passed with an input role in the step
    @in jobHandler an instance of jobhandler that might be possible used to append a job for parallel running
    """
    pass

  def collectOutput(self,collectFrom,storeTo):
    """
    This call collect the output of the run
    @in collectFrom: where the output is located, the form and the type is model dependent but should be compatible with the storeTo.addOutput method.
    """
    #if a addOutput is present in nameSpace of storeTo it is used
    if 'addOutput' in dir(storeTo): storeTo.addOutput(collectFrom)
    else                          : self.raiseAnError(IOError,'The place where to store the output has not a addOutput method')

  def getAdditionalInputEdits(self,inputInfo):
    """
    Collects additional edits for the sampler to use when creating a new input.  By default does nothing.
    @ In, inputInfo, dictionary in which to add edits
    @ Out, None.
    """
    pass
#
#
#
class Dummy(Model):
  """
  this is a dummy model that just return the effect of the sampler. The values reported as input in the output
  are the output of the sampler and the output is the counter of the performed sampling
  """
  def __init__(self):
    Model.__init__(self)
    self.admittedData = self.__class__.validateDict['Input' ][0]['type'] #the list of admitted data is saved also here for run time checks
    #the following variable are reset at each call of the initialize method
    self.printTag = 'DUMMY MODEL'

  @classmethod
  def specializeValidateDict(cls):
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['type'        ] = ['Point','PointSet']
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['Point','PointSet']

  def _manipulateInput(self,dataIn):
    if len(dataIn)>1: self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name '+self.name)
    if type(dataIn[0])!=tuple: inRun = self._inputToInternal(dataIn[0]) #this might happen when a single run is used and the input it does not come from self.createNewInput
    else:                      inRun = dataIn[0][0]
    return inRun

  def _inputToInternal(self,dataIN,full=False):
    """Transform it in the internal format the provided input. dataIN could be either a dictionary (then nothing to do) or one of the admitted data"""
    self.raiseADebug('wondering if a dictionary compatibility should be kept','FIXME')
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
    here only Point and PointSet are accepted a local copy of the values is performed.
    For a Point all value are copied, for a PointSet only the last set of entry
    The copied values are returned as a dictionary back
    """
    if len(myInput)>1: self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name'+self.name)
    inputDict = self._inputToInternal(myInput[0])
    #test if all sampled variables are in the inputs category of the data
    if set(list(Kwargs['SampledVars'].keys())+list(inputDict.keys())) != set(list(inputDict.keys())):
      self.raiseAnError(IOError,'When trying to sample the input for the model '+self.name+' of type '+self.type+' the sampled variable are '+str(Kwargs['SampledVars'].keys())+' while the variable in the input are'+str(inputDict.keys()))
    for key in Kwargs['SampledVars'].keys(): inputDict[key] = np.atleast_1d(Kwargs['SampledVars'][key])
    if None in inputDict.values(): self.raiseAnError(IOError,'While preparing the input for the model '+self.type+' with name '+self.name+' found an None input variable '+ str(inputDict.items()))
    #the inputs/outputs should not be store locally since they might be used as a part of a list of input for the parallel runs
    #same reason why it should not be used the value of the counter inside the class but the one returned from outside as a part of the input
    return [(inputDict)],copy.copy(Kwargs)

  def run(self,Input,jobHandler):
    """
    The input is a list of one element.
    The element is either a tuple of two dictionary [(InputDictionary, OutputDictionary)] if the input has been created by the  self.createNewInput
    otherwise is one of the accepted data.
    The first is the input the second the output. The output is just the counter
    """
    #this set of test is performed to avoid that if used in a single run we come in with the wrong input structure since the self.createNewInput is not called
    inRun = self._manipulateInput(Input[0])
    def lambdaReturnOut(inRun,prefix): return {'OutputPlaceHolder':np.atleast_1d(np.float(prefix))}
    #lambdaReturnOut = lambda inRun: {'OutputPlaceHolder':np.atleast_1d(np.float(Input[1]['prefix']))}
    jobHandler.submitDict['Internal']((inRun,Input[1]['prefix']),lambdaReturnOut,str(Input[1]['prefix']),metadata=Input[1], modulesToImport = self.mods, globs = self.globs)

  def collectOutput(self,finishedJob,output):
    if finishedJob.returnEvaluation() == -1: self.raiseAnError(AttributeError,"No available Output to collect (Run probabably is not finished yet)")
    evaluation = finishedJob.returnEvaluation()
    if type(evaluation[1]).__name__ == "tuple": outputeval = evaluation[1][0]
    else                                      : outputeval = evaluation[1]
    exportDict = {'input_space_params':evaluation[0],'output_space_params':outputeval,'metadata':finishedJob.returnMetadata()}
    if output.type == 'HDF5': output.addGroupDataObjects({'group':self.name+str(finishedJob.identifier)},exportDict,False)
    else:
      for key in exportDict['input_space_params' ] :
        if key in output.getParaKeys('inputs'): output.updateInputValue (key,exportDict['input_space_params' ][key])
      for key in exportDict['output_space_params'] :
        if key in output.getParaKeys('outputs'): output.updateOutputValue(key,exportDict['output_space_params'][key])
      for key in exportDict['metadata'] : output.updateMetadata(key,exportDict['metadata'][key])
#
#
#
class ROM(Dummy):
  """ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome"""
  @classmethod
  def specializeValidateDict(cls):
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['Point','PointSet']

  def __init__(self):
    Dummy.__init__(self)
    self.initializationOptionDict = {}          # ROM initialization options
    self.amITrained                = False      # boolean flag, is the ROM trained?
    self.howManyTargets            = 0          # how many targets?
    self.SupervisedEngine          = {}         # dict of ROM instances (== number of targets => keys are the targets)
    self.printTag = 'ROM MODEL'

  def __getstate__(self):
    """
    Overwrite state (for pickle-ing)
    we do not pickle the HDF5 (C++) instance
    but only the info to re-load it
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    if not self.amITrained:
      a = state.pop("SupervisedEngine")
      del a
    return state

  def __setstate__(self, newstate):
    self.__dict__.update(newstate)
    if not self.amITrained:
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
    Dummy._readMoreXML(self, xmlNode)
    for child in xmlNode:
      if child.attrib:
        if child.tag not in self.initializationOptionDict.keys():
          self.initializationOptionDict[child.tag]={}
        self.initializationOptionDict[child.tag][child.text]=child.attrib
      else:
        try: self.initializationOptionDict[child.tag] = int(child.text)
        except ValueError:
          try: self.initializationOptionDict[child.tag] = float(child.text)
          except ValueError: self.initializationOptionDict[child.tag] = child.text
    #the ROM is instanced and initialized
    # check how many targets
    if not 'Target' in self.initializationOptionDict.keys(): self.raiseAnError(IOError,'No Targets specified!!!')
    targets = self.initializationOptionDict['Target'].split(',')
    self.howManyTargets = len(targets)
    for target in targets:
      self.initializationOptionDict['Target'] = target
      self.SupervisedEngine[target] =  SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptionDict)
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(self.SupervisedEngine.values()[0])))
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(SupervisedLearning)))
    #restore targets to initialization option dict
    self.initializationOptionDict['Target'] = ','.join(targets)

  def printXML(self,options=None):
    '''
      Called by the OutStreamPrint object to cause the ROM to print itself to file.
      @ In, options, the options to use in printing, including filename, things to print, etc.
    '''
    if options:
      if ('filenameroot' in options.keys()): filenameLocal = options['filenameroot']
      else: filenameLocal = self.name + '_dump'
    else: options={}
    tree=self._localBuildPrintTree(options)
    msg=tree.stringNodeTree()
    file(filenameLocal+'.xml','w').writelines(msg)
    self.raiseAMessage('ROM XML printed to "'+filenameLocal+'"')

  def _localBuildPrintTree(self,options=None):
    node = TreeStructure.Node('ReducedOrderModel')
    tree = TreeStructure.NodeTree(node)
    #tree._setrootnode(node)
    if 'target' in options.keys():
      targets = options['target'].split(',')
    else:
      targets = 'all'
    if 'all' in targets:
      targets = list(key for key in self.SupervisedEngine.keys())
    for key,target in self.SupervisedEngine.items():
      if key in targets:
        target.printXML(node,options)
    return tree

  def reset(self):
    """
    Reset the ROM
    @ In,  None
    @ Out, None
    """
    for instrom in self.SupervisedEngine.values(): instrom.reset()
    self.amITrained   = False

  def addInitParams(self,originalDict):
    """the ROM setting parameters are added"""
    ROMdict = {}
    for target, instrom in self.SupervisedEngine.items(): ROMdict[self.name + '|' + target] = instrom.returnInitialParameters()
    for key in ROMdict.keys(): originalDict[key] = ROMdict[key]

  def train(self,trainingSet):
    """Here we do the training of the ROM"""
    """Fit the model according to the given training data.
    @in X : {array-like, sparse matrix}, shape = [n_samples, n_features] Training vector, where n_samples in the number of samples and n_features is the number of features.
    @in y : array-like, shape = [n_samples] Target vector relative to X class_weight : {dict, 'auto'}, optional Weights associated with classes. If not given, all classes
            are supposed to have weight one."""
    if type(trainingSet).__name__ == 'ROM':
      self.howManyTargets           = copy.deepcopy(trainingSet.howManyTargets)
      self.initializationOptionDict = copy.deepcopy(trainingSet.initializationOptionDict)
      self.trainingSet              = copy.copy(trainingSet.trainingSet)
      self.amITrained               = copy.deepcopy(trainingSet.amITrained)
      self.SupervisedEngine         = copy.deepcopy(trainingSet.SupervisedEngine)
    else:
      self.trainingSet = copy.copy(self._inputToInternal(trainingSet,full=True))
      self.amITrained = True
      for instrom in self.SupervisedEngine.values():
        instrom.train(self.trainingSet)
        self.aimITrained = self.amITrained and instrom.amITrained
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
    if target != None: return self.SupervisedEngine[target].confidence(inputToROM)
    else             : return self.SupervisedEngine.values()[0].confidence(inputToROM)

  def evaluate(self,request, target = None):
    """
    when the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used
    @ In, request, datatype, feature coordinates (request)
    @ In, target, string, optional, target name (by default the first target entered in the input file)
    """
    inputToROM = self._inputToInternal(request)
    if target != None: return self.SupervisedEngine[target].evaluate(inputToROM)
    else             : return self.SupervisedEngine.values()[0].evaluate(inputToROM)

  def __externalRun(self,inRun):
    returnDict = {}
    for target in self.SupervisedEngine.keys(): returnDict[target] = self.evaluate(inRun,target)
    return returnDict

  def run(self,Input,jobHandler):
    """This call run a ROM as a model"""
    inRun = self._manipulateInput(Input[0])
    jobHandler.submitDict['Internal']((inRun,),self.__externalRun,str(Input[1]['prefix']),metadata=Input[1],modulesToImport=self.mods, globs = self.globs)
#
#
#
class ExternalModel(Dummy):
  """ External model class: this model allows to interface with an external python module"""
  @classmethod
  def specializeValidateDict(cls):
    #one data is needed for the input
    #cls.raiseADebug('think about how to import the roles to allowed class for the external model. For the moment we have just all')
    pass

  def __init__(self):
    """
    Constructor
    @ In, None
    @ Out, None
    """
    Dummy.__init__(self)
    self.sim                      = None
    self.modelVariableValues      = {}                                                                                                       # dictionary of variable values for the external module imported at runtime
    self.modelVariableType        = {}                                                                                                       # dictionary of variable types, used for consistency checks
    self._availableVariableTypes = ['float','bool','int','ndarray','float16','float32','float64','float128','int16','int32','int64','bool8'] # available data types
    self._availableVariableTypes = self._availableVariableTypes + ['numpy.'+item for item in self._availableVariableTypes]                   # as above
    self.printTag                 = 'EXTERNAL MODEL'
    self.initExtSelf              = utils.Object()

  def initialize(self,runInfo,inputs,initDict=None):
    """
    Initialize method for the model
    @ In, runInfo is the run info from the jobHandler
    @ In, inputs is a list containing whatever is passed with an input role in the step
    @ In, initDict, optional, dictionary of all objects available in the step is using this model
    """
    for key in self.modelVariableType.keys(): self.modelVariableType[key] = None
    if 'initialize' in dir(self.sim): self.sim.initialize(self.initExtSelf,runInfo,inputs)
    Dummy.initialize(self, runInfo, inputs)
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(self.sim)))

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """
    Function to create a new input, through the info contained in Kwargs
    @ In, myInput, list of original inputs
    @ In, samplerType, string, sampler type (e.g. MonteCarlo, DET, etc.)
    @ In, Kwargs, dictionary containing information useful for creation of a newer input (e.g. sampled variables, etc.)
    """
    modelVariableValues ={}
    for key in Kwargs['SampledVars'].keys(): modelVariableValues[key] = Kwargs['SampledVars'][key]
    if 'createNewInput' in dir(self.sim):
      extCreateNewInput = self.sim.createNewInput(self,myInput,samplerType,**Kwargs)
      if extCreateNewInput== None: self.raiseAnError(AttributeError,'in external Model '+self.ModuleToLoad+' the method createNewInput must return something. Got: None')
      return ([(extCreateNewInput)],copy.copy(Kwargs)),copy.copy(modelVariableValues)
    else: return Dummy.createNewInput(self, myInput,samplerType,**Kwargs),copy.copy(modelVariableValues)

  def _readMoreXML(self,xmlNode):
    """
    Function to read the peace of input belongs to this model
    @ In, xmlTree object, xml node containg the peace of input that belongs to this model
    """
    Model._readMoreXML(self, xmlNode)
    if 'ModuleToLoad' in xmlNode.attrib.keys():
      self.ModuleToLoad = os.path.split(str(xmlNode.attrib['ModuleToLoad']))[1]
      if (os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0] != ''):
        abspath = os.path.abspath(os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0])
        if '~' in abspath:abspath = os.path.expanduser(abspath)
        if os.path.exists(abspath): os.sys.path.append(abspath)
        else: self.raiseAnError(IOError,'The path provided for the external model does not exist!!! Got: ' + abspath)
    else: self.raiseAnError(IOError,'ModuleToLoad not provided for module externalModule')
    # load the external module and point it to self.sim
    self.sim = utils.importFromPath(str(xmlNode.attrib['ModuleToLoad']),self.messageHandler.getDesiredVerbosity(self)>1)
    # check if there are variables and, in case, load them
    for son in xmlNode:
      if son.tag=='variable':
        if len(son.attrib.keys()) > 0: self.raiseAnError(IOError,'the block '+son.tag+' named '+son.text+' should not have attributes!!!!!')
        self.modelVariableType[son.text] = None
    # check if there are other information that the external module wants to load
    if '_readMoreXML' in dir(self.sim): self.sim._readMoreXML(self,xmlNode)

  def __externalRun(self, Input, modelVariables):
    """
    Method that performs the actual run of the imported external model (separated from run method for parallelization purposes)
    @ In, Input, list, list of the inputs needed for running the model
    """
    externalSelf        = utils.Object()
    #self.sim=__import__(self.ModuleToLoad)
    modelVariableValues = {}
    for key in self.modelVariableType.keys(): modelVariableValues[key] = None
    for key,value in self.initExtSelf.__dict__.items():
      CustomCommandExecuter.execCommand('self.'+ key +' = copy.copy(object)',self=externalSelf,object=value)  # exec('externalSelf.'+ key +' = copy.copy(value)')
      modelVariableValues[key] = copy.copy(value)
    for key in Input.keys(): modelVariableValues[key] = copy.copy(Input[key])
    if 'createNewInput' not in dir(self.sim):
      for key in Input.keys(): modelVariableValues[key] = copy.copy(Input[key])
      for key in self.modelVariableType.keys() : CustomCommandExecuter.execCommand('self.'+ key +' = copy.copy(object["'+key+'"])',self=externalSelf,object=modelVariableValues) #exec('externalSelf.'+ key +' = copy.copy(modelVariableValues[key])')  #self.__uploadSolution()
    self.sim.run(externalSelf, Input)
    for key in self.modelVariableType.keys()   : CustomCommandExecuter.execCommand('object["'+key+'"]  = copy.copy(self.'+key+')',self=externalSelf,object=modelVariableValues) #exec('modelVariableValues[key]  = copy.copy(externalSelf.'+key+')') #self.__pointSolution()
    for key in self.initExtSelf.__dict__.keys(): CustomCommandExecuter.execCommand('self.' +key+' = copy.copy(object.'+key+')',self=self.initExtSelf,object=externalSelf) #exec('self.initExtSelf.' +key+' = copy.copy(externalSelf.'+key+')')
    if None in self.modelVariableType.values():
      errorfound = False
      for key in self.modelVariableType.keys():
        self.modelVariableType[key] = type(modelVariableValues[key]).__name__
        if self.modelVariableType[key] not in self._availableVariableTypes:
          if not errorfound: self.raiseADebug('Unsupported type found. Available ones are: '+ str(self._availableVariableTypes).replace('[','').replace(']', ''),verbosity='silent')
          errorfound = True
          self.raiseADebug('variable '+ key+' has an unsupported type -> '+ self.modelVariableType[key],verbosity='silent')
      if errorfound: self.raiseAnError(RuntimeError,'Errors detected. See above!!')
    return copy.copy(modelVariableValues),self

  def run(self,Input,jobHandler):
    """
    Method that performs the actual run of the imported external model
    @ In, Input, list, list of the inputs needed for running the model
    @ In, jobHandler, jobHandler object, jobhandler instance
    """
    inRun = copy.copy(self._manipulateInput(Input[0][0]))
    jobHandler.submitDict['Internal']((inRun,Input[1],),self.__externalRun,str(Input[0][1]['prefix']),metadata=Input[0][1], modulesToImport = self.mods, globs = self.globs)

  def collectOutput(self,finishedJob,output):
    """
    Method that collects the outputs from the previous run
    @ In, finishedJob, InternalRunner object, instance of the run just finished
    @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
    """
    if finishedJob.returnEvaluation() == -1: self.raiseAnError(RuntimeError,"No available Output to collect (Run probabably is not finished yet)")
    def typeMatch(var,var_type_str):
      type_var = type(var)
      return type_var.__name__ == var_type_str or \
        type_var.__module__+"."+type_var.__name__ == var_type_str
    # check type consistency... This is needed in order to keep under control the external model... In order to avoid problems in collecting the outputs in our internal structures
    instanciatedSelf = finishedJob.returnEvaluation()[1][1]
    outcomes         = finishedJob.returnEvaluation()[1][0]
    for key in instanciatedSelf.modelVariableType.keys():
      if not (typeMatch(outcomes[key],instanciatedSelf.modelVariableType[key])):
        self.raiseAnError(RuntimeError,'type of variable '+ key + ' is ' + str(type(outcomes[key]))+' and mismatches with respect to the input ones (' + instanciatedSelf.modelVariableType[key] +')!!!')
    Dummy.collectOutput(self, finishedJob, output)
#
#
#
class Code(Model):
  """this is the generic class that import an external code into the framework"""
  CodeInterfaces = importlib.import_module("CodeInterfaces")
  @classmethod
  def specializeValidateDict(cls):
    #FIXME think about how to import the roles to allowed class for the codes. For the moment they are not specialized by executable
    cls.validateDict['Input'] = [cls.validateDict['Input'][1]]

  def __init__(self):
    Model.__init__(self)
    self.executable         = ''   #name of the executable (abs path)
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
    """extension of info to be read for the Code(model) as well as the code interface, and creates the interface.
    @ In: xmlNode, node object
    @ Out: None.
    """
    Model._readMoreXML(self, xmlNode)
    self.clargs={'text':'', 'input':{'noarg':[]}, 'pre':'', 'post':''} #output:''
    self.fargs={'input':{}, 'output':''}
    for child in xmlNode:
      if child.tag =='executable':
        self.executable = str(child.text)
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
        else: self.raiseAnError(IOError,'filearg type '+argtype+' not recognized!')
    if self.executable == '': self.raiseAnError(IOError,'not found the node <executable> in the body of the code model '+str(self.name))
    if '~' in self.executable: self.executable = os.path.expanduser(self.executable)
    abspath = os.path.abspath(self.executable)
    if os.path.exists(abspath):
      self.executable = abspath
    else: self.raiseAMessage('not found executable '+self.executable,'ExceptedError')
    self.code = Code.CodeInterfaces.returnCodeInterface(self.subType,self,self.messageHandler)
    self.code.readMoreXML(xmlNode)
    self.code.setInputExtension(list(a for b in (c for c in self.clargs['input'].values()) for a in b))
    self.code.addInputExtension(list(a for b in (c for c in self.fargs ['input'].values()) for a in b))
    self.code.addDefaultExtension()

  def addInitParams(self,tempDict):
    """extension of addInitParams for the Code(model)"""
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
    for key, value in self.alias.items():
      tempDict['The code variable '+str(value)+' it is filled using the framework variable '] = key

  def addCurrentSetting(self,originalDict):
    """extension of addInitParams for the Code(model)"""
    originalDict['current working directory'] = self.workingDir
    originalDict['current output file root' ] = self.outFileRoot
    originalDict['current input file'       ] = self.currentInputFiles
    originalDict['original input file'      ] = self.oriInputFiles

  def getAdditionalInputEdits(self,inputInfo):
    """
    Adds input edits besides the sampledVars to the inputInfo dictionary. Called by the sampler.
    @ In, inputInfo, dictionary object
    @Out, None.
    """
    inputInfo['additionalEdits']=self.fargs

  def initialize(self,runInfoDict,inputFiles,initDict=None):
    """initialize some of the current setting for the runs and generate the working
       directory with the starting input files"""
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except OSError:
      self.raiseAWarning('current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
      if utils.checkIfPathAreAccessedByAnotherProgram(self.workingDir,3.0): self.raiseAWarning('directory '+ self.workingDir + ' is likely used by another program!!! ')
      if utils.checkIfLockedRavenFileIsPresent(self.workingDir,self.lockedFileName): self.raiseAnError(RuntimeError, self, "another instance of RAVEN is running in the working directory "+ self.workingDir+". Please check your input!")
      # register function to remove the locked file at the end of execution
      atexit.register(lambda filenamelocked: os.remove(filenamelocked),os.path.join(self.workingDir,self.lockedFileName))
    for inputFile in inputFiles: shutil.copy(inputFile,self.workingDir)
    self.raiseADebug('original input files copied in the current working dir: '+self.workingDir)
    self.raiseADebug('files copied:')
    self.raiseADebug( '  '+str(inputFiles))
    self.oriInputFiles = []
    for i in range(len(inputFiles)): self.oriInputFiles.append(os.path.join(self.workingDir,os.path.split(inputFiles[i])[1]))
    self.currentInputFiles        = None
    self.outFileRoot              = None

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    """ This function creates a new input
        It is called from a sampler to get the implementation specific for this model"""
    Kwargs['executable'] = self.executable
    found = False
    for index, inputFile in enumerate(currentInput):
      if inputFile.endswith(self.code.getInputExtension()):
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.code.getInputExtension()))
    Kwargs['outfile'] = 'out~'+os.path.split(currentInput[index])[1].split('.')[0]
    if len(self.alias.keys()) != 0: Kwargs['alias']   = self.alias
    return (self.code.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs),Kwargs)

  def run(self,inputFiles,jobHandler):
    """append a run at the externalRunning list of the jobHandler"""
    self.currentInputFiles = inputFiles[0]
    executeCommand, self.outFileRoot = self.code.genCommand(self.currentInputFiles,self.executable, flags=self.clargs, fileargs=self.fargs)
    #executeCommand, self.outFileRoot = self.code.generateCommand(self.currentInputFiles,self.executable)
    jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'],metadata=inputFiles[1],codePointer=self.code)
    found = False
    for index, inputFile in enumerate(self.currentInputFiles):
      if inputFile.endswith(self.code.getInputExtension()):
        found = True
        break
    if not found: self.raiseAnError(IOError,'None of the input files has one of the extensions requested by code '
                                  + self.subType +': ' + ' '.join(self.getInputExtension()))
    self.raiseAMessage('job "'+ self.currentInputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')

  def collectOutput(self,finisishedjob,output):
    """collect the output file in the output object"""
    if 'finalizeCodeOutput' in dir(self.code):
      out = self.code.finalizeCodeOutput(finisishedjob.command,finisishedjob.output,self.workingDir)
      if out: finisishedjob.output = out
    attributes={"input_file":self.currentInputFiles,"type":"csv","name":FileObject(os.path.join(self.workingDir,finisishedjob.output+'.csv'))}
    metadata = finisishedjob.returnMetadata()
    if metadata: attributes['metadata'] = metadata
    if output.type == "HDF5"        : output.addGroup(attributes,attributes)
    elif output.type in ['Point','PointSet','History','HistorySet']:
      output.addOutput(FileObject(os.path.join(self.workingDir,finisishedjob.output) + ".csv"),attributes)
      if metadata:
        for key,value in metadata.items(): output.updateMetadata(key,value,attributes)
    else: self.raiseAnError(ValueError,"output type "+ output.type + " unknown for Model Code "+self.name)
#
#
#
class Projector(Model):
  """Projector is a data manipulator"""
  @classmethod
  def specializeValidateDict(cls):
    pass
    #FIXME self.raiseAMessage('PROJECTOR','Remember to add the data type supported the class filter')

  def __init__(self):
    Model.__init__(self)
    self.printTag = 'PROJECTOR MODEL'

  def _readMoreXML(self,xmlNode):
    Model._readMoreXML(self, xmlNode)
    self.code = PostProcessors.returnInstance(self.subType,self)
    self.code._readMoreXML(xmlNode)

  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)

  def initialize(self,runInfoDict,myInput,initDict=None):
    if myInput.type == 'ROM':
      pass
    #initialize some of the current setting for the runs and generate the working
    #   directory with the starting input files
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try:                   os.mkdir(self.workingDir)
    except AttributeError: self.raiseAWarning('current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    """run calls the interface finalizer"""
    self.interface.run(inObj,outObj,self.workingDir)
#
#
#
class PostProcessor(Model, Assembler):
  """PostProcessor is an Action System. All the models here, take an input and perform an action"""
  @classmethod
  def specializeValidateDict(cls):
    cls.validateDict['Input']                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input'][0]['required'    ] = False
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][1]['class'       ] = 'Databases'
    cls.validateDict['Input'  ][1]['type'        ] = ['HDF5']
    cls.validateDict['Input'  ][1]['required'    ] = False
    cls.validateDict['Input'  ][1]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][2]['class'       ] = 'DataObjects'
    cls.validateDict['Input'  ][2]['type'        ] = ['Point','PointSet','History','HistorySet']
    cls.validateDict['Input'  ][2]['required'    ] = False
    cls.validateDict['Input'  ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][0]['class'       ] = 'Files'
    cls.validateDict['Output' ][0]['type'        ] = ['']
    cls.validateDict['Output' ][0]['required'    ] = False
    cls.validateDict['Output' ][0]['multiplicity'] = 'n'
    cls.validateDict['Output' ][1]['class'       ] = 'DataObjects'
    cls.validateDict['Output' ][1]['type'        ] = ['Point','PointSet','History','HistorySet']
    cls.validateDict['Output' ][1]['required'    ] = False
    cls.validateDict['Output' ][1]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][2]['class'       ] = 'Databases'
    cls.validateDict['Output' ][2]['type'        ] = ['HDF5']
    cls.validateDict['Output' ][2]['required'    ] = False
    cls.validateDict['Output' ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][3]['class'       ] = 'OutStreamManager'
    cls.validateDict['Output' ][3]['type'        ] = ['Plot','Print']
    cls.validateDict['Output' ][3]['required'    ] = False
    cls.validateDict['Output' ][3]['multiplicity'] = 'n'
    cls.validateDict['Function'] = [cls.testDict.copy()]
    cls.validateDict['Function'  ][0]['class'       ] = 'Functions'
    cls.validateDict['Function'  ][0]['type'        ] = ['External','Internal']
    cls.validateDict['Function'  ][0]['required'    ] = False
    cls.validateDict['Function'  ][0]['multiplicity'] = '1'
    cls.validateDict['ROM'] = [cls.testDict.copy()]
    cls.validateDict['ROM'       ][0]['class'       ] = 'Models'
    cls.validateDict['ROM'       ][0]['type'        ] = ['ROM']
    cls.validateDict['ROM'       ][0]['required'    ] = False
    cls.validateDict['ROM'       ][0]['multiplicity'] = '1'

  def __init__(self):
    Model.__init__(self)
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
    @ In , None, None
    @ Out, needDict, dictionary of objects needed (class:tuple(object type{if None, Simulation does not check the type}, object name))
    """
    return self.interface.whatDoINeed()

  def generateAssembler(self,initDict):
    """
    This method is used mainly by the Simulation class at the Step construction stage.
    It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
    It is an abstract method -> It must be implemented in the derived class!
    NB. In this implementation, the method only calls the self.interface.generateAssembler(initDict) method
    @ In , initDict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
    @ Out, None, None
    """
    self.interface.generateAssembler(initDict)

  def _readMoreXML(self,xmlNode):
    Model._readMoreXML(self, xmlNode)
    self.interface = PostProcessors.returnInstance(self.subType,self)
    self.interface._readMoreXML(xmlNode)

  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)

  def initialize(self,runInfo,inputs, initDict=None):
    """initialize some of the current setting for the runs and generate the working
       directory with the starting input files"""
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    self.interface.initialize(runInfo, inputs, initDict)
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(PostProcessors)))

  def run(self,Input,jobHandler):
    """run calls the interface finalizer"""
    if len(Input) > 0 : jobHandler.submitDict['Internal']((Input,),self.interface.run,str(0),modulesToImport = self.mods, globs = self.globs)
    else: jobHandler.submitDict['Internal']((None,),self.interface.run,str(0),modulesToImport = self.mods, globs = self.globs)

  def collectOutput(self,finishedjob,output):
    self.interface.collectOutput(finishedjob,output)

  def createNewInput(self,myInput,samplerType,**Kwargs):
    """just for compatibility"""
    return self.interface.inputToInternal(self,myInput)

"""
 Factory......
"""
__base = 'model'
__interFaceDict = {}
__interFaceDict['Dummy'         ] = Dummy
__interFaceDict['ROM'           ] = ROM
__interFaceDict['ExternalModel' ] = ExternalModel
__interFaceDict['Code'          ] = Code
__interFaceDict['Projector'     ] = Projector
__interFaceDict['PostProcessor' ] = PostProcessor
#__interFaceDict                   = (__interFaceDict.items()+CodeInterfaces.__interFaceDict.items()) #try to use this and remove the code interface
__knownTypes                      = list(__interFaceDict.keys())

#here the class methods are called to fill the information about the usage of the classes
for classType in __interFaceDict.values():
  classType.generateValidateDict()
  classType.specializeValidateDict()

def addKnownTypes(newDict):
  for name,value in newDict.items():
    __interFaceDict[name]=value
    __knownTypes.append(name)

def knownTypes():
  return __knownTypes

def returnInstance(Type,caller):
  """This function return an instance of the request model type"""
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'MODELS','not known '+__base+' type '+Type)

def validate(className,role,what,caller):
  """This is the general interface for the validation of a model usage"""
  if className in __knownTypes: return __interFaceDict[className].localValidateMethod(role,what)
  else : caller.raiseAnError(IOError,'MODELS','the class '+str(className)+' it is not a registered model')
