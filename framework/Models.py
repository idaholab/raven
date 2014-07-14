'''
Module where the base class and the specialization of different type of Model are
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import shutil
import numpy
from utils import metaclass_insert
import abc
import importlib
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseType import BaseType
import SupervisedLearning
import PostProcessors #import returnFilterInterface
import Samplers
#Internal Modules End--------------------------------------------------------------------------------

class Model(metaclass_insert(abc.ABCMeta,BaseType)):
  '''
  A model is something that given an input will return an output reproducing some physical model
  it could as complex as a stand alone code, a reduced order model trained somehow or something
  externally build and imported by the user
  '''
  validateDict                  = {}
  validateDict['Input'  ]       = []
  validateDict['Output' ]       = []
  validateDict['Sampler']       = []
  testDict                      = {}
  testDict                      = {'class':'','type':[''],'multiplicity':0,'required':False}
  print('FIXME: a multiplicity value is needed to control role that can have different class')
  #the possible inputs
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][0]['class'       ] = 'Datas'
  validateDict['Input'  ][0]['type'        ] = ['TimePoint','TimePointSet','History','Histories']
  validateDict['Input'  ][0]['required'    ] = False
  validateDict['Input'  ][0]['multiplicity'] = 'n'
  validateDict['Input'].append(testDict.copy())
  validateDict['Input'  ][1]['class'       ] = 'Files'
  validateDict['Input'  ][1]['type'        ] = ['']
  validateDict['Input'  ][1]['required'    ] = False
  validateDict['Input'  ][1]['multiplicity'] = 'n'
  #the possible outputs
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][0]['class'       ] = 'Datas'
  validateDict['Output' ][0]['type'        ] = ['TimePoint','TimePointSet','History','Histories']
  validateDict['Output' ][0]['required'    ] = False
  validateDict['Output' ][0]['multiplicity'] = 'n'
  validateDict['Output'].append(testDict.copy())
  validateDict['Output' ][1]['class'       ] = 'DataBases'
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
  validateDict['Sampler'][0]['type'        ] = Samplers.knonwnTypes()
  validateDict['Sampler'][0]['required'    ] = False
  validateDict['Sampler'][0]['multiplicity'] = 1

  @classmethod
  def generateValidateDict(cls):
    '''This method generate a independent copy of validateDict for the calling class'''
    cls.validateDict = copy.deepcopy(Model.validateDict)

  @classmethod
  def specializeValidateDict(cls):
    ''' This method should be overridden to describe the types of input accepted with a certain role by the model class specialization'''
    raise NotImplementedError('The class '+str(cls.__name__)+' has not implemented the method specializeValidateDict')

  @classmethod
  def localValidateMethod(cls,who,what):
    '''
    This class method is called to test the compatibility of the class with its possible usage
    @in who: a string identifying the what is the role of what we are going to test (i.e. input, output etc)
    @in what: a list (or a general iterable) that will be playing the 'who' role
    ''' 
    #counting successful matches
    if who not in cls.validateDict.keys(): raise IOError ('The role '+str(who)+' does not exist in the class '+str(cls))
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
          raise IOError ('The number of time class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper')
        if tester['multiplicity']!='n' and tester['tempCounter']!=tester['multiplicity']:
          raise IOError ('The number of time class = '+str(tester['class'])+' type= ' +str(tester['type'])+' is used as '+str(who)+' is improper')
    #testing if all argument to be tested have been found
    for anItem in what:
      if anItem['found']==False:
        raise IOError ('It is not possible to use '+anItem['class']+' type= ' +anItem['type']+' as '+who)
    return True

  def __init__(self):
    BaseType.__init__(self)
    self.subType  = ''
    self.runQueue = []
    self.FIXME = False

  def _readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['subType']
    except KeyError: 
      print("Failed in Node: ",xmlNode)
      raise Exception('missed subType for the model '+self.name)
    del(xmlNode.attrib['subType'])
  
  def localInputAndChecks(self,xmlNode):
    '''place here the additional reading, remember to add initial parameters in the method localAddInitParams'''

  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType

  def localAddInitParams(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''

  def initialize(self,runInfo,inputs,initDict=None):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
    after this call the next one will be run
    @in runInfo is the run info from the jobHandler
    @in inputs is a list containing whatever is passed with an input role in the step'''
    pass

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''
    this function have to return a new input that will be submitted to the model, it is called by the sampler
    @in myInput the inputs (list) to start from to generate the new one
    @in samplerType is the type of sampler that is calling to generate a new input
    @in **Kwargs is a dictionary that contains the information coming from the sampler,
         a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
    @return the new input in a list form
    '''
    return [(copy.deepcopy(Kwargs))]
  
  @abc.abstractmethod
  def run(self,Input,jobHandler):
    '''
    This call should be over loaded and should not return any results,
    possible it places a run one of the jobhadler lists!!!
    @in inputs is a list containing whatever is passed with an input role in the step
    @in jobHandler an instance of jobhandler that might be possible used to append a job for parallel running
    '''
    pass
  
  def collectOutput(self,collectFrom,storeTo):
    '''
    This call collect the output of the run
    @in collectFrom: where the output is located, the form and the type is model dependent but should be compatible with the storeTo.addOutput method.
    '''
    #if a addOutput is present in nameSpace of storeTo it is used
    if 'addOutput' in dir(storeTo): storeTo.addOutput(collectFrom)
    else                          : raise IOError('The place where to store the output has not a addOutput method')
#
#
#
class Dummy(Model):
  '''
  this is a dummy model that just return the effect of the sampler. The values reported as input in the output
  are the output of the sampler and the output is the counter of the performed sampling
  '''
  def __init__(self):
    Model.__init__(self)
    self.admittedData = self.__class__.validateDict['Input' ][0]['type'] #the list of admitted data is saved also here for run time checks
    #the following variable are reset at each call of the initialize method
    self.counterInput = 0   #the number of input already generated
    
  @classmethod
  def specializeValidateDict(cls):
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['type'        ] = ['TimePoint','TimePointSet']
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['TimePoint','TimePointSet']
    
  def initialize(self,runInfo,inputs,initDict=None):
    self.counterInput = 0

  def _manipulateInput(self,dataIn):
    if len(dataIn)>1: raise IOError('Only one input is accepted by the model type '+self.type+' with name'+self.name)
    if type(dataIn[0])!=tuple: inRun = self._inputToInternal(dataIn[0]) #this might happen when a single run is used and the input it does not come from self.createNewInput
    else:                      inRun = dataIn[0][0]    
    return inRun

  def _inputToInternal(self,dataIN,full=False):
    '''Transform it in the internal format the provided input. dataIN could be either a dictionary (then nothing to do) or one of the admitted data'''  
    if self.FIXME: print('FIXME: wondering if a dictionary compatibility should be kept')
    if  type(dataIN)!=dict and dataIN.type not in self.admittedData: raise IOError('type '+dataIN.type+' is not compatible with the ROM '+self.name)
    if full==True:  length = 0
    if full==False: length = -1
    localInput = {}
    if type(dataIN)!=dict:
      for entries in dataIN.getParaKeys('inputs' ):
        if not dataIN.isItEmpty(): localInput[entries] = copy.copy(dataIN.getParam('input' ,entries)[length:])
        else:                      localInput[entries] = None
      for entries in dataIN.getParaKeys('outputs'):
        if not dataIN.isItEmpty(): localInput[entries] = copy.copy(dataIN.getParam('output',entries)[length:])
        else:                      localInput[entries] = None
      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in dataIN.getParaKeys('outputs'): localInput.pop('OutputPlaceHolder') # this remove the counter from the inputs to be placed among the outputs
    else: localInput = dataIN #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
    return localInput
  
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''
    here only TimePoint and TimePointSet are accepted a local copy of the values is performed.
    For a TimePoint all value are copied, for a TimePointSet only the last set of entry
    The copied values are returned as a dictionary back
    '''
    if len(myInput)>1: raise IOError('Only one input is accepted by the model type '+self.type+' with name'+self.name)
    inputDict = self._inputToInternal(myInput[0])
    self.counterInput +=1 
    #test if all sampled variables are in the inputs category of the data
    if set(list(Kwargs['SampledVars'].keys())+list(inputDict.keys())) != set(list(inputDict.keys())):
      raise IOError ('When trying to sample the input for the model '+self.name+' of type '+self.type+' the sampled variable are '+str(Kwargs['SampledVars'].keys())+' while the variable in the input are'+str(inputDict.keys()))
    for key in Kwargs['SampledVars'].keys(): inputDict[key] = numpy.atleast_1d(Kwargs['SampledVars'][key])
    if None in inputDict.values(): raise IOError ('While preparing the input for the model '+self.type+' with name'+self.name+' found an None input variable '+ str(inputDict.items()))
    #the inputs/outputs should not be store locally since they might be used as a part of a list of input for the parallel runs
    #same reason why it should not be used the value of the counter inside the class but the one returned from outside as a part of the input
    return [(inputDict)],copy.deepcopy(Kwargs) 
  
  def run(self,Input,jobHandler):
    '''
    The input is a list of one element.
    The element is either a tuple of two dictionary [(InputDictionary, OutputDictionary)] if the input has been created by the  self.createNewInput
    otherwise is one of the accepted data.
    The first is the input the second the output. The output is just the counter
    '''
    #this set of test is performed to avoid that if used in a single run we come in with the wrong input structure since the self.createNewInput is not called
    inRun = self._manipulateInput(Input[0])
    lambdaReturnOut = lambda inRun: {'OutputPlaceHolder':numpy.atleast_1d(numpy.float(Input[1]['prefix']))}
    jobHandler.submitDict['Internal']((inRun,),lambdaReturnOut,str(Input[1]['prefix']),metadata=Input[1])
    
  def collectOutput(self,finishedJob,output):
    if finishedJob.returnEvaluation() == -1: raise Exception("MODEL DUMMY: ERROR -> No available Output to collect (Run probabably is not finished yet)")
    exportDict = {'input_space_params':copy.deepcopy(finishedJob.returnEvaluation()[0]),'output_space_params':copy.deepcopy(finishedJob.returnEvaluation()[1]),'metadata':copy.deepcopy(finishedJob.returnMetadata())}
    
    if output.type == 'HDF5': output.addGroupDatas({'group':self.name+str(finishedJob.identifier)},exportDict,False)
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
  '''ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome'''
  @classmethod
  def specializeValidateDict(cls):
    print(cls.specializeValidateDict.__doc__)
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['TimePoint','TimePointSet']
    
  def __init__(self):
    Dummy.__init__(self)
    self.initializzationOptionDict = {}
    self.amITrained   = False
  
  def _readMoreXML(self,xmlNode):
    Dummy._readMoreXML(self, xmlNode)
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except ValueError:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except ValueError: self.initializzationOptionDict[child.tag] = child.text
    #the ROM is instanced and initialized
    print(self.initializzationOptionDict)
    self.SupervisedEngine = SupervisedLearning.returnInstance(self.subType,**self.initializzationOptionDict)
    
  def addInitParams(self,originalDict):
    '''the ROM setting parameters are added'''
    ROMdict = self.SupervisedEngine.returnInitialParamters()
    for key in ROMdict.keys(): originalDict[key] = ROMdict[key]
  
  def train(self,trainingSet):
    '''Here we do the training of the ROM'''
    '''Fit the model according to the given training data.
    @in X : {array-like, sparse matrix}, shape = [n_samples, n_features] Training vector, where n_samples in the number of samples and n_features is the number of features.
    @in y : array-like, shape = [n_samples] Target vector relative to X class_weight : {dict, 'auto'}, optional Weights associated with classes. If not given, all classes
            are supposed to have weight one.'''
    self.trainingSet = copy.copy(self._inputToInternal(trainingSet,full=True))
    self.SupervisedEngine.train(self.trainingSet)
    self.amITrained = self.SupervisedEngine.amITrained
    if self.FIXME:print('FIXME: add self.amITrained to currentParamters')
  
  def confidence(self,request):
    '''
    This is to get a value that is inversely proportional to the confidence that we have
    forecasting the target value for the given set of features. The reason to chose the inverse is because
    in case of normal distance this would be 1/distance that could be infinity
    '''
    inputToROM = self._inputToInternal(request)
    return self.SupervisedEngine.confidence(inputToROM)

  def evaluate(self,request):
    '''when the ROM is used directly without need of having the sampler passing in the new values evaluate instead of run should be used'''
    inputToROM = self._inputToInternal(request)
    return self.SupervisedEngine.evaluate(inputToROM)

  def run(self,Input,jobHandler):
    '''This call run a ROM as a model'''
    inRun = self._manipulateInput(Input[0])
    lambdaReturnOut = lambda inRun: {self.SupervisedEngine.target:self.evaluate(inRun)}
    jobHandler.submitDict['Internal']((inRun,),lambdaReturnOut,str(Input[1]['prefix']),metadata=Input[1])
#
#
#  
class ExternalModel(Dummy):
  ''' External model class: this model allows to interface with an external python module'''
  @classmethod
  def specializeValidateDict(cls):
    #one data is needed for the input
    print('FIXME: think about how to import the roles to allowed class for the external model. For the moment we have just all')

  def __init__(self):
    Dummy.__init__(self)
    self.modelVariableValues = {}
    self.modelVariableType   = {}
    self.__availableVariableTypes = ['float','int','bool','numpy.ndarray']

  def initialize(self,runInfo,inputs,initDict=None):
    if 'initialize' in dir(self.sim): self.sim.initialize(self,runInfo,inputs)
    Dummy.initialize(self, runInfo, inputs)      
  
  def createNewInput(self,myInput,samplerType,**Kwargs):
    if 'createNewInput' in dir(self.sim): 
      extCreateNewInput = self.sim.createNewInput(self,myInput,samplerType,**Kwargs)
      if extCreateNewInput== None: raise Exception('MODEL EXTERNAL: ERROR -> in external Model '+self.ModuleToLoad+' the method createNewInput must return something. Got: None')
      self.counterInput += 1
      return [(extCreateNewInput)],copy.deepcopy(Kwargs)
    else                                : return Dummy.createNewInput(self, myInput,samplerType,**Kwargs)   

  def _readMoreXML(self,xmlNode):
    Model._readMoreXML(self, xmlNode)
    if 'ModuleToLoad' in xmlNode.attrib.keys(): 
      self.ModuleToLoad = os.path.split(str(xmlNode.attrib['ModuleToLoad']))[1]
      if (os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0] != ''):
        abspath = os.path.abspath(os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0])
        if '~' in abspath:abspath = os.path.expanduser(abspath)
        if os.path.exists(abspath): os.sys.path.append(abspath)
        else: raise IOError('MODEL EXTERNAL: ERROR -> The path provided for the external model does not exist!!! Got: ' + abspath)
    else: raise IOError('MODEL EXTERNAL: ERROR -> ModuleToLoad not provided for module externalModule')
    # load the external module and point it to self.sim
    self.sim=__import__(self.ModuleToLoad)
    # check if there are variables and, in case, load them
    for son in xmlNode:
      if son.tag=='variable':
        self.modelVariableValues[son.text] = None
        exec('self.'+son.text+' = self.modelVariableValues['+'son.text'+']')
        if 'type' in son.attrib.keys():
          if not (son.attrib['type'].lower() in self.__availableVariableTypes):
            raise IOError('MODEL EXTERNAL: ERROR -> the "type" of variable ' + son.text + 'not among available types')
          self.modelVariableType[son.text] = son.attrib['type']
        else: raise IOError('MODEL EXTERNAL: ERROR -> the attribute "type" for variable '+son.text+' is missed')
    # check if there are other information that the external module wants to load
    if '_readMoreXML' in dir(self.sim): self.sim._readMoreXML(self,xmlNode)
  def __externalRun(self, Input): 
    if 'createNewInput' not in dir(self.sim):
      for key in Input.keys(): self.modelVariableValues[key] = Input[key]
      self.__uploadValues() 
    self.sim.run(self,Input)
    
    self.__pointSolution()
    return copy.deepcopy(self.modelVariableValues) 

  def run(self,Input,jobHandler):
    inRun = copy.deepcopy(self._manipulateInput(Input[0]))
    jobHandler.submitDict['Internal']((inRun,),self.__externalRun,str(Input[1]['prefix']),metadata=Input[1])  
    
  def collectOutput(self,finishedJob,output):
    if finishedJob.returnEvaluation() == -1: raise Exception("MODEL EXTERNAL: ERROR -> No available Output to collect (Run probabably is not finished yet)")
    def typeMatch(var,var_type_str):
      type_var = type(var)
      return type_var.__name__ == var_type_str or \
        type_var.__module__+"."+type_var.__name__ == var_type_str
    # check type consistency... This is needed in order to keep under control the external model... In order to avoid problems in collecting the outputs in our internal structures
    for key in finishedJob.returnEvaluation()[1]: 
      if not (typeMatch(finishedJob.returnEvaluation()[1][key],self.modelVariableType[key])): raise RuntimeError('MODEL EXTERNAL: ERROR -> type of variable '+ key + ' is ' + str(type(finishedJob.returnEvaluation()[1][key]))+' and mismatches with respect to the input ones (' + self.modelVariableType[key] +')!!!') 
    Dummy.collectOutput(self, finishedJob, output)
    
  def __pointSolution(self):
    for variable in self.modelVariableValues.keys(): exec('self.modelVariableValues[variable] = self.'+  variable)

  def __uploadValues(self):
    for variable in self.modelVariableValues.keys(): exec('self.'+ variable +' = self.modelVariableValues[variable]')
#
#
#
class Code(Model):
  '''this is the generic class that import an external code into the framework'''
  CodeInterfaces = importlib.import_module("CodeInterfaces")
  @classmethod
  def specializeValidateDict(cls):
    print('think about how to import the roles to allowed class for the codes. For the moment they are not specialized by executable')
    cls.validateDict['Input'] = [cls.validateDict['Input'][1]]

  def __init__(self):
    Model.__init__(self)
    self.executable         = ''   #name of the executable (abs path)
    self.oriInputFiles      = []   #list of the original input files (abs path)
    self.workingDir         = ''   #location where the code is currently running
    self.outFileRoot        = ''   #root to be used to generate the sequence of output files
    self.currentInputFiles  = []   #list of the modified (possibly) input files (abs path)
    self.alias              = {}   #if alias are defined in the input it defines a mapping between the variable names in the framework and the one for the generation of the input
                                   #self.alias[framework variable name] = [input code name]. For Example, for a MooseBasedApp, the alias would be self.alias['internal_variable_name'] = 'Material|Fuel|thermal_conductivity'

  def _readMoreXML(self,xmlNode):
    '''extension of info to be read for the Code(model)
    !!!!generate also the code interface for the proper type of code!!!!'''
    Model._readMoreXML(self, xmlNode)

    for child in xmlNode:
      if child.tag=='executable': 
        self.executable = str(child.text)
      elif child.tag=='alias':
        # the input would be <alias variable='internal_variable_name'>Material|Fuel|thermal_conductivity</alias>
        if 'variable' in child.attrib.keys(): self.alias[child.attrib['variable']] = child.text
        else: raise Exception ('not found the attribute variable in the definition of one of the alias for code model '+str(self.name))
      else: raise Exception ('unknown tag within the definition of the code model '+str(self.name))
    if self.executable == '': raise IOError('MODEL CODE    : not found the node <executable> in the body of the code model '+str(self.name))
    if '~' in self.executable: self.executable = os.path.expanduser(self.executable)
    abspath = os.path.abspath(self.executable)
    if os.path.exists(abspath):
      self.executable = abspath
    else: print('not found executable '+self.executable)
    self.code = Code.CodeInterfaces.returnCodeInterface(self.subType)
    print('please finish the importing of available codes and valid interface form the codeInterfaces')
    
  def addInitParams(self,tempDict):
    '''extension of addInitParams for the Code(model)'''
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
    for key, value in self.alias.items():
      tempDict['The code variable '+str(value)+' it is filled using the framework variable '] = key
      
  def addCurrentSetting(self,originalDict):
    '''extension of addInitParams for the Code(model)'''
    originalDict['current working directory'] = self.workingDir
    originalDict['current output file root' ] = self.outFileRoot
    originalDict['current input file'       ] = self.currentInputFiles
    originalDict['original input file'      ] = self.oriInputFiles

  def initialize(self,runInfoDict,inputFiles,initDict=None):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except OSError: print('MODEL CODE    : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    for inputFile in inputFiles:    
      shutil.copy(inputFile,self.workingDir)
    if self.debug: print('MODEL CODE    : original input files copied in the current working dir: '+self.workingDir)
    if self.debug: print('MODEL CODE    : files copied:')
    if self.debug: print(inputFiles)
    self.oriInputFiles = []
    for i in range(len(inputFiles)):
      self.oriInputFiles.append(os.path.join(self.workingDir,os.path.split(inputFiles[i])[1]))
    self.currentInputFiles        = None
    self.outFileRoot              = None

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input
        It is called from a sampler to get the implementation specific for this model'''
    Kwargs['executable'] = self.executable
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'out~'+os.path.split(currentInput[index])[1].split('.')[0]
    if len(self.alias.keys()) != 0: Kwargs['alias']   = self.alias
    return (self.code.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs),copy.deepcopy(Kwargs))
 
  def run(self,inputFiles,jobHandler):
    '''append a run at the externalRunning list of the jobHandler'''
    self.currentInputFiles = inputFiles[0]
    executeCommand, self.outFileRoot = self.code.generateCommand(self.currentInputFiles,self.executable)
    jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'],metadata=inputFiles[1])
    if self.currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    if self.debug: print('MODEL CODE    : job "'+ self.currentInputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')

  def collectOutput(self,finisishedjob,output):
    '''collect the output file in the output object'''
    # TODO This errors if output doesn't have .type (csv for example), it will be necessary a file class
    attributes={"input_file":self.currentInputFiles,"type":"csv","name":os.path.join(self.workingDir,finisishedjob.output+'.csv')}
    metadata = finisishedjob.returnMetadata()
    if metadata:
      #for key in metadata: attributes[key] = metadata[key]
      attributes['metadata'] = metadata
    try:                   output.addGroup(attributes,attributes)
    except AttributeError: 
      output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv",attributes)
      if metadata: 
        for key,value in metadata.items(): output.updateMetadata(key,value,attributes)
#
#
#
class Projector(Model):
  '''Projector is a data manipulator'''
  @classmethod
  def specializeValidateDict(cls):
    print('Remember to add the data type supported the class filter')

  def __init__(self):
    Model.__init__(self)

  def _readMoreXML(self,xmlNode):
    Model._readMoreXML(self, xmlNode)
    self.code = PostProcessors.returnInstance(self.subType)
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
    except AttributeError: print('MODEL FILTER  : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    self.interface.run(inObj,outObj,self.workingDir)
#
#
#
class PostProcessor(Model):
  '''Filter is an Action System. All the models here, take an input and perform an action'''
  @classmethod
  def specializeValidateDict(cls):
    cls.validateDict['Input']                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input'][0]['required'    ] = False
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][1]['class'       ] = 'DataBases'
    cls.validateDict['Input'  ][1]['type'        ] = ['HDF5']
    cls.validateDict['Input'  ][1]['required'    ] = False
    cls.validateDict['Input'  ][1]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][2]['class'       ] = 'Datas'
    cls.validateDict['Input'  ][2]['type'        ] = ['TimePoint','TimePointSet','History','Histories']
    cls.validateDict['Input'  ][2]['required'    ] = False
    cls.validateDict['Input'  ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][0]['class'       ] = 'Files'
    cls.validateDict['Output' ][0]['type'        ] = ['']
    cls.validateDict['Output' ][0]['required'    ] = False
    cls.validateDict['Output' ][0]['multiplicity'] = 'n'
    cls.validateDict['Output' ][1]['class'       ] = 'Datas'
    cls.validateDict['Output' ][1]['type'        ] = ['TimePoint','TimePointSet','History','Histories']
    cls.validateDict['Output' ][1]['required'    ] = False
    cls.validateDict['Output' ][1]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][2]['class'       ] = 'DataBases'
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

  def _readMoreXML(self,xmlNode):
    Model._readMoreXML(self, xmlNode)
    self.interface = PostProcessors.returnInstance(self.subType)
    self.interface._readMoreXML(xmlNode)
 
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)
  
  
  def initialize(self,runInfo,inputs, initDict=None):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    self.interface.initialize(runInfo, inputs, initDict)
    
  def run(self,Input,jobHandler):
    '''run calls the interface finalizer'''
    if len(Input) > 0 :
      for i in range(len(Input)):
        lumbdaToRun = lambda x: self.interface.run(x)
        jobHandler.submitDict['Internal'](((Input[i]),),lumbdaToRun,str(i)) 
    else:
      lumbdaToRun = lambda x: self.interface.run(x)
      jobHandler.submitDict['Internal'](((None),),lumbdaToRun,str(0))   
 
  def collectOutput(self,finishedjob,output):
    self.interface.collectOutput(finishedjob,output)
    
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''just for compatibility'''
    return self.interface.inputToInternal(self,myInput)
     

'''
 Factory......
'''
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

def knonwnTypes():
  return __knownTypes

def returnInstance(Type,debug=False):
  '''This function return an instance of the request model type'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)

def validate(className,role,what,debug=False):
  '''This is the general interface for the validation of a model usage'''
  if className in __knownTypes: return __interFaceDict[className].localValidateMethod(role,what)
  else                        : raise IOError('the class '+str(className)+' it is not a registered model')
    
  
  
