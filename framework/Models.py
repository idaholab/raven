'''
Created on Feb 19, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import shutil
import Datas
import numpy as np
from BaseType import BaseType
import SupervisedLearning
from Filters import *
#import Postprocessors
#import ROM interfaces

class Model(BaseType):
  ''' a model is something that given an input will return an output reproducing some physical model
      it could as complex as a stand alone code or a reduced order model trained somehow'''
  def __init__(self):
    BaseType.__init__(self)
    self.subType  = ''
    self.runQueue = []

  def readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['type']
    except: raise 'missed type for the model'+self.name

  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType

  def initialize(self,runInfo,inputs):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
    after this call the next one will be run
    @in runInfo is the run info from the jobHandler
    @in inputs is a list containing whatever is passed with an input role in the step'''
    pass

  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''this function have to return a new input that will be submitted to the model, it is called by the sampler
    @in myInput the inputs (list) to start from to generate the new one
    @in samplerType is the type of sampler that is calling to generate a new input
    @in **Kwargs is a dictionary that contains the information coming from the sampler,
         a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
    @return the new input (list)'''
    raise IOError('for this model the createNewInput has not yet being implemented')

  def run(self,Input,jobHandler):
    '''This call should be over loaded and should not return any results,
        possible it places a run one of the jobhadler lists!!!
        @in inputs is a list containing whatever is passed with an input role in the step
        @in jobHandler an instance of jobhandler that might be possible used to append a job for parallel running'''
    raise IOError('the model '+self.name+' that has no run method' )
  
  def collectOutput(self,collectFrom,storeTo):
    '''This call collect the output of the run
    @in collectFrom where the output is located, the form and the type is model dependent but should be compatible with the storeTo.addOutput method'''
    if 'addOutput' in dir(storeTo):
      try   : storeTo.addOutput(collectFrom)
      except: raise IOError('The place where to store the output '+type(storeTo)+' was not compatible with the addOutput of '+type(collectFrom))
    else: raise IOError('The place where to store the output has not a addOutput method')
     

class Code(Model):
  '''this is the generic class that import an external code into the framework'''
  def __init__(self):
    Model.__init__(self)
    self.executable         = ''   #name of the executable (abs path)
    self.oriInputFiles      = []   #list of the original input files (abs path)
    self.workingDir         = ''   #location where the code is currently running
    self.outFileRoot        = ''   #root to be used to generate the sequence of output files
    self.currentInputFiles  = []   #list of the modified (possibly) input files (abs path)
    self.infoForOut         = {}   #it contains the information needed for outputting 

  def readMoreXML(self,xmlNode):
    '''extension of info to be read for the Code(model)
    !!!!generate also the code interface for the proper type of code!!!!'''
    import CodeInterfaces
    Model.readMoreXML(self, xmlNode)
    try:
      self.executable = xmlNode.text
      abspath = os.path.abspath(self.executable)
      if os.path.exists(abspath):
        self.executable = abspath
    except: raise IOError('not found executable '+xmlNode.text)
    self.interface = CodeInterfaces.returnCodeInterface(self.subType)
    
  def addInitParams(self,tempDict):
    '''extension of addInitParams for the Code(model)'''
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
  
  def addCurrentSetting(self,originalDict):
    '''extension of addInitParams for the Code(model)'''
    originalDict['current working directory'] = self.workingDir
    originalDict['current output file root']  = self.outFileRoot
    originalDict['current input file']        = self.currentInputFiles
    originalDict['original input file']       = self.oriInputFiles

  def initialize(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except: print('MODEL CODE    : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
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
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'out~'+os.path.split(currentInput[index])[1].split('.')[0]
    self.infoForOut[Kwargs['prefix']] = copy.deepcopy(Kwargs)
    return self.interface.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs)

  def run(self,inputFiles,jobHandler):
    '''append a run at the externalRunning list of the jobHandler'''
    self.currentInputFiles = inputFiles
    executeCommand, self.outFileRoot = self.interface.generateCommand(self.currentInputFiles,self.executable)
    jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'])
    if self.currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    if self.debug: print('MODEL CODE    : job "'+ inputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')

  def collectOutput(self,finisishedjob,output):
    '''collect the output file in the output object'''
    # TODO This errors if output doesn't have .type (csv for example), it will be necessary a file class
    try:
      if output.type == "HDF5":
        self.__addDataBaseGroup(finisishedjob,output)
        return
    except AttributeError:
      pass
    output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv")
    return

  def __addDataBaseGroup(self,finisishedjob,database):
    # add a group into the database
    attributes={}
    attributes["input_file"] = self.currentInputFiles
    attributes["type"] = "csv"
    attributes["name"] = os.path.join(self.workingDir,finisishedjob.output+'.csv')
    if finisishedjob.identifier in self.infoForOut:
      infoForOut = self.infoForOut.pop(finisishedjob.identifier)
      for key in infoForOut:
        attributes[key] = infoForOut[key]
    database.addGroup(attributes,attributes)


class ROM(Model):
  '''ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome'''
  def __init__(self):
    Model.__init__(self)
    self.initializzationOptionDict = {}
    self.inputNames = []
    self.outputName = ''

  def readMoreXML(self,xmlNode):
    '''read the additional input needed and create an instance the underlying ROM'''
    Model.readMoreXML(self, xmlNode)
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except: self.initializzationOptionDict[child.tag] = child.text
    self.ROM =  SupervisedLearning.returnInstance(self.subType)
    self.SupervisedEngine = self.ROM(**self.initializzationOptionDict)
    
  def addInitParams(self,originalDict):
    ROMdict = self.SupervisedEngine.returnInitialParamters()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]

  def addCurrentSetting(self,originalDict):
    ROMdict = self.SupervisedEngine.returnCurrentSetting()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]

  def initializeTrain(self,runInfoDict,loadFrom):
    '''just provide an internal pointer to the external data and check compatibility'''
    if loadFrom.type not in self.ROM.admittedData: raise IOError('type '+loadFrom.type+' is not compatible with the ROM '+self.ROM.type)
    else: self.toLoadFrom = loadFrom

  def close(self):
    '''remember to call this function to decouple the data owned by the ROM and the environment data after each training'''
    self.toLoadFrom = copy.deepcopy(self.toLoadFrom)

  def train(self):
    '''Here we do the training of the ROM'''
    '''Fit the model according to the given training data.
    @in X : {array-like, sparse matrix}, shape = [n_samples, n_features] Training vector, where n_samples in the number of samples and n_features is the number of features.
    @in y : array-like, shape = [n_samples] Target vector relative to X class_weight : {dict, 'auto'}, optional Weights associated with classes. If not given, all classes
            are supposed to have weight one.'''
    self.inputNames, inputsValues = self.toLoadFrom.getInpParametersValues().items()
    try: outputValues = self.toLoadFrom.getOutParametersValues()[self.outputName]
    except: raise IOError('The output sought '+self.outputName+' is not in the training set')
    self.inputsValues = np.zeros(shape=(inputsValues[1].size,len(self.inputNames)))
    self.outputValues = np.zeros(shape=(inputsValues[1].size))
    for i in range(len(self.inputNames)):
      self.inputsValues[:,i] = inputsValues[i][:]
    self.outputValues[:] = outputValues[:]
    self.SupervisedEngine.train(self.inputsValues,self.outputValues)

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input
        It is called from a sampler to get the implementation specific for this model
        it support string input
        dictionary input and datas input'''
    if len(currentInput)>1: raise IOError('ROM accepts only one input not a list of inputs')
    else: currentInput =currentInput[0]
    if  type(currentInput)==str:#one input point requested a as a string
      inputNames  = [component.split('=')[0] for component in currentInput.split(',')]
      inputValues = [component.split('=')[1] for component in currentInput.split(',')]
      for name, newValue in Kwargs['sampledVars'].items(): inputValues[inputNames.index(name)] = newValue
      newInput = [inputNames[i]+'='+inputValues[i] for i in range(inputNames)]
      newInput = newInput.join(',')
    elif type(currentInput)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      for name, newValue in Kwargs['sampledVars'].items(): currentInput[name] = newValue
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        if currentInput.type in self.ROM.admittedData:
          for name, newValue in Kwargs['sampledVars'].items():
            try   : self.currentInput.getInpParametersValues()[name] = np.array(newValue)
            except: raise IOError('trying to sample '+name+' that is not in the original input')
      except: raise IOError('the request of ROM evaluation is done via a not compatible input')
    currentInput = [currentInput]
    return currentInput

  def run(self,request,jobHandler):
    '''This call run a ROM as a model
    The input should be translated in to a set of coordinate where to perform the predictions
    It is possible either to send in just one point in the input space or a set of points
    input are accepted in the following form:
    -as a strings: 'input_name=value,input_name=value,..' this supports only one point in the input space
    -as a dictionary where keys are the input names and the values the corresponding values (it should be either vector or list)
    -as one of the admitted data for the specific ROM sub-type among the data type available in the datas.py module'''
    if len(request)>1: raise IOError('ROM accepts only one input not a list of inputs')
    else: request =request[0]
    #first we extract the input names and the corresponding values (it is an implicit mapping)
    if  type(request)==str:#one input point requested a as a string
      inputNames  = [entry.split('=')[0]  for entry in request.split(',')]
      inputValues = [entry.split('=')[1]  for entry in request.split(',')]
    elif type(request)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      inputNames, inputValues = request.items()[:][0], request.items()[:][1]
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        if input.type in self.ROM.admittedData:
          inputNames, inputValues = self.request.getInpParametersValues().items()[:][0], self.request.getInpParametersValues().items()[:][1]
      except: raise IOError('the request of ROM evaluation is done via a not compatible data')
    #now that the prediction points are read we check the compatibility with the ROM input-output set
    lenght = len(set(inputNames).intersection(self.inputNames))
    if lenght!=len(self.inputNames) or lenght!=len(inputNames):
      raise IOError ('there is a mismatch between the provided request and the ROM structure')
    #build a mapping from the ordering of the input sent in and the ordering inside the ROM
    self.requestToLocalOrdering = []
    for local in self.inputNames:
      self.requestToLocalOrdering.append(inputNames.index(local))
    #building the arrays to send in for the prediction by the ROM
    self.request = np.array([inputValues[index] for index in self.requestToLocalOrdering]).T
    ############################------FIXME----------#######################################################
    # we need to submit self.ROM.evaluate(self.request) to the job handler
    self.output = self.ROM.evaluate(self.request)
#    raise IOError('the multi treading is not yet in place neither the appending')
  
  def collectOutput(self,finishedJob,output):
    '''This method append the ROM evaluation into the output'''
    try: #try is used to be sure input.type exist
      if output.type in self.ROM.admittedData:
        for inputName in self.inputNames:
          self.request.getInpParametersValues[inputName] = np.concatenate((self.request.getInpParametersValues[inputName], self.request[self.inputNames.index(inputName)][:]))
    except: raise IOError('the output of the ROM is requested on a not compatible data')
    self.request.getOutParametersValues[self.outputName] = np.concatenate((self.request.getOutParametersValues[inputName], self.output))
  
class Projector(Model):
  '''Projector is a data manipulator'''
  def __init__(self):
    Model.__init__(self)

  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    self.interface = returnFilterInterface(self.subType)
    self.interface.readMoreXML(xmlNode)
 
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)

  def reset(self,runInfoDict,input):
    if input.type == 'ROM':
      pass
    #initialize some of the current setting for the runs and generate the working 
    #   directory with the starting input files
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except: print('MODEL FILTER  : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    self.interface.finalizeFilter(inObj,outObj,self.workingDir)
    
    
    

class Filter(Model):
  '''Filter is an Action System. All the models here, take an input and perform an action'''
  def __init__(self):
    Model.__init__(self)
    self.input  = {}     # input source
    self.action = None   # action
    self.workingDir = ''

  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    self.interface = returnFilterInterface(self.subType)
    self.interface.readMoreXML(xmlNode)
 
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)

  def reset(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except: print('MODEL FILTER  : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    for i in range(len(inObj)):
      self.interface.finalizeFilter(inObj[i],outObj[i],self.workingDir)
  def collectOutput(self,finishedjob,output):
    self.interface.collectOutput(output)



base = 'model'
InterfaceDict = {}
InterfaceDict['ROM'      ] = ROM
InterfaceDict['Code'     ] = Code
InterfaceDict['Filter'   ] = Filter
InterfaceDict['Projector'] = Projector

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
