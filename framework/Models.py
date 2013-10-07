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
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step'''
    raise IOError('the model '+self.name+' that has no initialize method' )

  def run(self,*args):
    '''This call should be over loaded and return a jobHandler.External/InternalRunner'''
    raise IOError('the model '+self.name+' that has no run method' )
  
  def collectOutput(self,collectFrom,storeTo):
    storeTo.addOutput(collectFrom)

  def createNewInput(self,myInput,samplerType,**Kwargs):
    raise IOError('for this model the createNewInput has not yet being implemented')

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
    return #self.oriInputFiles

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input
        It is called from a sampler to get the implementation specific for this model'''
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'out~'+os.path.split(currentInput[index])[1].split('.')[0]
    self.infoForOut[Kwargs['prefix']] = copy.deepcopy(Kwargs)
    return self.interface.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs)

  def run(self,inputFiles,outputDatas,jobHandler):
    '''append a run at the externalRunning list of the jobHandler'''
    self.currentInputFiles = inputFiles
    executeCommand, self.outFileRoot = self.interface.generateCommand(self.currentInputFiles,self.executable)
    jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'])
    if self.currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    if self.debug: print('MODEL CODE    : job "'+ inputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')

  def collectOutput(self,finisishedjob,output):
    '''collect the output file in the output object'''
    # TODO This errors if output doesn't have .type (csv for example)
    try:
      if output.type == "HDF5":
        self.__addDataBaseGroup(finisishedjob,output)
        return
    except AttributeError:
      pass
    #print('this:',output)
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
    '''remember to call this function to decouple the data owned by the ROM and the environment data'''
    self.toLoadFrom = copy.deepcopy(self.toLoadFrom)

  def train(self):
    '''Here we do the training of the ROM'''
    '''Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        class_weight : {dict, 'auto'}, optional
            Weights associated with classes. If not given, all classes
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

  def run(self,request,outputDatas,jobHandler):
    '''This call run a ROM as a model
    The input should be translated in to a set of coordinate where to perform the predictions
    It is possible either to send in just one point in the input space or a set of points
    input are accepted in the following form:
    -as a strings: 'input_name=value,input_name=value,..' this supports only one point in the input space
    -as a dictionary where keys are the input names and the values the corresponding values (either one value or a vector/list)
    -as one of the admitted data for the specific ROM sub-type among the data type available in the datas.py module'''
    if  type(request)==str:#as a string
      inputNames  = [entry.split('=')[0]  for entry in request.split(',')]
      inputValues = [entry.split('=')[1]  for entry in request.split(',')]
    elif type(request)==dict:#as a dictionary
      inputNames, inputValues = request.items()
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        if input.type in self.ROM.admittedData:
          inputNames, inputValues = self.request.getInpParametersValues().items()
      except: raise IOError('the request of ROM evaluation is done via a not compatible data')
    #now that the prediction points are read we check the compatibility with the ROM input-output set
    lenght = len(set(inputNames).intersection(self.inputNames))
    if lenght!=len(self.inputNames) or lenght!=len(inputNames):
      raise IOError ('there is a mismatch between the provided request and the ROM structure')
    #once compatibility is assert we allocate the needed storage (numbers of samples)x(size input space)  
    self.request    = np.zeros(shape=(len(inputValues[1],self.inputNames)))
    i = 0
    for name in self.inputNames:
      self.request[:][i] = inputValues[inputNames.index(name)]
      i +=1
#    jobHandler.submitDict['Internal'](self.SupervisedEngine.train(self.request,self.outputValues))
    raise IOError('the multi treading is not yet in place neither the appanding')
    
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



def returnInstance(Type):
  '''This function return an instance of the request model type'''
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM'      ] = ROM
  InterfaceDict['Code'     ] = Code
  InterfaceDict['Filter'   ] = Filter
  InterfaceDict['Projector'] = Projector
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
