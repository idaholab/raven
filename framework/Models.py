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
import SupervisionedLearning
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

  def reset(self,runInfo,inputs):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step'''
    raise IOError('the model '+self.name+' that has no reset method' )

  def train(self,trainingSet,stepName):
    '''This needs to be over written if the model requires an initialization'''
    raise IOError('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )

  def run(self):
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

  def reset(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working 
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
                   
    except: print('MODEL CODE    : warning current working dir '+self.workingDir+'already exists, this might imply deletion of present files')
    for inputFile in inputFiles:
      shutil.copy(inputFile,self.workingDir)
    print('MODEL CODE    : original input files copied in the current working dir: '+self.workingDir)
    print('MODEL CODE    : files copied:')
    print(inputFiles)
    self.oriInputFiles = []
    for i in range(len(inputFiles)):
      self.oriInputFiles.append(os.path.join(self.workingDir,os.path.split(inputFiles[i])[1]))
    self.currentInputFiles        = None
    self.outFileRoot              = None
    return #self.oriInputFiles

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input '''
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'outFrom~'+os.path.split(currentInput[index])[1].split('.')[0]
    self.infoForOut[Kwargs['prefix']] = copy.deepcopy(Kwargs)
    return self.interface.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs)

  def run(self,inputFiles,outputDatas,jobHandler):
    '''return an instance of external runner'''
    self.currentInputFiles = inputFiles
    executeCommand, self.outFileRoot = self.interface.generateCommand(self.currentInputFiles,self.executable)
#    for inputFile in self.currentInputFiles: shutil.copy(inputFile,self.workingDir)
    self.process = jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'])
    #XXX what does this if block do?  Should it be a for loop and look thru the array?
    if self.currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    print('MODEL CODE    : job "'+ inputFiles[index].split('/')[-1].split('.')[-2] +'" submitted!')
    return self.process

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
  '''ROM stands for Reduced Order Models. All the models here, first learn than predict the outcome'''
  def __init__(self):
    Model.__init__(self)
    self.initializzationOptionDict = {}

  def readMoreXML(self,xmlNode):
    '''read the additional input needed and istanziate the underlying ROM'''
    Model.readMoreXML(self, xmlNode)
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except: self.initializzationOptionDict[child.tag] = child.text
    self.test =  SupervisionedLearning.returnInstance(self.subType)
    self.SupervisedEngine = self.test(**self.initializzationOptionDict)
    #self.test.
    #self.SupervisedEngine = SupervisionedLearning.returnInstance(self.subType)(self.initializzationOptionDict) #create an instance of the ROM
  
  def addLoadingSource(self,loadFrom):
    self.toLoadFrom = loadFrom
  
  def addInitParams(self,originalDict):
    ROMdict = self.SupervisedEngine.returnInitialParamters()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]

  def addCurrentSetting(self,originalDict):
    ROMdict = self.SupervisedEngine.returnCurrentSetting()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]

  def reset(self):
    self.SupervisedEngine.reset()

  def train(self,trainingSet=None):
    '''This needs to be over written if the model requires an initialization'''
    #raise IOError('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )
    print('we are in train ROM')
    #self.test.type
    if trainingSet:
      self.SupervisedEngine.train(trainingSet)
    else:
      self.SupervisedEngine.train(self.toLoadFrom)
    return
#  def run(self):
#    return
#  def collectOutput(self,collectFrom,storeTo):
#    storeTo.addOutput(collectFrom)
#  def createNewInput(self,myInput,samplerType,**Kwargs):
#    raise IOError('for this model the createNewInput has not yet being implemented')
#  TODO how is ROM tied to Supervisioned Learning?  "train" method in Model isn't overwritten...

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
    self.interface.finalizeFilter(inObj,outObj,self.workingDir)

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM'   ] = ROM
  InterfaceDict['Code'  ] = Code
  InterfaceDict['Filter'] = Filter
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
