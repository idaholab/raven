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
import DataObjects
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

  def reset(self,runInfo,inputs):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step'''
    raise IOError('the model '+self.name+' that has no reset method' )

  def train(self,trainingSet,stepName):
    '''This needs to be over written if the model requires an initialization'''
    raise IOError('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )

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

  def reset(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working
       directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)

    except: print('MODEL CODE    : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    for inputFile in inputFiles:
      shutil.copy(inputFile,self.workingDir)
    print('MODEL CODE    : original input files copied in the current working dir: '+self.workingDir)
    print('MODEL CODE    : files copied:')
    for f in inputFiles:print(f)
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

  def run(self,inputFiles,outputDataObjects,jobHandler):
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
        self.__addDatabaseGroup(finisishedjob,output)
        return
    except AttributeError:
      pass
    #print('this:',output)
    output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv")

    return

  def finalizeOutput(self,output):
    output.finalizeOut()

  def __addDatabaseGroup(self,finisishedjob,database):
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

  def readMoreXML(self,xmlNode):
    '''read the additional input needed and istanziate the underlying ROM'''
    Model.readMoreXML(self, xmlNode)
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except: self.initializzationOptionDict[child.tag] = child.text
    self.test =  SupervisedLearning.returnInstance(self.subType)
    self.SupervisedEngine = self.test(**self.initializzationOptionDict)
    #self.test.
    #self.SupervisedEngine = SupervisedLearning.returnInstance(self.subType)(self.initializzationOptionDict) #create an instance of the ROM

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

  def reset(self,*args):
    self.SupervisedEngine.reset(*args)

  def train(self,trainingSet=None):
    '''This needs to be over written if the model requires an initialization'''
    #raise IOError('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )
    #self.test.type
    if trainingSet:
      self.SupervisedEngine.train(trainingSet)
      try:
        if trainingSet['Input'][0].type == 'HDF5': self.outputName = trainingSet['Input'][0].targetParam
      except: pass
    else:
      self.SupervisedEngine.train(self.toLoadFrom)
    return

  def fillDistribution(self,distributions):
    self.distDict = distributions
    self.SupervisedEngine.fillDist(distributions)

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input
        It is called from a sampler to get the implementation specific for this model
        it support string input
        dictionary input and dataObjects input'''
    import itertools

    if len(currentInput)>1: raise IOError('ROM accepts only one input not a list of inputs')
    else: currentInput =currentInput[0]
    if  type(currentInput)==str:#one input point requested a as a string
      inputNames  = [component.split('=')[0] for component in currentInput.split(',')]
      inputValues = [component.split('=')[1] for component in currentInput.split(',')]
      for name, newValue in itertools.izip(Kwargs['sampledVars'].keys(),Kwargs['sampledVars'].values()):
        inputValues[inputNames.index(name)] = newValue
      newInput = [inputNames[i]+'='+inputValues[i] for i in range(inputNames)]
      newInput = newInput.join(',')
    elif type(currentInput)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      for name, newValue in itertools.izip(Kwargs['sampledVars'].keys(),Kwargs['sampledVars'].values()):
        currentInput[name] = newValue
      newInput = copy.deepcopy(currentInput)
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        if currentInput.type in ['TimePoint','TimePointSet']:
          newInput = DataObjects.returnInstance(currentInput.type)
          newInput.type = currentInput.type
          for name,value in itertools.izip(currentInput.getInpParametersValues().keys(),currentInput.getInpParametersValues().values()): newInput.updateInputValue(name,np.atleast_1d(np.array(value)))
          for name, newValue in itertools.izip(Kwargs['sampledVars'].keys(),Kwargs['sampledVars'].values()):
            # for now, even if the ROM accepts a TimePointSet, we create a TimePoint
            try   : newInput.updateInputValue(name,np.atleast_1d(np.array(newValue)))
            except: raise IOError('trying to sample '+name+' that is not in the original input')
      except: raise IOError('the request of ROM evaluation is done via a not compatible input')
    currentInput = [newInput]
    return currentInput

  def run(self,request,output,jobHandler):
    '''This call run a ROM as a model
    The input should be translated in to a set of coordinate where to perform the predictions
    It is possible either to send in just one point in the input space or a set of points
    input are accepted in the following form:
    -as a strings: 'input_name=value,input_name=value,..' this supports only one point in the input space
    -as a dictionary where keys are the input names and the values the corresponding values (it should be either vector or list)
    -as one of the admitted data for the specific ROM sub-type among the data type available in the dataObjects.py module'''
    if len(request)>1: raise IOError('ROM accepts only one input not a list of inputs')
    else: self.request =request[0]
    #first we extract the input names and the corresponding values (it is an implicit mapping)
    if  type(self.request)==str:#one input point requested a as a string
      inputNames  = [entry.split('=')[0]  for entry in self.request.split(',')]
      inputValues = [entry.split('=')[1]  for entry in self.request.split(',')]
    elif type(self.request)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      inputNames, inputValues = self.request.keys(), self.request.values()
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        print(self.request.type)
        if self.request.type in ['TimePoint','TimePointSet']:
          inputNames, inputValues = self.request.getInpParametersValues().keys(), self.request.getInpParametersValues().values()
          self.inputNames = inputNames
      except: raise IOError('the request of ROM evaluation is done via a not compatible data')
    #now that the prediction points are read we check the compatibility with the ROM input-output set
    #lenght = len(set(inputNames).intersection(self.inputNames))
    #if lenght!=len(self.inputNames) or lenght!=len(inputNames):
    #  raise IOError ('there is a mismatch between the provided request and the ROM structure')
    #build a mapping from the ordering of the input sent in and the ordering inside the ROM
    #self.requestToLocalOrdering = []
    #for local in self.inputNames:
    #  self.requestToLocalOrdering.append(inputNames.index(local))
    #building the arrays to send in for the prediction by the ROM
    #self.request = np.array([inputValues[index] for index in self.requestToLocalOrdering]).T[0]
    ############################------FIXME----------#######################################################
    # we need to submit self.ROM.evaluate(self.request) to the job handler
    self.output = self.SupervisedEngine.evaluate(self.request)
#    raise IOError('the multi treading is not yet in place neither the appending')

  def collectOutput(self,finishedJob,output):
    '''This method append the ROM evaluation into the output'''
    try: #try is used to be sure input.type exist
      if output.type in ['TimePoint','TimePointSet']:
        print('IN COLLECT OUTPUT ROM!!!!!!!! for output ' + str(output.name))
        for inputName in self.inputNames:
          if type(self.request) == 'numpy.ndarray':
            output.updateInputValue(inputName,self.request[self.inputNames.index(inputName)])
          else:
            test = self.request.getInpParametersValues().keys()
            found = False
            for i in range(len(test)):
              if inputName in test[i]:
                nametouse = test[i]
                found = True
            if not found:
              raise IOError( 'inputName ' + inputName + 'not contained into data used in training of the ROM')
            output.updateInputValue(inputName,self.request.getInpParametersValues()[nametouse])
    except: raise IOError('the output of the ROM is requested on a not compatible data')
    output.updateOutputValue(self.outputName,self.output)
    print(str(self.output))


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

class ExternalModel(Model):
  ''' External model class: this model allows to interface with an external python module'''
  def __init__(self):
    Model.__init__(self)
    self.modelVariableValues = {}
    self.modelVariableType   = {}
    self.__availableVariableTypes = ['float','int','bool','numpy.ndarray']
  def reset(self,runInfo,inputs):
    self.counter=0
    if 'initialize' in dir(self.sim):
      self.sim.initialize(self,runInfo,inputs)

  def createNewInput(self,myInput,samplerType,**Kwargs):
    if 'createNewInput' in dir(self.sim):
      newInput = self.sim.createNewInput(self,myInput,samplerType,**Kwargs)
      return [newInput]
    else:
      return [None]

  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    if 'ModuleToLoad' in xmlNode.attrib.keys():
      self.ModuleToLoad = os.path.split(str(xmlNode.attrib['ModuleToLoad']))[1]
      if (os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0] != ''):
        abspath = os.path.abspath(os.path.split(str(xmlNode.attrib['ModuleToLoad']))[0])
        if os.path.exists(abspath): os.sys.path.append(abspath)
        else: raise IOError('MODEL EXTERNAL: ERROR -> The path provided for the external model does not exist!!! Got ' + abspath)
    else: raise IOError('MODEL EXTERNAL: ERROR -> ModuleToLoad not provided for module externalModule')
    exec('import ' + self.ModuleToLoad + ' as sim')
    # point to the external module
    self.sim=sim
    # check if there are variables and, in case, load them
    for son in xmlNode:
      if son.tag=='variable':
        self.modelVariableValues[son.text] = None
        exec('self.'+son.text+' = self.modelVariableValues['+'son.text'+']')
        if 'type' in son.attrib.keys():
          if not (son.attrib['type'].lower() in self.__availableVariableTypes):
            raise IOError('MODEL EXTERNAL: ERROR -> the "type" of variable ' + son.text + 'not')
          self.modelVariableType[son.text] = son.attrib['type']
        else                          :
          raise IOError('MODEL EXTERNAL: ERROR -> the attribute "type" for variable '+son.text+' is missed')
    # check if there are other information that the external module wants to load
    if 'readMoreXML' in dir(self.sim):
      self.sim.readMoreXML(self,xmlNode)


  def run(self,Input,output,jobHandler):
    self.sim.run(self,Input,jobHandler)

  def collectOutput(self,finisishedjob,output):
    if 'collectOutput' in dir(self.sim):
      self.sim.collectOutput(self,finisishedjob,output)
    self.__pointSolution()
    if 'HDF5' in output.type: raise NotImplementedError('MODEL EXTERNAL: ERROR -> output type HDF5 not implemented yet for externalModel')

    if output.type not in ['TimePoint','TimePointSet','History','Histories']: raise RuntimeError('MODEL EXTERNAL: ERROR -> output type ' + output.type + ' unknown')
    for inputName in output.dataParameters['inParam']:
      exec('if not (type(self.modelVariableValues[inputName]) == ' + self.modelVariableType[inputName] + '):raise RuntimeError("MODEL EXTERNAL: ERROR -> type of variable '+ inputName + ' mismatches with respect to the inputted one!!!")')
      output.updateInputValue(inputName,self.modelVariableValues[inputName])
    for outName in output.dataParameters['outParam']:
      output.updateOutputValue(outName,self.modelVariableValues[outName])

  def __pointSolution(self):
    for variable in self.modelVariableValues.keys(): exec('self.modelVariableValues[variable] = self.'+  variable)

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM'   ] = ROM
  InterfaceDict['Code'  ] = Code
  InterfaceDict['Filter'] = Filter
  InterfaceDict['ExternalModel'] = ExternalModel
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
