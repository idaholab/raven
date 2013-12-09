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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Datas
from BaseType import BaseType
import SupervisedLearning
from Filters import returnFilterInterface
#Internal Modules End--------------------------------------------------------------------------------

class Model(metaclass_insert(abc.ABCMeta,BaseType)):
  ''' a model is something that given an input will return an output reproducing some physical model
      it could as complex as a stand alone code or a reduced order model trained somehow'''
  def __init__(self):
    BaseType.__init__(self)
    self.subType  = ''
    self.runQueue = []

  def readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['subType']
    except KeyError: 
      print("Failed in Node: ",xmlNode)
      raise Exception('missed subType for the model '+self.name)
  
  def localInputAndChecks(self,xmlNode):
    '''place here the additional reading, remember to add initial parameters in the method localAddInitParams'''

  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType

  def localAddInitParams(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''

  def initialize(self,runInfo,inputs):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
    after this call the next one will be run
    @in runInfo is the run info from the jobHandler
    @in inputs is a list containing whatever is passed with an input role in the step'''
    pass

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''this function have to return a new input that will be submitted to the model, it is called by the sampler
    @in myInput the inputs (list) to start from to generate the new one
    @in samplerType is the type of sampler that is calling to generate a new input
    @in **Kwargs is a dictionary that contains the information coming from the sampler,
         a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
    @return the new input (list)'''
    pass
  
  @abc.abstractmethod
  def run(self,Input,jobHandler):
    '''This call should be over loaded and should not return any results,
        possible it places a run one of the jobhadler lists!!!
        @in inputs is a list containing whatever is passed with an input role in the step
        @in jobHandler an instance of jobhandler that might be possible used to append a job for parallel running'''
    pass
  
  def collectOutput(self,collectFrom,storeTo):
    '''This call collect the output of the run
    @in collectFrom where the output is located, the form and the type is model dependent but should be compatible with the storeTo.addOutput method'''
    if 'addOutput' in dir(storeTo):
      storeTo.addOutput(collectFrom)
      #except? raise IOError('The place where to store the output '+type(storeTo)+' was not compatible with the addOutput of '+type(collectFrom))
    else: raise IOError('The place where to store the output has not a addOutput method')
#
#
#
class Dummy(Model):
  '''
  this is a dummy model that just return the input in the data
  it suppose to get a TimePoint or TimePointSet as input and also a TimePoint or TimePointSet or HDF5 as output
  The input is changed following the sampler info and than reported in the output
  '''
  def initialize(self,runInfo,inputs):
    self.counterInput =0
    self.counterOutput=0
    self.inputDict    ={}
    self.outputDict   ={}
    self.admittedData = []
    self.admittedData.append('TimePoint')
    self.admittedData.append('TimePointSet')

  def __returnAdmittedData(self):
    return self.admittedData
    
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''
    here only TimePoint and TimePointSet are accepted a local copy of the values is performed.
    For a TimePoint all value are copied, for a TimePointSet only the last set of entry
    The copied values are returned as a dictionary back
    '''
    inputDict    = {}
    outputDict   = {}
    if myInput[0].type not in self.__returnAdmittedData(): raise IOError('MODEL DUMMY  : ERROR -> The Dummy Model accepts only '+str(self.__returnAdmittedData())+' as input only!!!!')
    for key in myInput[0].getInpParametersValues().keys()  : inputDict[key] = copy.deepcopy(myInput[0].getInpParametersValues()[key][-1])
    if len(myInput[0].getOutParametersValues().keys())!=0:
      for key in myInput[0].getOutParametersValues().keys(): outputDict[key] = copy.deepcopy(myInput[0].getOutParametersValues()[key][-1])
    else:
      for key in myInput[0].dataParameters['outParam']: outputDict[key] = self.counterInput
    for key in Kwargs['SampledVars'].keys():
      inputDict[key] = copy.deepcopy(Kwargs['SampledVars'][key])
    self.counterInput+=1
    print('returning input')
    return [(inputDict,outputDict)]

  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    if 'print' in xmlNode.attrib.keys(): self.printFile = bool(xmlNode.attrib['print'])
    else: self.printFile = False
  
  def run(self,Input,jobHandler):
    '''The input should be under the form of a tuple of dictionary. The dictionary are copied and ready to be sent to the output'''
    self.inputDict  = copy.deepcopy(Input[0][0])
    self.outputDict = copy.deepcopy(Input[0][1])
    print('running')
  
  def collectOutput(self,finisishedjob,output):
    '''the input and output are sent back by the output'''
    self.counterOutput += 1
    print('looking for output')
    if output.type not in self.__returnAdmittedData()+['HDF5']: raise IOError('MODEL DUMMY  : ERROR -> The Dummy Model accepts TimePoint, TimePointSet or HDF5 as output only!!!!')
    if   output.type == 'HDF5':
      exportDict = copy.deepcopy(self.outputDict)
      exportDict['input_space_params'] = copy.deepcopy(self.inputDict)
      output.addGroupDatas({'group':self.name+str(self.counterOutput)},exportDict,False)
    else:
      for key in self.inputDict.keys() : output.updateInputValue(key,self.inputDict[key])
      for key in self.outputDict.keys(): output.updateOutputValue(key,self.outputDict[key])
      if self.printFile:
        output.printCSV()
    print('collected output')
#
#
#
class ExternalModel(Model):
  ''' External model class: this model allows to interface with an external python module'''
  def __init__(self):
    Model.__init__(self)
    self.modelVariableValues = {}
    self.modelVariableType   = {}
    self.__availableVariableTypes = ['float','int','bool','numpy.ndarray']
    self.counter = 0

  def initialize(self,runInfo,inputs):
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
    # load the external module and point it to self.sim
    self.sim=__import__(self.ModuleToLoad)
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

  def run(self,Input,jobHandler):
    self.sim.run(self,Input,jobHandler)
    self.counter += 1
    
  def collectOutput(self,finisishedjob,output):
    if 'collectOutput' in dir(self.sim):
      self.sim.collectOutput(self,finisishedjob,output)
    self.__pointSolution()
    def typeMatch(var,var_type_str):
      type_var = type(var)
      return type_var.__name__ == var_type_str or \
        type_var.__module__+"."+type_var.__name__ == var_type_str
    if 'HDF5' in output.type:
      for key in self.modelVariableValues: 
        if not (typeMatch(self.modelVariableValues[key],self.modelVariableType[key])):
          raise RuntimeError('MODEL EXTERNAL: ERROR -> type of variable '+ key + ' is ' + str(type(self.modelVariableValues[key]))+' and mismatches with respect to the input ones (' + self.modelVariableType[key] +')!!!')
      output.addGroupDatas({'group':str(self.counter)},self.modelVariableValues)
    else:
      if output.type not in ['TimePoint','TimePointSet','History','Histories']: raise RuntimeError('MODEL EXTERNAL: ERROR -> output type ' + output.type + ' unknown')
      for inputName in output.dataParameters['inParam']:
        if not (typeMatch(self.modelVariableValues[inputName],self.modelVariableType[inputName])):
          raise RuntimeError('MODEL EXTERNAL: ERROR -> type of variable '+ inputName + ' is ' + str(type(self.modelVariableValues[inputName]))+' and mismatches with respect to the inputted one (' + self.modelVariableType[inputName] +')!!!')
        output.updateInputValue(inputName,self.modelVariableValues[inputName])
      for outName in output.dataParameters['outParam']:
        if not (typeMatch(self.modelVariableValues[outName],self.modelVariableType[outName])):
          raise RuntimeError('MODEL EXTERNAL: ERROR -> type of variable '+ outName + ' is ' + str(type(self.modelVariableValues[outName]))+' and mismatches with respect to the inputted one (' + self.modelVariableType[outName] +')!!!')
        output.updateOutputValue(outName,self.modelVariableValues[outName])
      output.printCSV()    
    
  def __pointSolution(self):
    for variable in self.modelVariableValues.keys(): exec('self.modelVariableValues[variable] = self.'+  variable)
#
#
#
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
    self.alias              = {}   #if alias are defined in the input it defines a mapping between the variable names in the framework and the one for the generation of the input
                                   #self.alias[framework variable name] = [input code name]

  def readMoreXML(self,xmlNode):
    '''extension of info to be read for the Code(model)
    !!!!generate also the code interface for the proper type of code!!!!'''
    import CodeInterfaces
    Model.readMoreXML(self, xmlNode)
    
    if 'executable' in xmlNode.attrib.keys(): self.executable = xmlNode.attrib['executable']
    else: 
      try: self.executable = str(xmlNode.text)
      except IOError: raise Exception ('not found the attribute executable in the definition of the code model '+str(self.name))
    for child in xmlNode:
      if child.tag=='alias':
        if 'variable' in child.attrib.keys(): self.alias[child.attrib['variable']] = child.text
        else: raise Exception ('not found the attribute variable in the definition of one of the alias for code model '+str(self.name))
      else: raise Exception ('unknown tag within the definition of the code model '+str(self.name))

    abspath = os.path.abspath(self.executable)
    if os.path.exists(abspath):
      self.executable = abspath
    else: print('not found executable '+xmlNode.text)
    self.interface = CodeInterfaces.returnCodeInterface(self.subType)
    
  def addInitParams(self,tempDict):
    '''extension of addInitParams for the Code(model)'''
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
    for key, value in self.alias.items():
      tempDict['The code variable '+str(value)+' it is filled using the framework variable '] = key
      
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
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'out~'+os.path.split(currentInput[index])[1].split('.')[0]
    Kwargs['alias']   = self.alias
    if 'raven' in self.executable.lower(): Kwargs['reportit'] = False
    else: Kwargs['reportit'] = True 
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
    #if output.type == "HDF5": self.__addDataBaseGroup(finisishedjob,output)
    try:self.__addDataBaseGroup(finisishedjob,output)
    except AttributeError: output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv")

  def __addDataBaseGroup(self,finisishedjob,database):
    # add a group into the database
    attributes={}
    attributes["input_file"] = self.currentInputFiles
    attributes["type"] = "csv"
    attributes["name"] = os.path.join(self.workingDir,finisishedjob.output+'.csv')
    if finisishedjob.identifier in self.infoForOut:
      infoForOut = self.infoForOut.pop(finisishedjob.identifier)
      for key in infoForOut: attributes[key] = infoForOut[key]
    database.addGroup(attributes,attributes)
#
#
#
class ROM(Model):
  '''ROM stands for Reduced Order Model. All the models here, first learn than predict the outcome'''
  def __init__(self):
    Model.__init__(self)
    self.initializzationOptionDict = {}
    self.inputNames = []
    self.outputName = ''
    self.admittedData = []
    self.admittedData.append('TimePoint')
    self.admittedData.append('TimePointSet')

  def __returnAdmittedData(self):
    return self.admittedData

  def readMoreXML(self,xmlNode):
    '''read the additional input needed and create an instance the underlying ROM'''
    Model.readMoreXML(self, xmlNode)
    if 'target_response_name' in xmlNode.attrib:    
      self.outputName = xmlNode.attrib['target_response_name']
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except ValueError:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except ValueError: self.initializzationOptionDict[child.tag] = child.text
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
    if loadFrom.type not in self.__returnAdmittedData(): raise IOError('type '+loadFrom.type+' is not compatible with the ROM '+self.name)
    else: self.toLoadFrom = loadFrom
  
  def close(self):
    '''remember to call this function to decouple the data owned by the ROM and the environment data after each training'''
    self.toLoadFrom = copy.copy(self.toLoadFrom)

  def train(self):
    '''Here we do the training of the ROM'''
    '''Fit the model according to the given training data.
    @in X : {array-like, sparse matrix}, shape = [n_samples, n_features] Training vector, where n_samples in the number of samples and n_features is the number of features.
    @in y : array-like, shape = [n_samples] Target vector relative to X class_weight : {dict, 'auto'}, optional Weights associated with classes. If not given, all classes
            are supposed to have weight one.'''
    self.inputNames, inputsValues  = self.toLoadFrom.getInpParametersValues().keys(), self.toLoadFrom.getInpParametersValues().values()
    if self.outputName in self.toLoadFrom.getOutParametersValues(): 
      outputValues = self.toLoadFrom.getOutParametersValues()[self.outputName]
    else: raise IOError('The output sought '+self.outputName+' is not in the training set')
    self.inputsValues = numpy.zeros(shape=(inputsValues[0].size,len(self.inputNames)))
    self.outputValues = numpy.zeros(shape=(inputsValues[0].size))
    for i in range(len(self.inputNames)):
      self.inputsValues[:,i] = inputsValues[i][:]
    self.outputValues[:] = outputValues[:]
    self.SupervisedEngine.train(self.inputsValues,self.outputValues)

  def createNewInput(self,currentInput,samplerType,**Kwargs):
    ''' This function creates a new input
        It is called from a sampler to get the implementation specific for this model
        it support string input
        dictionary input and datas input'''
    import itertools
    if len(currentInput)>1: raise IOError('ROM accepts only one input not a list of inputs')
    else: currentInput =currentInput[0]
    if  type(currentInput)==str:#one input point requested a as a string
      inputNames  = [component.split('=')[0] for component in currentInput.split(',')]
      inputValues = [component.split('=')[1] for component in currentInput.split(',')]
      for name, newValue in itertools.izip(Kwargs['SampledVars'].keys(),Kwargs['SampledVars'].values()): 
        inputValues[inputNames.index(name)] = newValue
      newInput = [inputNames[i]+'='+inputValues[i] for i in range(inputNames)]
      newInput = newInput.join(',')
    elif type(currentInput)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      for name, newValue in itertools.izip(Kwargs['SampledVars'].keys(),Kwargs['SampledVars'].values()): 
        currentInput[name] = newValue
      newInput = copy.deepcopy(currentInput)
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        if currentInput.type in self.__returnAdmittedData():
          newInput = Datas.returnInstance(currentInput.type)
          newInput.type = currentInput.type
          for name,value in itertools.izip(currentInput.getInpParametersValues().keys(),currentInput.getInpParametersValues().values()): newInput.updateInputValue(name,numpy.atleast_1d(numpy.array(value)))
          for name, newValue in itertools.izip(Kwargs['SampledVars'].keys(),Kwargs['SampledVars'].values()):
            # for now, even if the ROM accepts a TimePointSet, we create a TimePoint
            newInput.updateInputValue(name,numpy.atleast_1d(numpy.array(newValue)))
            #except? raise IOError('trying to sample '+name+' that is not in the original input')
      except AttributeError: raise IOError('the request of ROM evaluation is done via a not compatible input')
    currentInput = [newInput]
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
        if self.request.type in self.__returnAdmittedData():
          inputNames, inputValues = self.request.getInpParametersValues().keys(), self.request.getInpParametersValues().values()
      except AttributeError: raise IOError('the request of ROM evaluation is done via a not compatible data')
    #now that the prediction points are read we check the compatibility with the ROM input-output set
    lenght = len(set(inputNames).intersection(self.inputNames))
    if lenght!=len(self.inputNames) or lenght!=len(inputNames):
      raise IOError ('there is a mismatch between the provided request and the ROM structure')
    #build a mapping from the ordering of the input sent in and the ordering inside the ROM
    self.requestToLocalOrdering = []
    for local in self.inputNames:
      self.requestToLocalOrdering.append(inputNames.index(local))
    #building the arrays to send in for the prediction by the ROM
    self.request = numpy.array([inputValues[index] for index in self.requestToLocalOrdering]).T[0]
    ############################------FIXME----------#######################################################
    # we need to submit self.ROM.evaluate(self.request) to the job handler
    self.output = self.SupervisedEngine.evaluate(self.request)

  def collectOutput(self,finishedJob,output):
    '''This method append the ROM evaluation into the output'''
    try: #try is used to be sure input.type exist
      if output.type in self.__returnAdmittedData():
        for inputName in self.inputNames:
          output.updateInputValue(inputName,self.request[self.inputNames.index(inputName)])
    except AttributeError: raise IOError('the output of the ROM is requested on a not compatible data')
    output.updateOutputValue(self.outputName,self.output)
    output.printCSV()
#
#
#  
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

  def reset(self,runInfoDict,myInput):
    if myInput.type == 'ROM':
      pass
    #initialize some of the current setting for the runs and generate the working 
    #   directory with the starting input files
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName']) #generate current working dir
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except FileExistsError: print('MODEL FILTER  : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    self.interface.finalizeFilter(inObj,outObj,self.workingDir)
#
#
#
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
    except FileExistsError: print('MODEL FILTER  : warning current working dir '+self.workingDir+' already exists, this might imply deletion of present files')
    return

  def run(self,inObj,outObj):
    '''run calls the interface finalizer'''
    for i in range(len(inObj)):
      self.interface.finalizeFilter(inObj[i],outObj[i],self.workingDir)
  def collectOutput(self,finishedjob,output):
    self.interface.collectOutput(output)
  def createNewInput(self,myInput,samplerType,**Kwargs):
    '''just for compatibility'''
    pass

'''
 Interface Dictionary (factory) (private)
'''

__base = 'model'
__interFaceDict = {}
__interFaceDict['ROM'           ] = ROM
__interFaceDict['Code'          ] = Code
__interFaceDict['Filter'        ] = Filter
__interFaceDict['Projector'     ] = Projector
__interFaceDict['Dummy'         ] = Dummy
__interFaceDict['ExternalModel' ] = ExternalModel
__knownTypes                      = __interFaceDict.keys()


def knonwnTypes():
  return __knownTypes

def returnInstance(Type,debug=False):
  '''This function return an instance of the request model type'''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)
  
