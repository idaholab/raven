'''
Created on Feb 19, 2013

@author: crisr
'''
import os
import shutil
import Datas
from BaseType import BaseType

class RavenInterface:
  '''this class is used a part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable):
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'outFrom~'+os.path.split(inputFiles[index])[1].split('.')[0]
    executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1]+' Output/postprocessor_csv=true' + 
    ' Output/file_base='+ outputfile)
    return executeCommand,outputfile
  
  def appendLoadFileExtension(self,fileRoot):
    return fileRoot + '.csv'
  
  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    import MOOSEparser
    self.samplersDictionary = {}
    self.samplersDictionary['MonteCarlo']     = self.MonteCarloForRAVEN
    self.samplersDictionary['EquallySpaced']  = self.EquallySpacedForRAVEN
    self.samplersDictionary['LatinHyperCube'] = self.LatinHyperCubeForRAVEN
    self.samplersDictionary['DynamicEventTree'] = self.DynamicEventTreeForRAVEN
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index])
    modifDict = self.samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,False)
    temp = str(oriInputFiles[index][:])
    currentInputFiles[index] = os.path.join(os.path.split(temp)[0],Kwargs['prefix']+os.path.split(temp)[1])
    parser.printInput(currentInputFiles[index])
    return currentInputFiles

  def MonteCarloForRAVEN(self,**Kwargs):
    try: counter = Kwargs['counter']
    except: raise IOError('a counter is needed for the Monte Carlo sampler for RAVEN')
    listDict = []
    modifDict = {}
    modifDict['name'] = ['Distributions']
    modifDict['RNG_seed'] = counter
    listDict.append(modifDict)
    return listDict
  
  def DynamicEventTreeForRAVEN(self,**Kwargs):
    listDict = []
    
    for i in xrange(len(Kwargs['initiator_distribution'])):
      modifDict = {}
      modifDict['name'] = ['Distributions',Kwargs['initiator_distribution'][i]]
      modifDict['ProbabilityThreshold'] = Kwargs['PbThreshold'][i]
      listDict.append(modifDict)
      del modifDict
    if 'start_time' in Kwargs.keys():
      modifDict = {}
      st_time = Kwargs['start_time']
      modifDict['name'] = ['Executioner']
      modifDict['start_time'] = st_time
      listDict.append(modifDict)
      del modifDict
      
    if 'end_ts' in Kwargs.keys():
      modifDict = {}
      end_ts_str = str(Kwargs['end_ts'])
      if(self.endInfo['end_ts'] <= 9999):
        n_zeros = 4 - len(end_ts_str)
        for i in xrange(len(n_zeros)-1):
          end_ts_str = "0" + end_ts_str
      restart_file_base = Kwargs['outfile'] + "_restart_" + end_ts_str      
      modifDict['name'] = ['Executioner']
      modifDict['restart_file_base'] = restart_file_base
      listDict.append(modifDict)
      del modifDict

    if 'end_time' in Kwargs.keys():
      modifDict = {}
      end_time = Kwargs['end_time']
      modifDict['name'] = ['Executioner']
      modifDict['end_time'] = end_time
      listDict.append(modifDict)
      del modifDict
      
    modifDict = {}
    modifDict['name'] = ['Output']
    modifDict['num_restart_files'] = 1
    listDict.append(modifDict)
    del modifDict
    
    if 'branch_changed_param' in Kwargs.keys():
      for i in xrange(len(Kwargs['branch_changed_param'])):
        modifDict = {}
        modifDict['name'] = ['RestartInitialize',Kwargs['branch_changed_param'][i]]
        modifDict['value'] = Kwargs['branch_changed_param_value'][i]
        listDict.append(modifDict)
        del modifDict

    return listDict  
  
  def EquallySpacedForRAVEN(self,**Kwargs):
    raise IOError('EquallySpacedForRAVEN not yet implemented')
    listDict = []
    return listDict
  
  def LatinHyperCubeForRAVEN(self,**Kwargs):
    raise IOError('LatinHyperCubeForRAVEN not yet implemented')
    listDict = []
    return listDict


class ExternalTest:
  def generateCommand(self,inputFiles,executable):
    return '', ''
  def findOutputFile(self,command):
    return ''
  
def returnCodeInterface(Type):
  base = 'Code'
  codeInterfaceDict = {}
  codeInterfaceDict['RAVEN'] = RavenInterface
  codeInterfaceDict['ExternalTest'] = ExternalTest
  try: return codeInterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)

class Model(BaseType):
  ''' a model is something that given an input will return an output reproducing some phisical model
      it could as complex as a stand alone code or a reduced order model trained somehow'''
  def __init__(self):
    BaseType.__init__(self)
    self.subType = ''
  def readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['type']
    except: raise 'missed type for the model'+self.name
  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType
  def reset(self):
    ''' this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step'''
  def train(self,trainingSet,stepName):
    '''This needs to be over loaded if the model requires an initialization'''
    print('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )
    return
  def run(self):
    '''This call should be over loaded and return a jobHandler.External/InternalRunner'''
    return
  def collectOutput(self,collectFrom,storeTo):
    storeTo.addOutput(collectFrom)
  def collectOutput(self,collectFrom,obj,storeTo):
    storeTo.addOutput(collectFrom)  
  def createNewInput(self,myInput,samplerType,**Kwargs):
    raise IOError('for this model the createNewInput has not yet being implemented')
  def addDataSetGroup(self):
    pass
class Code(Model):
  def __init__(self):
    Model.__init__(self)
    self.executable         = ''
    self.oriInputFiles      = []
    self.workingDir         = ''
    self.outFileRoot        = ''
    self.currentInputFiles  = []
  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    try: self.executable = os.path.abspath(xmlNode.text)
    except: raise IOError('not found executable '+xmlNode.text)
    self.interface = returnCodeInterface(self.subType)
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
  def addCurrentSetting(self,originalDict):
    originalDict['current working directory'] = self.workingDir
    originalDict['current output file root']  = self.outFileRoot
    originalDict['current input file']        = self.currentInputFiles
    originalDict['original input file']       = self.oriInputFiles
  def reset(self,runInfoDict,inputFiles):
    '''initialize some of the current setting for the runs and generate the working directory with the starting input files'''
    self.workingDir               = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName'])
    runInfoDict['TempWorkingDir'] = self.workingDir
    try: os.mkdir(self.workingDir)
    except: pass
    for inputFile in inputFiles:
      shutil.copy(inputFile,self.workingDir)
    self.oriInputFiles = []
    for i in range(len(inputFiles)):
      self.oriInputFiles.append(os.path.join(self.workingDir,os.path.split(inputFiles[i])[1]))
    self.currentInputFiles        = None
    self.outFileRoot              = None
    return #self.oriInputFiles
  def createNewInput(self,currentInput,samplerType,**Kwargs):
    if currentInput[0].endswith('.i'): index = 0
    else: index = 1
    Kwargs['outfile'] = 'outFrom~'+os.path.split(currentInput[index])[1].split('.')[0]
    return self.interface.createNewInput(currentInput,self.oriInputFiles,samplerType,**Kwargs)
  def run(self,inputFiles,outputDatas,jobHandler):
    '''return an instance of external runner'''
    self.currentInputFiles = inputFiles
    executeCommand, self.outFileRoot = self.interface.generateCommand(self.currentInputFiles,self.executable)
#    for inputFile in self.currentInputFiles: shutil.copy(inputFile,self.workingDir)
    self.process = jobHandler.submitDict['External'](executeCommand,self.outFileRoot,jobHandler.runInfoDict['TempWorkingDir'])
    print('job submitted')
    return self.process

  def collectOutput(self,finisishedjob,output):
    '''collect the output file in the output object'''
    if output.type == "HDF5":
      self.__addDataSetGroup(finisishedjob,output)
      
    else:
      output.addOutput(os.path.join(self.workingDir,finisishedjob.output) + ".csv")
    return
  
  def __addDataSetGroup(self,finisishedjob,dataset):
    # add a group into the database
    attributes={}
    attributes["input_file"] = self.currentInputFiles
    attributes["type"] = "csv"
    attributes["name"] = os.path.join(self.workingDir,finisishedjob.output+'.csv')
    dataset.addGroup(attributes,attributes)
      
class ROM(Model):
  '''
  ROM stands for Reduced Order Models. All the models here, first learn than predict the outcome
  '''
  def __init__(self):
    Model.__init__(self)
    self.initializzationOptionDict = {}
  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    import scikitLearnInterface
    for child in xmlNode:
      try: self.initializzationOptionDict[child.tag] = int(child.text)
      except:
        try: self.initializzationOptionDict[child.tag] = float(child.text)
        except: self.initializzationOptionDict[child.tag] = child.text
    self.importedROM = scikitLearnInterface.classDictionary[self.subType](**self.initializzationOptionDict) #create an instance of the ROM
  def addCurrentSetting(self,originalDict):
    ROMdict = self.importedROM.get_params()
    for key in ROMdict.keys():
      originalDict[key] = ROMdict[key]
  def reset(self):
    pass




#function used to generate a Model class
def returnInstance(Type):
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM' ] = ROM
  InterfaceDict['Code'] = Code
  #InterfaceDict['Code'] = Code
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  
  
  
  
  
  