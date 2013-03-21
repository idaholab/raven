'''
Created on Feb 19, 2013

@author: crisr
'''
import os
import shutil
import Datas
from BaseType import BaseType

class RavenInterface:
  def generateCommand(self,inputFiles,executable):
    if inputFiles[0].endswith('.i'):
      outputfile = 'outFrom'+os.path.split(inputFiles[0])[1].split('.')[0]
      executeCommand = (executable+' -i '+os.path.split(inputFiles[0])[1]+' Output/postprocessor_csv=true' + 
      ' Output/file_base='+ outputfile)
    else: 
      outputfile = 'outFrom'+os.path.split(inputFiles[1])[1].split('.')[0]
      executeCommand = (executable+' -i '+os.path.split(inputFiles[1])[1]+' Output/postprocessor_csv=true' + 
      ' Output/file_base='+ outputfile)
    return executeCommand, outputfile

def returnCodeInterface(Type):
  base = 'Code'
  codeInterfaceDict = {}
  codeInterfaceDict['RAVEN'] = RavenInterface
  try: return codeInterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)

class Model(BaseType):
  ''' 
      a model is something that given an input will return an output reproducing some phisical model
      it could as complex as a stand alone code or a reduced order model trained somehow
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.subType = ''
  def readMoreXML(self,xmlNode):
    try: self.subType = xmlNode.attrib['type']
    except: raise 'missed type for the model'+self.name
  def addInitParams(self,tempDict):
    tempDict['subType'] = self.subType
  
  def reSet(self,stepName):
    print('Model'+self.name+' has been reset by '+stepName)
    return

  def train(self,trainingSet,stepName):
    print('Step '+stepName+' tried to train the model '+self.name+' that has no training step' )
    return

  def run(self):
    #
    #collect data
    return
  def collectFrom(self,collectFrom,storeTo):
    print('collecting from')
    print(collectFrom)
    print('storing to')
    print(storeTo)


class Code(Model):
  def __init__(self):
    Model.__init__(self)
    self.executable = ''
  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    try: self.executable = os.path.abspath(xmlNode.text)
    except: raise IOError('not found executable '+xmlNode.text)
    self.interface = returnCodeInterface(self.subType)
    print(type(self.interface))
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
  def setUpWorkingDir(self,runInfoDict,inputFiles):
    '''generate and fill a new working directory'''
    #start checking the existence and/or creating the working directory
    runInfoDict['TempWorkingDir'] = os.path.join(runInfoDict['WorkingDir'],runInfoDict['stepName'])
    print(runInfoDict['TempWorkingDir'])
    try: os.mkdir(runInfoDict['TempWorkingDir'])
    except: pass
    for inputFile in inputFiles:
      shutil.copy(inputFile,runInfoDict['TempWorkingDir'])
      inputFile
  def run(self,inputFiles,outputDatas,jobHandler):
    self.setUpWorkingDir(jobHandler.runInfoDict, inputFiles)
    executeCommand, outputfile = self.interface.generateCommand(inputFiles,self.executable)
    print(executeCommand)
    self.process = jobHandler.submitDict['External'](executeCommand,outputDatas,outputfile,jobHandler.runInfoDict['TempWorkingDir'])
    return self.process
    
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
  def reSet(self):
    pass




#function used to generate a Model class
def returnInstance(Type):
  base = 'model'
  InterfaceDict = {}
  InterfaceDict['ROM' ] = ROM
  InterfaceDict['Code'] = Code
  InterfaceDict['Code'] = Code
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  

  
  
  
  
  