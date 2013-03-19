'''
Created on Feb 19, 2013

@author: crisr
'''
import os
import Datas
from BaseType import BaseType



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


class Code(Model):
  def __init__(self):
    Model.__init__(self)
    self.executable = ''
  def readMoreXML(self,xmlNode):
    Model.readMoreXML(self, xmlNode)
    try: self.executable = os.path.abspath(xmlNode.text)
    except: raise IOError('not found executable '+xmlNode.text)
  def addInitParams(self,tempDict):
    Model.addInitParams(self, tempDict)
    tempDict['executable']=self.executable
  def run(self,inputFile,outputData,jobHandler):
    print(outputData)
    self.outputfile = os.path.join(jobHandler.runInfoDict['tempWorkingDir'],outputData.name)
    executeCommand = self.executable+' -i '+inputFile+' -o '+ self.outputfile
    self.process = jobHandler.submitDict['External'](executeCommand,outputData,self.outputfile,jobHandler.runInfoDict['tempWorkingDir'])
    #take out the job from the queue only when data are recovered
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
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  

  
  
  
  
  