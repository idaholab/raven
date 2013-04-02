'''
Created on Mar 8, 2013

@author: crisr
'''
import time
import Datas
from BaseType import BaseType

class Sampler(BaseType):
  ''' 
  this is the base class for samplers
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.limit       = 0        #maximum number of sampler it will perform every time it is used
    self.counter     = 0
    self.toBeSampled = {}  #key=feature to be sampled, value = ['type of distribution to be used', 'name of the distribution']
    self.distDict    = {}  #contain the instance of the distribution to be used, it is created every time the sampler is initialize
  def readMoreXML(self,xmlNode):
    try: self.limit = xmlNode.attrib['limit']
    except: raise IOError('not found limit for the Sampler '+self.name)
    for child in xmlNode:
      self.toBeSampled[child.text] = [child.attrib['type'],child.attrib['distName']]
  
  def addInitParams(self,tempDict):
    tempDict['limit' ] = self.limit
    for value in self.toBeSampled.items():
      tempDict[value[0]] = value[1][0]+':'+value[1][1]
  
  def addCurrentSetting(self,tempDict):
    tempDict['counter' ] = self.counter
  def initialize(self):
    self.counter = 0
  
  def fillDistribution(self,availableDist):
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]].inDistr()
    return

  def generateInputBatch(self,myInput,model,batchSize):
    if batchSize<=self.limit:newInputs = [None]*batchSize
    else:newInputs = [None]*self.limit
    for i in range(len(newInputs)):
      newInputs[i]=self.generateInput(model,myInput)
    return newInputs


class MonteCarlo(Sampler):
  def generateInput(self,model,myInput):
    self.counter += 1
    values = {'counter':self.counter}
    for key in self.distDict:
      values[key] = self.distDict[key].distribution.rvs()
    return model.createNewInput(myInput,self.type,**values)

class LatinHyperCube(Sampler):
  pass

class EquallySpaced(Sampler):
  pass

#function used to generate a Model class
def returnInstance(Type):
  base = 'Sampler'
  InterfaceDict = {}
  InterfaceDict['MonteCarlo'    ] = MonteCarlo
  InterfaceDict['LatinHyperCube'] = LatinHyperCube
  InterfaceDict['EquallySpaced' ] = EquallySpaced
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  
  
  