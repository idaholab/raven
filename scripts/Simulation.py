'''
Created on Feb 19, 2013

@author: crisr
'''
import Steps
import Datas
import Samplers
import Models
import Tests
import RunInfo



class Simulation:
  '''
  This is a class that contain all the object needed to run the simulation
  '''
  def __init__(self):
    self.stepsDict    = {}
    self.dataDict     = {}
    self.samplersDict = {}
    self.modelsDict   = {}
    self.testsDict    = {}
    self.runInfoDict  = {}
    self.addWhatDict  = {}
    self.addWhatDict['Steps'   ] = self.addSteps
    self.addWhatDict['Datas'   ] = self.addSDatas
    self.addWhatDict['Samplers'] = self.addSamplers
    self.addWhatDict['Models'  ] = self.addModels
    self.addWhatDict['Tests'   ] = self.addTests
    self.addWhatDict['RunInfo' ] = self.addRunInfo
    
    
  def add(self,xmlNode):
    '''
    add to the dictionaries the instances of all type needed
    '''
    try:
      self.addWhatDict[xmlNode.tag](xmlNode)
    except:
        raise IOError('the '+xmlNode.tag+'is not among the known simulation components')

  def addSteps(self,xmlNode):
    '''
    generate the Step objects 
    '''
    for child in xmlNode:
      try:
        self.stepsDict[child.attr['name']] = Steps.returnStepClass(child.attr['type'],child)
      except:
        raise IOError('not found name or type attribute for one of the Steps')
  
  def addDatas(self,xmlNode):
    '''
    generate the Data objects
    '''
    for child in xmlNode:
      try:
        self.dataDict[child.attr['name']] = Datas.returnDataClass(child)
      except:
        raise IOError('not found name attribute for one of the Data')

  def addSamplers(self,xmlNode):
    '''
    generate the Samplers objects
    '''
    for child in xmlNode:
      try:
        self.samplersDict[child.attr['name']] = Samplers.returnSamplerClass(child)
      except:
        raise IOError('not found name attribute for one of the Sampler')
    
  def addModels(self,xmlNode):
    '''
    generate the instance of the models 
    '''
    for child in xmlNode:
      try:
        self.modelsDict[child.attr['name']] = Models.returnModelClass(child)
      except:
        raise IOError('not found name attribute for one of the models')

  def addTests(self,xmlNode):
    '''
    generate the instance of the Tests 
    '''
    for child in xmlNode:
      try:
        self.testsDict[child.attr['name']] = Tests.returnTestClass(child)
      except:
        raise IOError('not found name attribute for one of the Tests')







