'''
Created on Feb 19, 2013

@author: crisr
'''
import Samplers
import Models
import Datas


class Simulation:
  '''
  This is a class that contain all the istanciated classes for the simulation 
  '''
  def __init__(self):
    self.modelsDict  = {}
    self.datasDict   = {}
    self.samplerDict = {}
    self.addWhatDict = {}
    self.addWhatDict['model']   = self.addModel
    self.addWhatDict['sampler'] = self.addSampler
    self.addWhatDict['data']    = self.addData
    
    
  def addModel(self,xmlNode):
    '''
    generate the instance of the models 
    '''
    for child in xmlNode:
      try:
        self.samplerDict[child.attr['name']] = Models.returnModelClass(child.attr['type'],child)
      except:
        raise IOError('not found name attribute for one of the models')
   
  
  def addData(self,xmlNode):
    '''
    generate the instance of the data 
    '''
    for child in xmlNode:
      try:
        self.samplerDict[child.attr['name']] = Datas.returnDataClass(child.attr['type'],child)
      except:
        raise IOError('not found name attribute for one of the Data')
   
  def addSampler(self,xmlNode):
    '''
    generate the instance of the models 
    '''
    for child in xmlNode:
      try:
        self.samplerDict[child.attr['name']] = returnSamplerClass(child.attr['type'],child)
      except:
        raise IOError('not found name attribute for one of the Sampler')
    
  def add(self,xmlNode):
    '''
    generate the instance of all type needed
    '''
    try:
      self.addWhatDict[xmlNode.tag](xmlNode)
    except:
        raise IOError('the '+xmlNode.tag+'is not among the known simulation components')

    
