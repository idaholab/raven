'''
Created on Feb 19, 2013

@author: crisr
'''
import subprocess, sys, os
import Datas

class Model:
  def __init__(self,xmlNode):
    '''
    Constructor for the template class
    '''
    self.counter +=1
    self.paramters = self.getparamters()
    self.readXml()

  def iniCounter(self):
    self.counter = 0
 
  def getparamters(self):
    return None

  def readXml(self,xmlNode):
    

  def evaluate(self):
    self.counter +=1
    
  
  def generateData(self,sampler,data):
    self.sampler.generaInput()
    self.evaluate(self.args, self.stdout)
    data.load()




class RAVEN(Model):
  '''
  This is a model that use RAVEN to perform in-->out
  '''
  def __init__(self,xmlNode):
    '''
    Constructor
    '''

  def evaluate(self):
    subprocess.Popen(self.args,stdout=self.stdout,stderr=subprocess.STDOUT)
      
  def generateData(self,data):
    self.evaluate(self.args, self.stdout)
        

class SVMsClassifier(Model):
  '''
  This is a model that use a SVM classifier to perform in-->out
  '''
  def __init__(self,xmlNode):
    '''
    Constructor
    '''
  
  def evaluate(self):
    return
    
  
  def generateData(self,sampler,data):
    self.evaluate(self.args, self.stdout)
     

#function used to generate a Model class
def returnModelClass(self,modelType,xmlNode):
  '''
  Constructor
  '''
  modelInterfaceDict = {}
  modelInterfaceDict['RAVEN'] = RAVEN
  modelInterfaceDict['SVMsClassifier'] = SVMsClassifier
  try:
    if modelType in modelInterfaceDict.keys():
      return modelInterfaceDict[modelType](xmlNode)
  except:
    raise NameError('not known model type'+modelType)
  
    