'''
Created on Mar 8, 2013

@author: crisr
'''
import time
from BaseType import BaseType


class Test(BaseType):
  ''' 
  a genral class containing the distributions
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.toBeTested  = []
    self.tolerance = 0.0

  def readMoreXML(self,xmlNode):
    try: self.toBeTested = xmlNode.text.split(',')
    except: raise IOError('not found variable list to be tested in tester '+self.name)
    try: self.name = xmlNode.attrib['tolerance']
    except: raise IOError('not found tolerance for tester '+self.name)
  def addInitParams(self,tempDict):
    tempDict['toBeTested'] = self.toBeTested
    tempDict['tolerance' ] = self.tolerance



class Sigma(Test):
  def seekConvergence(self,inDictionary):  #if a ROM present ??? 
    if 'Sampler' in inDictionary.keys:
      currentSigma = self.sigma*2.0
      submitted = 0
      done      = 0
      while currentSigma>self.sigma or submitted<self.limit:
        if inDictionary['jobHandler'].spaceAvailable:
          newInput = inDictionary['Sampler'].sampleInput(inDictionary['Model'],inDictionary['Input'],submitted)
          inDictionary['jobHandler'].addRunningList(inDictionary['Model'].evaluate(Input=newInput,Output=inDictionary['Output']),inDictionary['Output'])
          submitted +=1
          time.sleep(1.0)  #every sec check for space available
          if done < inDictionary['jobHandler'].done:
            done = inDictionary['jobHandler'].done
            currentSigma = self.test(inDictionary['Output'])
          
      while inDictionary['jobHandler'].busy:
        time.sleep(1.0) #every sec check if remaining job are done
    else:
      raise IOError('the probality test')


class Integral(Test):

  def seekConvergence(self,inDictionary):
    if 'Sampler' in inDictionary.keys:
      currentError = self.error*2.0
      submitted = 0
      done      = 0
      while currentError>self.error or submitted<self.limit:
        if inDictionary['jobHandler'].spaceAvailable:
          newInput = inDictionary['Sampler'].sampleInput(inDictionary['Model'],inDictionary['Input'],submitted)
          inDictionary['jobHandler'].addRunningList(inDictionary['Model'].evaluate(Input=newInput,Output=inDictionary['Output']),inDictionary['Output'])
          submitted +=1
          time.sleep(1.0)  #every sec check for space available
          if done < inDictionary['jobHandler'].done:
            done = inDictionary['jobHandler'].done
            currentSigma = self.test(inDictionary['Output'])
          
      while inDictionary['jobHandler'].busy:
        time.sleep(1.0) #every sec check if remaining job are done

      


#function used to generate a Model class
def returnInstance(Type):
  base = 'Test'
  InterfaceDict = {}
  InterfaceDict['Sigma'   ] = Sigma
  InterfaceDict['Integral'] = Integral
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
  
  
  
  
  