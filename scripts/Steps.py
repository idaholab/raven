'''
Created on Feb 21, 2013

@author: crisr
'''
import xml.etree.ElementTree as ET
import time
import os
import copy
from BaseType import BaseType

class Step(BaseType):
  '''
  this class implement one step of the simulation pattern
  '''
  def __init__(self):
    BaseType.__init__(self)
    self.parList  = []    #list of list [[internal name, type, subtype, global name]]
    self.directory= ''    #where eventual files need to be saved
    self.typeDict = {}    #for each internal name identifier the allowed type
  def readMoreXML(self,xmlNode):
    try:self.directory = xmlNode.attrib['directory']
    except: pass
    for child in xmlNode:
      self.typeDict[child.tag] = child.attrib['type']
      self.parList.append([child.tag,self.typeDict[child.tag],child.attrib['subtype'],child.text])
  def addInitParams(self,tempDict):
    for List in self.parList: 
      tempDict[List[0]] = List[1]+':'+List[2]+':'+List[3]


  def takeAstep(self,inDictionary):
    '''main driver for a step'''
    #set up the working directory in the job handler
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].reset()                       #reset the ROM for a new run
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].train(inDictionary['Output']) #train      the ROM for a new run
    if 'Tester' in inDictionary.keys():
      if 'ROM' in inDictionary.keys(): inDictionary['Tester'].testROM(inDictionary['ROM'])
      else: inDictionary['Tester'].testOutput(inDictionary['Output'])
    else: counter = 0
    if 'Sampler' in inDictionary.keys():
      inDictionary['Sampler'].initialize()
      converged = False
      newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary['Model'],
                                                        inDictionary['jobHandler'].runInfoDict['batchSize'])
    else:
      newInputs = [inDictionary['Input']]
    runningList = []
    for newInput in newInputs:
      runningList.append(inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler']))

    print('qui')
    print(len(runningList))
    print(inDictionary['Output'])
    
    
    
    for i in range(len(runningList)):
      if runningList[i].isDone:
        finisishedjob = runningList.pop(i)
        for output in inDictionary['Output']:
          print(output)
          output.add(finisishedjob.output)
        counter += counter
        if 'ROM' in inDictionary.keys: inDictionary['ROM'].trainROM(    inDictionary['Output']) #train      the ROM for a new run
        if 'Tester' in inDictionary.keys:
          if 'ROM' in inDictionary.keys:
            coverged = inDictionary['Tester'].testROM(inDictionary['ROM'])
          else:
            coverged = inDictionary['Tester'].testOutput(inDictionary['Output'])
        else:
          coverged = counter > self.numberIteration
        if not coverged:
          newInput = inDictionary['Sampler'].generateInput(inDictionary['Input'],inDictionary['Model'])
          runningList.append(inDictionary['Model'].run(newInput))
    time.sleep(2.0)

    

class SimpleRun(Step):
  pass


 





def returnInstance(Type):
  base = 'Step'
  InterfaceDict = {}
  InterfaceDict['SimpleRun'    ] = SimpleRun
  InterfaceDict['MultiRun'     ] = Step

  try:
    if Type in InterfaceDict.keys():
      return InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
  
  
  
  
  