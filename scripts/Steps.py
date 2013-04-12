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
  '''this class implement one step of the simulation pattern'''
  def __init__(self):
    self.debug = True
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
  def addCurrentSetting(self,originalDict):
    originalDict['max number of iteration'] = self.maxNumberIteration

  def initializeStep(self,inDictionary):
    #get the max number of iteration in the step
    if 'Sampler' in inDictionary.keys(): self.maxNumberIteration = inDictionary['Sampler'].limit
    else: self.maxNumberIteration = 1
    if self.debug: print('limit to the number of simulation is: '+str(self.maxNumberIteration))
    # cleaning up the model
    inDictionary['Model'].reset(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    if self.debug: print('the model '+inDictionary['Model'].name+' has been reset')
    if self.debug: inDictionary['Model'].printMe()
  
  def takeAstep(self,inDictionary):
    '''main driver for a step'''
    if self.debug: print('beginning the step: '+self.name)
    self.initializeStep(inDictionary)
    #clean up the ROM
    if 'ROM' in inDictionary.keys():
      inDictionary['ROM'].reset()                             #if present reset the ROM for a new run
      if self.debug: print('the ROM '+inDictionary['ROM'].name+' has been reset')
    #train the ROM
    if 'ROM' in inDictionary.keys():
      inDictionary['ROM'].train(inDictionary['Output'])       #if present train the ROM for a new run
      if self.debug: print('the ROM '+inDictionary['ROM'].name+' has been trained')
    
    if 'Tester' in inDictionary.keys():
      inDictionary['Tester'].reset()
      if 'ROM' in inDictionary.keys():
        inDictionary['Tester'].getROM(inDictionary['ROM'])     #make aware the tester (if present) of the presence of a ROM
        if self.debug: print('the tester '+self.inDictionary['Tester'].name+' have been target on the ROM '+self.inDictionary['ROM'].name)
      inDictionary['Tester'].getOutput(inDictionary['Output'])                                #initialize the output with the tester
      if self.debug: print('the tester '+self.inDictionary['Tester'].name+' have been initialized on the output '+self.inDictionary['Output'])
    
    runningList =[]
    submittedCounter = 0                               #we initialize a counter to be safe
    if 'Sampler' in inDictionary.keys():
      inDictionary['Sampler'].initialize()              #if a sampler is use it gets initialized
      for i in range(inDictionary['jobHandler'].runInfoDict['batchSize']):
        newInput = inDictionary['Sampler'].generateInput(inDictionary["Model"],inDictionary['Input'])
        runningList.append(inDictionary["Model"].run(newInput,inDictionary['Output'],inDictionary['jobHandler']))
        submittedCounter += 1
    else:
      runningList.append(inDictionary["Model"].run(inDictionary['Input'],inDictionary['Output'],inDictionary['jobHandler']))
      submittedCounter += 1
    converged = False

    #since now the list is full up to the limit (batch size number)
    while len(runningList)>0:
      i=0
      while i <=len(runningList)-1:                                                                  #check on all the job in list
        #job in the list finished event
        if runningList[i].isDone():                                                                  #if the job is done
          finisishedjob = runningList.pop(i)                                                         #remove it from the list
          if inDictionary['DataSet']:
            inDictionary['Model'].addDataSetGroup(finisishedjob,inDictionary['DataSet'])
          for output in inDictionary['Output']:                                                      #for all expected outputs
            loadingObj = None
            if output.dname:
              loadingObj = inDictionary['DataSet']
              inDictionary['Model'].collectOutput(finisishedjob,loadingObj,output)
            else:
              inDictionary['Model'].collectOutput(finisishedjob,loadingObj,output)                                #the model is tasket to provide the needed info to harvest the output
          if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
          #the harvesting process is done moving forward with the convergence checks
          if 'Tester' in inDictionary.keys():
            if 'ROM' in inDictionary.keys():
              converged = inDictionary['Tester'].testROM(inDictionary['ROM'])                           #the check is performed on the information content of the ROM
            else:
              converged = inDictionary['Tester'].testOutput(inDictionary['Output'])                     #the check is done on the information content of the output
          if not converged:
            if int(submittedCounter) < int(self.maxNumberIteration):
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              runningList.append(inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler']))
              submittedCounter = inDictionary['Sampler'].counter
          elif converged:
            j=0
            while j <=len(runningList)-1:
              runningList[j].kill
              runningList.pop(j)
              j+=1
            break
        i+=1
      time.sleep(0.3)
    for output in inDictionary['Output']:
      output.finalizeOutput()
      
    quit()

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
  
  
  
  
  