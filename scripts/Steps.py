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
    if 'Sampler' in inDictionary.keys(): maxNumberIteration = inDictionary['Sampler'].limit
    else: maxNumberIteration = 1
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].reset()                             #if present reset the ROM for a new run
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].train(inDictionary['Output'])       #if present train the ROM for a new run
    if 'Tester' in inDictionary.keys():
      if 'ROM' in inDictionary.keys(): inDictionary['Tester'].testROM(inDictionary['ROM'])     #make aware the tester (if present) of the presence of a ROM
      inDictionary['Tester'].testOutput(inDictionary['Output'])                                #initialize the output with the tester
    counter = 0                                                                              #we initialize a counter to be safe
    if 'Sampler' in inDictionary.keys():
      inDictionary['Sampler'].initialize()                                                     #if a sampler is use it gets initialized
      newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],            #if the sampler is used the new input is generated
                  inDictionary['Model'],inDictionary['jobHandler'].runInfoDict['batchSize'])   #..continuation
    else:
      newInputs = [inDictionary['Input']]                                                      #no sampler the input is the original one (list of input for multiple input sets)
    converged = False   
    runningList = []                                                                         #this list is used to store running jobs
    for newInput in newInputs:
      runningList.append(inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])) #adding to the running job list
      counter += 1
      if counter >= maxNumberIteration: break
    #since now the list is full up to the limit (batch size number)
    while len(runningList)>0:
      for i in range(len(runningList)):                                                          #check on all the job in list
        #job in the list finished event
        if runningList[i].isDone:                                                                  #if the job is done
          finisishedjob = runningList.pop(i)                                                         #remove it from the list
          for output in inDictionary['Output']:                                                      #for all expected outputs
            inDictionary['Model'].collectOutput(finisishedjob,output)                             #the model is tasket to provide the needed info to harvest the output
          if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])    #train      the ROM for a new run
          #the harvesting process is done moving forward with the convergence checks
          if 'Tester' in inDictionary.keys():
            if 'ROM' in inDictionary.keys():
              converged = inDictionary['Tester'].testROM(inDictionary['ROM'])                           #the check is performed on the information content of the ROM
            else:
              converged = inDictionary['Tester'].testOutput(inDictionary['Output'])                     #the check is done on the information content of the output
          if not converged and counter < maxNumberIteration:
            newInput = inDictionary['Sampler'].generateInput(inDictionary['Input'],inDictionary['Model'])
            runningList.append(inDictionary['Model'].run(newInput))
            counter += counter
          elif converged:
            for j in range(len(runningList)):
              runningList[j].kill
              runningList.pop(j)
            break

    

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
  
  
  
  
  