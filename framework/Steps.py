'''
Created on Feb 21, 2013

@author: crisr
'''
import xml.etree.ElementTree as ET
import time
from BaseType import BaseType
import copy

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
  def initializeStep(self,inDictionary):
    # cleaning up the model
    inDictionary['Model'].reset(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    print('the model '+inDictionary['Model'].name+' has been reset')
  def takeAstep(self,inDictionary):
    raise IOError('for this model the takeAstep has not yet being implemented')


class SimpleRun(Step):
  '''This is the step that will perform just one evaluation'''
  def takeAstep(self,inDictionary):
    '''main driver for a step'''
    #inDictionary['OriginalInput'] = copy.deepcopy(inDictionary['Input'])
    print('beginning of the step: '+self.name)
    self.initializeStep(inDictionary)
    jobHandler = inDictionary['jobHandler']
    inDictionary["Model"].run(inDictionary['Input'],inDictionary['Output'],inDictionary['jobHandler'])
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        for output in inDictionary['Output']:                                                      #for all expected outputs
            inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(0.1)
    for output in inDictionary['Output']:
      output.finalize()

class PostProcessing(Step):
  '''this class is used to perform post processing on data'''
  def takeAstep(self,inDictionary):
    '''main driver for a step'''
    if 'Model' in inDictionary.keys():
      if inDictionary['Model'].type == 'Filter':
        for i in xrange(len(inDictionary['Input'])):
          inDictionary['Model'].run(inDictionary['Input'][i],inDictionary['Output'][i])

class MultiRun(Step):
  '''this class implement one step of the simulation pattern' where several runs are needed'''
  def __init__(self):
    Step.__init__(self)
    self.maxNumberIteration = 0
  def addCurrentSetting(self,originalDict):
    originalDict['max number of iteration'] = self.maxNumberIteration
  def initializeStep(self,inDictionary):
    Step.initializeStep(self,inDictionary)
    #get the max number of iteration in the step
    if 'Sampler' in inDictionary.keys(): self.maxNumberIteration = inDictionary['Sampler'].limit
    else: self.maxNumberIteration = 1
    print('limit to the number of simulation is: '+str(self.maxNumberIteration))
  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
    if 'Sampler' in inDictionary.keys():
      if inDictionary['Sampler'].type == 'DynamicEventTree':
        self.takeAstepRunDET(inDictionary)
        return
    self.takeAstepRun(inDictionary)
  def takeAstepIni(self,inDictionary):
    '''main driver for a step'''
    print('beginning of the step: '+self.name)
    self.initializeStep(inDictionary)
####ROM
    if 'ROM' in inDictionary.keys():
      #clean up the ROM, currently we can not add to an already existing ROM
      inDictionary['ROM'].reset()
      print('the ROM '+inDictionary['ROM'].name+' has been reset')
      #train the ROM
      inDictionary['ROM'].train(inDictionary['Output'])
      print('the ROM '+inDictionary['ROM'].name+' has been trained')
####Tester
    if 'Tester' in inDictionary.keys():
      inDictionary['Tester'].reset()
      if 'ROM' in inDictionary.keys():
        inDictionary['Tester'].getROM(inDictionary['ROM'])     #make aware the tester (if present) of the presence of a ROM
        print('the tester '+ inDictionary['Tester'].name +' have been target on the ROM ' + inDictionary['ROM'].name)
      inDictionary['Tester'].getOutput(inDictionary['Output'])                                #initialize the output with the tester
      if self.debug: print('the tester '+inDictionary['Tester'].name+' have been initialized on the output '+ inDictionary['Output'])
####Sampler and run
    if 'Sampler' in inDictionary.keys():
      #if a sampler is use it gets initialized & generate new inputs
      inDictionary['Sampler'].initialize()
      newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary["Model"],inDictionary['jobHandler'].runInfoDict['batchSize'])
      for newInput in newInputs:
        inDictionary["Model"].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
    else:
      #we start the only case we have
      inDictionary["Model"].run(inDictionary['Input'],inDictionary['Output'],inDictionary['jobHandler'])
  def takeAstepRun(self,inDictionary):
    converged = False
    jobHandler = inDictionary['jobHandler']
    while True:
      finishedJobs = jobHandler.getFinished()
      #loop on the finished jobs
      for finishedJob in finishedJobs:
        if 'Sampler' in inDictionary.keys():
          inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
        for output in inDictionary['Output']:                                                      #for all expected outputs
            inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
        if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
        #the harvesting process is done moving forward with the convergence checks
        if 'Tester' in inDictionary.keys():
          if 'ROM' in inDictionary.keys():
            converged = inDictionary['Tester'].testROM(inDictionary['ROM'])                           #the check is performed on the information content of the ROM
          else:
            converged = inDictionary['Tester'].testOutput(inDictionary['Output'])                     #the check is done on the information content of the output
        if not converged:
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
            if (jobHandler.getNumSubmitted() < int(self.maxNumberIteration)) and inDictionary['Sampler'].amIreadyToProvideAnInput():
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
        elif converged:
          jobHandler.terminateAll()
          break
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(0.1)
    for output in inDictionary['Output']:
      output.finalize()
  def takeAstepRunDET(self,inDictionary):
    converged = False
    jobHandler = inDictionary['jobHandler']
    while True:
      finishedJobs = jobHandler.getFinished()
      #loop on the finished jobs
      for finishedJob in finishedJobs:
        if 'Sampler' in inDictionary.keys():
          inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
        for output in inDictionary['Output']:                                                      #for all expected outputs
            inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
        if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
        #the harvesting process is done moving forward with the convergence checks
        if 'Tester' in inDictionary.keys():
          if 'ROM' in inDictionary.keys():
            converged = inDictionary['Tester'].testROM(inDictionary['ROM'])                           #the check is performed on the information content of the ROM
          else:
            converged = inDictionary['Tester'].testOutput(inDictionary['Output'])                     #the check is done on the information content of the output
        if not converged:
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
            if (jobHandler.getNumSubmitted() < int(self.maxNumberIteration)) and inDictionary['Sampler'].amIreadyToProvideAnInput():
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
        elif converged:
          jobHandler.terminateAll()
          break
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(0.1)
      
    for output in inDictionary['Output']:
      output.finalize()

def returnInstance(Type):
  base = 'Step'
  InterfaceDict = {}
  InterfaceDict['SimpleRun'     ] = SimpleRun
  InterfaceDict['MultiRun'      ] = MultiRun
  InterfaceDict['PostProcessing'] = PostProcessing
  try:
    if Type in InterfaceDict.keys():
      return InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
  
  
  
  
  