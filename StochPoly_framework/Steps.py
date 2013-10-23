'''
Created on Feb 21, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

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
    #print(inDictionary['Model'])
    inDictionary['Model'].reset(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    print('STEPS         : the model '+inDictionary['Model'].name+' has been reset')
    for i in range(len(inDictionary['Output'])):
      try: 
        if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].addGroupInit(self.name)
      except: pass
    return

  def takeAstep(self,inDictionary):
    raise IOError('STEPS         : For this model the takeAstep has not yet being implemented')


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

class PostProcessing(Step):
  '''this class is used to perform post processing on data'''
  def initializeStep(self,inDictionary):
    Step.initializeStep(self,inDictionary)

  def takeAstep(self,inDictionary):
    '''main driver for a step'''
    # Initialize the Step
    print('Intialize step PostProcessing')
    Step.initializeStep(self,inDictionary)
    # Run the step
    print('Run step PostProcessing')
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
    print('STEPS         : limit to the number of simulation is: '+str(self.maxNumberIteration))
    if 'ROM' in inDictionary.keys():
      inDictionary['ROM'].addLoadingSource(inDictionary['Input'])

    #if 'DataBases' in inDictionary.keys():
    #  addGroupInit()
    #FIXME this reports falsely if sampler.limit is set in sampler.initialize

  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
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
        if inDictionary['Model'].type != 'Code':
          for output in inDictionary['Output']:                                                      #for all expected outputs
            inDictionary['Model'].collectOutput(None,output) 
    else:
      #we start the only case we have
      inDictionary["Model"].run(inDictionary['Input'],inDictionary['Output'],inDictionary['jobHandler'])

  def takeAstepRun(self,inDictionary):
    converged = False
    jobHandler = inDictionary['jobHandler']
    while True:
      if inDictionary["Model"].type == 'Code':
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
        time.sleep(0.001)
      else:
        finishedJob = 'empty'
        if inDictionary['Sampler'].amIreadyToProvideAnInput():
          newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
          inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
          for output in inDictionary['Output']:
            inDictionary['Model'].collectOutput(finishedJob,output) 
        else:
          break
        time.sleep(0.001)
   

class SCRun(Step):
  '''this class implement one step of the simulation pattern' where several runs are needed'''
  def __init__(self):
    Step.__init__(self)
    self.maxNumberIteration = 0

  def addCurrentSetting(self,originalDict):
    originalDict['max number of iteration'] = self.maxNumberIteration

  def initializeStep(self,inDictionary):
    # TODO is this necessary? Step.initializeStep(self,inDictionary)
    #get the max number of iteration in the step
    #if 'Sampler' in inDictionary.keys(): self.maxNumberIteration = inDictionary['Sampler'].limit
    #else: self.maxNumberIteration = 1
    #TODO make sure this is getting the right limit
    print('limit to the number of simulation is: '+str(self.maxNumberIteration))
    if 'ROM' in inDictionary.keys():
      inDictionary['ROM'].addLoadingSource(inDictionary['Input'])

  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
    self.takeAstepRun(inDictionary)

  def takeAstepIni(self,inDictionary):
    '''main driver for a step'''
    print('STEPS         : beginning of the step: '+self.name)
    self.initializeStep(inDictionary)
####ROM
#    if 'ROM' in inDictionary.keys():
#      #clean up the ROM, currently we can not add to an already existing ROM
#      inDictionary['ROM'].reset()
#      print('the ROM '+inDictionary['ROM'].name+' has been reset')
#      #train the ROM
#      inDictionary['ROM'].train(inDictionary['Output'])
#      print('the ROM '+inDictionary['ROM'].name+' has been trained')
####Tester
    if 'Tester' in inDictionary.keys():
      inDictionary['Tester'].reset()
      if 'ROM' in inDictionary.keys():
        inDictionary['Tester'].getROM(inDictionary['ROM'])     #make aware the tester (if present) of the presence of a ROM
        print('the tester '+ inDictionary['Tester'].name +' have been target on the ROM ' + inDictionary['ROM'].name)
      inDictionary['Tester'].getOutput(inDictionary['Output'])                                #initialize the output with the tester
      if self.debug: print('the tester '+inDictionary['Tester'].name+' have been initialized on the output '+ inDictionary['Output'])
####Sampler and run
#    if 'Sampler' in inDictionary.keys(): #it shouldn't be
#      #if a sampler is use it gets initialized & generate new inputs
#      inDictionary['Sampler'].initialize()
#      newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary["Model"],inDictionary['jobHandler'].runInfoDict['batchSize'])
#      for newInput in newInputs:
#        #noteToSelf this runs the inputs made by the sampler
#        inDictionary["Model"].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
#    else:
#      pass
      #we start the only case we have
      #noteToSelf we don't want to run anything
      #inDictionary["Model"].run(inDictionary['Input'],inDictionary['Output'],inDictionary['jobHandler'])

  def takeAstepRun(self,inDictionary):
    #print('At takeAstepRun',inDictionary['Input'])
    inDictionary['ROM'].train(inDictionary)
    #VERY WRONG... IT IS GOING TO BE RESTRUCTURED SOON!!!!!!!!! ANDREA
    inDictionary['ROM'].fillDistribution(inDictionary['Sampler'].distDict)
#    converged = False
#    jobHandler = inDictionary['jobHandler']
#    while not converged:
#      finishedJobs = jobHandler.getFinished()
#      converged = len(finishedJobs)==self.maxNumberIteration
#      #loop on the finished jobs
#      for finishedJob in finishedJobs:
#        if 'Sampler' in inDictionary.keys():
#          inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
#        for output in inDictionary['Output']:                                                      #for all expected outputs
#            inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
#        #if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
#        #the harvesting process is done moving forward with the convergence checks
#        #if 'Tester' in inDictionary.keys():
#        #  if 'ROM' in inDictionary.keys():
#        #    converged = inDictionary['Tester'].testROM(inDictionary['ROM'])                           #the check is performed on the information content of the ROM
#        #  else:
#        #    converged = inDictionary['Tester'].testOutput(inDictionary['Output'])                     #the check is done on the information content of the output
#        if not converged:
#          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
#            if (jobHandler.getNumSubmitted() < int(self.maxNumberIteration)) and inDictionary['Sampler'].amIreadyToProvideAnInput():
#              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
#              inDictionary['Model'].run(newInput,inDictionary['Output'],inDictionary['jobHandler'])
#        #elif converged:
#        #  jobHandler.terminateAll()
#        #  break
#      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
#        break
#      time.sleep(0.1)
#    for output in inDictionary['Output']:
#      output.finalize()
#    print('HERE',inDictionary.keys())
    #if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run

class ExtractFromDataBase(Step):
  '''
    This step type is used only to extract information from a DataBase
    @Input, DataBase (for example, HDF5)
    @Output,Data(s) (for example, History)
  '''
  def __init__(self):
    Step.__init__(self)
    

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting(self,originalDict)

  def initializeStep(self,inDictionary):
    # No Model initialization here... There is no model at all!!!!
    return

  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
    self.takeAstepRun(inDictionary)

  def takeAstepIni(self,inDictionary):
    avail_out = 'TimePoint-TimePointSet-History-Histories'
    print('STEPS         : beginning of step named: ' + self.name)
    self.initializeStep(inDictionary)
    # check if #inputs == #outputs
    if len(inDictionary['Input']) != len(inDictionary['Output']):
      raise IOError('STEPS         : ERROR: In Step named ' + self.name + ', the number of Inputs != number of Outputs')
    for i in xrange(len(inDictionary['Input'])):
      if (inDictionary['Input'][i].type != "HDF5"):
        raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts HDF5 as Input only. Got ' + inDictionary['Input'][i].type)
    for i in xrange(len(inDictionary['Output'])):
      if (not inDictionary['Output'][i].type in avail_out.split('-')):
        raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts ' + avail_out + ' as Output only. Got ' + inDictionary['Output'][i].type)
    return    
    
  def takeAstepRun(self,inDictionary):
    for i in xrange(len(inDictionary['Output'])):
      #link the output to the database and construct the Data(s)
      inDictionary['Output'][i].addOutput(inDictionary['Input'][i])
    return

class RomTrainer(Step):
  '''
    This step type is used only to train a ROM
    @Input, DataBase (for example, HDF5)
    @Output,Data(s) (for example, History)
  '''
  def __init__(self):
    Step.__init__(self)

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting(self,originalDict)

  def initializeStep(self,inDictionary):
    # No Model initialization here... There is no model at all!!!!
    for i in xrange(len(inDictionary['Input'])):
      inDictionary['Output'][i].addLoadingSource(inDictionary['Input'][i])
    return

  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
    self.takeAstepRun(inDictionary)

  def takeAstepIni(self,inDictionary):
    avail_in = 'TimePoint-TimePointSet-History-Histories'
    print('STEPS         : beginning of step named: ' + self.name)
    for i in xrange(len(inDictionary['Input'])):
      if (not inDictionary['Input'][i].type in avail_in.split('-')):
        raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts '+avail_in+' as Input only. Got ' + inDictionary['Input'][i].type)
    for i in xrange(len(inDictionary['Output'])):
      if (inDictionary['Output'][i].type != 'ROM'):
        raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts a ROM as Output only. Got ' + inDictionary['Output'][i].type)
    self.initializeStep(inDictionary)
    
    return    
    
  def takeAstepRun(self,inDictionary):
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for i in xrange(len(inDictionary['Output'])):
      inDictionary['Output'][i].train()
    return

class PlottingStep(Step):
  '''this class implement one step of the simulation pattern' where several runs are needed'''
  def __init__(self):
    Step.__init__(self)
    

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting()

  def initializeStep(self,inDictionary):
    pass

  def takeAstep(self,inDictionary):
    '''this need to be fixed for the moment we branch for Dynamic Event Trees'''
    self.takeAstepIni(inDictionary)
    self.takeAstepRun(inDictionary)

  def takeAstepIni(self,inDictionary):
    '''main driver for a step'''
    print('STEPS         : beginning of the step: '+self.name)
    self.initializeStep(inDictionary)

  def takeAstepRun(self,inDictionary):
    pass


def returnInstance(Type):
  base = 'Step'
  InterfaceDict = {}
  InterfaceDict['SimpleRun'     ] = SimpleRun
  InterfaceDict['MultiRun'      ] = MultiRun
  InterfaceDict['PostProcessing'] = PostProcessing
  InterfaceDict['SCRun'         ] = SCRun
  InterfaceDict['Extract'       ] = ExtractFromDataBase 
  InterfaceDict['RomTrainer'    ] = RomTrainer
  InterfaceDict['Plotting'      ] = PlottingStep
  try:
    if Type in InterfaceDict.keys():
      return InterfaceDict[Type]()
  except:
    raise NameError('not known '+base+' type'+Type)
  
  
  
  
  
