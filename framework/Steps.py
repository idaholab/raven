'''
Module containing the different type of step allowed
Step is called by simulation
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import time
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseType import BaseType
from utils import metaclass_insert
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
class Step(metaclass_insert(abc.ABCMeta,BaseType)):
  '''this class implement one step of the simulation pattern.
  Initialization happens when the method self is called
  A step could be used more times during the same simulation, if it make sense.

  --Instance--
  myInstance = Simulation(inputFile, frameworkDir,debug=False)
  myInstance.readXML(xml.etree.ElementTree.Element)

  --Usage--
  myInstance.takeAstep(self,inDictionary) nothing more, this initialize the step and run it. Call is coming from Simulation

  --Other external methods--
  myInstance.whoAreYou()                 -see BaseType class-
  myInstance.myInitializzationParams()   -see BaseType class-
  myInstance.myCurrentSetting()          -see BaseType class-

  --Adding a new step subclass--  
  <MyClass> should inherit at least from Step or from another step already presents

  DO NOT OVERRIDE any of the class method that are not starting with self.local*
  
  ADD your class to the dictionary __InterfaceDict at the end of the module

  The following method overriding is MANDATORY:
  self.localInputAndChecks(xmlNode)     : used to specialize the xml reading
  self.localAddInitParams(tempDict)     : used to add the local parameters and values to be printed
  self.localInitializeStep(inDictionary): called after this call the step should be able the accept the call self.takeAstep(inDictionary):
  self.localTakeAstepRun(inDictionary)  : this is where the step happens, after this call the output is ready
  '''
  
  def __init__(self):
    BaseType.__init__(self)
    self.parList  = []    #list of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.typeDict = {}    #for each role of the step the corresponding  used type

  def readMoreXML(self,xmlNode):
    '''add the readings for who plays the step roles
    after this call everything will not change further in the life of the step object should have been set
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    for child in xmlNode:
      self.parList.append([child.tag,child.attrib['class'],child.attrib['type'],child.text])
    self.localInputAndChecks(xmlNode)

  def localInputAndChecks(self,xmlNode):
    '''place here specialized reading, input consistency check and 
    initialization of what will not change during the whole life of the object
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    pass
  
  def addInitParams(self,tempDict):
    '''Export to tempDict the information that will stay constant during the existence of the instance of this class'''
    for List in self.parList:
      tempDict[List[0]] = ' type: '+List[1]+' SubType: '+List[2]+'  Global name: '+List[3]
    self.localAddInitParams(tempDict)

  def localAddInitParams(self,tempDict):
    '''place here a specialization of the exporting of what in the step is added to the initial parameters
    the printing format of tempDict is key: tempDict[key]'''
    pass

  def __initializeStep(self,inDictionary):
    '''the job handler is restarted'''
    inDictionary['jobHandler'].StartingNewStep()
    self.localInitializeStep(inDictionary)
  
  @abc.abstractmethod
  def localInitializeStep(self,inDictionary):
    '''this is the API for the local initialization of the children classes'''
    pass

  @abc.abstractmethod
  def localTakeAstepRun(self,inDictionary):
    '''this is the API for the local run of a step for the children classes'''
    pass

  def takeAstep(self,inDictionary):
    '''this should work for everybody just split the step in an initialization and the run itself
    inDictionary[role]=instance or list of instance'''
    if self.debug: print('Initializing....')
    self.__initializeStep(inDictionary)
    if self.debug: print('Initialization done starting the run....')
    self.localTakeAstepRun(inDictionary)


#----------------------------------------------------------------------------------------------------
class SingleRun(Step):
  '''This is the step that will perform just one evaluation'''
  def localInitializeStep(self,inDictionary):
    '''this is the initialization for a generic step performing runs '''
    #checks
    if 'Model'  not in inDictionary.keys(): raise IOError ('It is not possible a run without a model!!!')
    if 'Input'  not in inDictionary.keys(): raise IOError ('It is not possible a run without an Input!!!')
    if 'Output' not in inDictionary.keys(): raise IOError ('It is not possible a run without an Output!!!')
    #Model initialization
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    if self.debug: print('The model '+inDictionary['Model'].name+' has been initialized')
    #HDF5 initialization
    for i in range(len(inDictionary['Output'])):
      try: #try is used since files for the moment have no type attribute
        if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].addGroupInit(self.name)
      except: pass
    
  def localTakeAstepRun(self,inDictionary):
    '''main driver for a step'''
    jobHandler = inDictionary['jobHandler']
    inDictionary["Model"].run(inDictionary['Input'],inDictionary['jobHandler'])
    if inDictionary["Model"].type == 'Code': 
      while True:
        finishedJobs = jobHandler.getFinished()
        for finishedJob in finishedJobs:
          for output in inDictionary['Output']:                                         #for all expected outputs
              inDictionary['Model'].collectOutput(finishedJob,output)                   #the model is tasket to provide the needed info to harvest the output
        if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
          break
        time.sleep(0.1)
    else:
      for output in inDictionary['Output']:
        inDictionary['Model'].collectOutput(None,output)


#----------------------------------------------------------------------------------------------------
class MultiRun(SingleRun):
  '''this class implement one step of the simulation pattern' where several runs are needed without being adaptive'''
  def __init__(self):
    SingleRun.__init__(self)
    self.maxNumberIteration = 0

  def addCurrentSetting(self,originalDict):
    originalDict['max number of iteration'] = self.maxNumberIteration

  def localInitializeStep(self,inDictionary):
    SingleRun.localInitializeStep(self,inDictionary)
    #checks
    if 'Sampler'  not in inDictionary.keys(): raise IOError ('It is not possible a multi-run without a Sampler!!!')
    #get the max number of iteration in the step
    if self.debug: print('The max the number of simulation is: '+str(self.maxNumberIteration))
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].addLoadingSource(inDictionary['Output'])
    inDictionary['Sampler'].initialize()
    newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary["Model"],inDictionary['jobHandler'].runInfoDict['batchSize'])
    for newInput in newInputs:
      inDictionary["Model"].run(newInput,inDictionary['jobHandler'])
      if inDictionary["Model"].type != 'Code':
        # if the model is not a code, collect the output right after the evaluation => the response is overwritten at each "run"
        for output in inDictionary['Output']: inDictionary['Model'].collectOutput(inDictionary['jobHandler'],output)

  def localTakeAstepRun(self,inDictionary):
    jobHandler = inDictionary['jobHandler']
    while True:
      if inDictionary["Model"].type == 'Code': 
        finishedJobs = jobHandler.getFinished()
        #loop on the finished jobs
        for finishedJob in finishedJobs:
          if 'Sampler' in inDictionary.keys(): inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
          for output in inDictionary['Output']:                                                      #for all expected outputs
              inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
          if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
          #the harvesting process is done moving forward with the convergence checks
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
            if inDictionary['Sampler'].amIreadyToProvideAnInput():
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
        if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
          break
        time.sleep(0.001)
      else:
        finishedJob = 'empty'
        if inDictionary['Sampler'].amIreadyToProvideAnInput():
          newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
          inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
          for output in inDictionary['Output']:
            inDictionary['Model'].collectOutput(finishedJob,output) 
        else:
          break
        time.sleep(0.001)
    #remember to close the rom to decouple the data stroed in the rom from the framework
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].close()


#----------------------------------------------------------------------------------------------------
class Adaptive(MultiRun):
  '''this class implement one step of the simulation pattern' where several runs are needed in an adaptive scheme'''
  def readMoreXML(self,xmlNode):
    '''we add the information where the projector should take the output'''
    MultiRun.readMoreXML(self, xmlNode)
    #checks
    if xmlNode.find('Tester') == None: raise IOError('it is not possible to define an adaptive step without a tester')
    if xmlNode.find('Projector') == None: raise IOError('it is not possible to define an adaptive step without a projector')
    #find out where in the self.parList is the source of the projector
    for string in self.parList:
      if string[1:] == xmlNode.find('Tester').attrib('From').split('|'):
        self.projectorFromIndex = self.parList.index(string)
    
  def addInitParams(self,tempDict):
    '''we add the capability to print to projector source'''
    MultiRun.addInitParams(self, tempDict)
    tempDict['ProjectorSource'] = 'type: '+self.parList[self.projectorFromIndex][0]+'SubType :'+self.parList[self.projectorFromIndex][1]+'Global name :'+self.parList[self.projectorFromIndex][2]

  def localInitializeStep(self,inDictionary):
    MultiRun.localInitializeStep(self,inDictionary)
    #point the projector to its input
    inDictionary['Tester'].reset()
    notYet = True
    while notYet:
      for candidate in inDictionary[self.parList[self.projectorFromIndex]]:
        try:
          if candidate.type == self.parList[self.projectorFromIndex][1] and candidate.subType == self.parList[self.projectorFromIndex][2] and candidate.name == self.parList[self.projectorFromIndex][3]:
            inDictionary['Projector'].initialize(None,candidate)
            notYet = False
        except: pass

  def localTakeAstepRun(self,inDictionary):
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
        inDictionary['Projector'].evaluate()
        converged = inDictionary['Tester'].test(inDictionary['Projector'].output)
        if not converged:
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
            if (jobHandler.getNumSubmitted() < int(self.maxNumberIteration)) and inDictionary['Sampler'].amIreadyToProvideAnInput():
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'],inDictionary['Projector'])
              inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
        elif converged:
          jobHandler.terminateAll()
          break
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
        break
      time.sleep(0.1)
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].close()


#----------------------------------------------------------------------------------------------------
class InOutFromDataBase(Step):
  '''
    This step type is used only to extract information from a DataBase
    @Input, DataBase (for example, HDF5) OR Datas
    @Output,Data(s) (for example, History) or DataBase
  '''
  def localInitializeStep(self,inDictionary):
    avail_out = ['TimePoint','TimePointSet','History','Histories']
    print('STEPS         : beginning of step named: ' + self.name)
    #self.initializeStep(inDictionary)
    # check if #inputs == #outputs
    if len(inDictionary['Input']) != len(inDictionary['Output']):
      raise IOError('STEPS         : ERROR: In Step named ' + self.name + ', the number of Inputs != number of Outputs')
    else:
      self.actionType = []
    for i in xrange(len(inDictionary['Input'])):
      if (inDictionary['Input'][i].type != 'HDF5'):
        if (not (inDictionary['Input'][i].type in ['TimePoint','TimePointSet','History','Histories'])): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts HDF5 as Input only. Got ' + inDictionary['Input'][i].type)
        else:
          if(inDictionary['Output'][i].type != 'HDF5'): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts ' + 'HDF5' + ' as Output only, when the Input is a Datas. Got ' + inDictionary['Output'][i].type)
          else: self.actionType.append('DATAS-HDF5')
      else:
        if (not (inDictionary['Output'][i].type in ['TimePoint','TimePointSet','History','Histories'])): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts A Datas as Output only, when the Input is an HDF5. Got ' + inDictionary['Output'][i].type)
        else: self.actionType.append('HDF5-DATAS')
    try: #try is used since files for the moment have no type attribute
      if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].addGroupInit(self.name)
    except: pass    
    
  def localTakeAstepRun(self,inDictionary):
    for i in xrange(len(inDictionary['Output'])):
      #link the output to the database and construct the Data(s)
      # I have to change it
      if self.actionType[i] == 'HDF5-DATAS':
        inDictionary['Output'][i].addOutput(inDictionary['Input'][i])
      else:
        inDictionary['Output'][i].addGroupDatas(inDictionary['Input'][i])
    return


#----------------------------------------------------------------------------------------------------
class RomTrainer(Step):
  '''This step type is used only to train a ROM
    @Input, DataBase (for example, HDF5)
  '''
  def __init__(self):
    Step.__init__(self)

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting(self,originalDict)

  def localInitializeStep(self,inDictionary):
    '''The initialization step  for a ROM is copying the data out to the ROM (it is a copy not a reference) '''
    for i in xrange(len(inDictionary['Output'])):
      inDictionary['Output'][i].initializeTrain(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'][0])
    try: #try is used since files for the moment have no type attribute
      if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].addGroupInit(self.name)
    except: pass

  def takeAstepIni(self,inDictionary):
    print('STEPS         : beginning of step named: ' + self.name)
    for i in xrange(len(inDictionary['Output'])):
      if (inDictionary['Output'][i].type != 'ROM'):
        raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts a ROM as Output only. Got ' + inDictionary['Output'][i].type)
    if len(inDictionary['Input']) > 1: raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts an Input Only. Number of Inputs = ' + str(len(inDictionary['Input'])))
    self.initializeStep(inDictionary)
    
  def localTakeAstepRun(self,inDictionary):
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for i in xrange(len(inDictionary['Output'])):
      inDictionary['Output'][i].train()
      inDictionary['Output'][i].close()
    return


#----------------------------------------------------------------------------------------------------
class PlottingStep(Step):
  '''this class implement one step of the simulation pattern' where several runs are needed'''
  def __init__(self):
    Step.__init__(self)

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting()

  def localInitializeStep(self,inDictionary):
    pass

  def takeAstepIni(self,inDictionary):
    '''main driver for a step'''
    print('STEPS         : beginning of the step: '+self.name)
    self.initializeStep(inDictionary)

  def localTakeAstepRun(self,inDictionary):
    pass
   


#----------------------------------------------------------------------------------------------------
class SCRun(Step):
  '''this class implement one step of the simulation pattern' where several runs are needed'''
  def __init__(self):
    Step.__init__(self)
    self.maxNumberIteration = 0

  def addCurrentSetting(self,originalDict):
    originalDict['max number of iteration'] = self.maxNumberIteration

  def localInitializeStep(self,inDictionary):
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


__interFaceDict                      = {}
__interFaceDict['SingleRun'        ] = SingleRun
__interFaceDict['MultiRun'         ] = MultiRun
__interFaceDict['SCRun'            ] = SCRun
__interFaceDict['Adaptive'         ] = Adaptive
__interFaceDict['InOutFromDataBase'] = InOutFromDataBase 
__interFaceDict['RomTrainer'       ] = RomTrainer
__interFaceDict['Plotting'         ] = PlottingStep
__base                               = 'Step'

def returnInstance(Type):
  return __interFaceDict[Type]()
  raise NameError('not known '+__base+' type '+Type)
  
  
  
  
  
