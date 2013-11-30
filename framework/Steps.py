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
    self.parList    = []    #list of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.__typeDict = {}    #for each role of the step the corresponding  used type
    self.sleepTime  = 0.001 #waiting time before checking if a run is finished

  def readMoreXML(self,xmlNode):
    '''add the readings for who plays the step roles
    after this call everything will not change further in the life of the step object should have been set
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    if 'sleepTime' in xmlNode.attrib.keys(): self.sleepTime = float(xmlNode.attrib['sleepTime'])
    for child in xmlNode:
      self.parList.append([child.tag,child.attrib['class'],child.attrib['type'],child.text])
    self.localInputAndChecks(xmlNode)
    if None in self.parList: raise Exception ('A problem was found in  the definition of the step '+str(self.name))

  @abc.abstractmethod
  def localInputAndChecks(self,xmlNode):
    '''place here specialized reading, input consistency check and 
    initialization of what will not change during the whole life of the object
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    pass
  
  def addInitParams(self,tempDict):
    '''Export to tempDict the information that will stay constant during the existence of the instance of this class'''
    tempDict['sleep Time'] = 'sleep time between testing the end of a run is '+str(self.sleepTime)
    for List in self.parList:
      tempDict[List[0]] = ' Class: '+str(List[1])+' Type: '+str(List[2])+'  Global name: '+str(List[3])
    self.localAddInitParams(tempDict)

  @abc.abstractmethod
  def localAddInitParams(self,tempDict):
    '''place here a specialization of the exporting of what in the step is added to the initial parameters
    the printing format of tempDict is key: tempDict[key]'''
    pass

  def __initializeStep(self,inDictionary):
    '''the job handler is restarted'''
    inDictionary['jobHandler'].startingNewStep()
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
#
#
#
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
        if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
        elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: inDictionary['Output'][i].initialize(inDictionary)
      except AttributeError as ae: print("Error: "+repr(ae))
    
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
        time.sleep(self.sleepTime)
    else:
      for output in inDictionary['Output']:
        inDictionary['Model'].collectOutput(None,output)

  def localInputAndChecks(self,xmlNode):
    pass

  def localAddInitParams(self,tempDict):
    #TODO implement
    pass
#
#
#
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
              inDictionary['Model'].collectOutput(finishedJob,output)                                #the model is tasked to provide the needed info to harvest the output
          if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):                                     #the harvesting process is done moving forward with the convergence checks
            if inDictionary['Sampler'].amIreadyToProvideAnInput():
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
        if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
          break
        time.sleep(self.sleepTime)
      else:
        finishedJob = 'empty'
        if inDictionary['Sampler'].amIreadyToProvideAnInput():
          newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
          inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
          for output in inDictionary['Output']:
            inDictionary['Model'].collectOutput(finishedJob,output) 
        else:
          break
        time.sleep(self.sleepTime)
    #remember to close the rom to decouple the data stroed in the rom from the framework
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].close()
#
#
#
class Adaptive(MultiRun):
  '''this class implement one step of the simulation pattern' where several runs are needed in an adaptive scheme'''
  def localInputAndChecks(self,xmlNode):
    '''we check coherence of Sampler, Functions and Solution Output'''
    #test sampler information:
    foundSampler    = False
    samplCounter    = 0
    foundTargEval   = False
    targEvalCounter = 0
    solExpCounter   = 0
    functionCounter = 0
    foundFunction   = False
    for role in self.parList:
      if   role[0] == 'Sampler'         :
        foundSampler    =True
        samplCounter   +=1
        if not(role[1]=='Samplers' and role[2]=='Adaptive'): raise Exception('The type of sampler used for the step '+str(self.name)+' is not coherent with and adaptive strategy')
      elif role[0] == 'TargetEvaluation':
        foundTargEval   = True
        targEvalCounter+=1
        if role[1]!='Datas'                               : raise Exception('The data chosen for the evaluation of the adaptive strategy is not compatible,  in the step '+self.name)
        if not(['Output']+role[1:] in self.parList[:])    : raise Exception('The data chosen for the evaluation of the adaptive strategy is not in the output list for step'+self.name)
      elif role[0] == 'SolutionExport'  :
        solExpCounter  +=1
        if role[1]!='Datas'                               : raise Exception('The data chosen for exporting the goal function solution is not compatible, in the step '+self.name)
      elif role[0] == 'Function'       :
        functionCounter+=1
        foundFunction   = True
        if role[1]!='Functions'                           : raise Exception('A class function is required as function in an adaptive step, in the step '+self.name)
    if foundSampler ==False: raise Exception('It is not possible to run an adaptive step without a sampler in step '           +self.name)
    if foundTargEval==False: raise Exception('It is not possible to run an adaptive step without a target output in step '     +self.name)
    if foundFunction==False: raise Exception('It is not possible to run an adaptive step without a proper function, in step '  +self.name)
    if samplCounter   >1   : raise Exception('More than one sampler found in step '                                            +self.name)
    if targEvalCounter>1   : raise Exception('More than one target defined for the adaptive sampler found in step '            +self.name)
    if solExpCounter  >1   : raise Exception('More than one output to export the solution of the goal function, found in step '+self.name)
    if functionCounter>1   : raise Exception('More than one function defined in the step '                                     +self.name)
    
  def localInitializeStep(self,inDictionary):
    '''this is the initialization for a generic step performing runs '''
    #checks
    if 'Model'            not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without a model!'                     )
    if 'Input'            not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without an Input!'                    )
    if 'Output'           not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without an Output!'                   )
    if 'Sampler'          not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without an Sampler!'                  )
    if 'TargetEvaluation' not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without an a target for the function!')
    if 'Function'         not in inDictionary.keys(): raise IOError ('It is not possible run '+self.name+' step without an a function!'               )
    #Initialize model
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    if self.debug: print('The model '+inDictionary['Model'].name+' has been initialized')
    #Initialize sampler
    if 'SolutionExport' in inDictionary.keys(): inDictionary['Sampler'].initialize(goalFunction=inDictionary['Function'],solutionExport=inDictionary['SolutionExport'])
    else                                      : inDictionary['Sampler'].initialize(goalFunction=inDictionary['Function'])
    if self.debug: print('The sampler '+inDictionary['Sampler'].name+' has been initialized')
    #HDF5 initialization
    for i in range(len(inDictionary['Output'])):
      try: #try is used since files for the moment have no type attribute
        if 'HDF5' in inDictionary['Output'][i].type:
          inDictionary['Output'][i].addGroupInit(self.name)
          if self.debug: print('The HDF5 '+inDictionary['Output'][i].name+' has been initialized')
        elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPlot']: inDictionary['Output'][i].initialize(inDictionary)
      except AttributeError as ae: print("Error: "+repr(ae))    
    #the first batch of input is generated (and run if the model is not a code)
    newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary["Model"],inDictionary['jobHandler'].runInfoDict['batchSize'])
    for newInput in newInputs:
      inDictionary["Model"].run(newInput,inDictionary['jobHandler'])
      if inDictionary["Model"].type != 'Code':
        # if the model is not a code, collect the output right after the evaluation => the response is overwritten at each "run"
        for output in inDictionary['Output']: 
          if output.type not in ['OutStreamPlot','OutStreamPrint'] : inDictionary['Model'].collectOutput(inDictionary['jobHandler'],output)
          else: output.addOutput()

  def localTakeAstepRun(self,inDictionary):
    jobHandler = inDictionary['jobHandler']
    print('I am running')
    while True:
      if inDictionary["Model"].type == 'Code': 
        finishedJobs = jobHandler.getFinished()
        #loop on the finished jobs
        for finishedJob in finishedJobs:
          if 'Sampler' in inDictionary.keys(): inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
          for output in inDictionary['Output']:                                                      #for all expected outputs
              inDictionary['Model'].collectOutput(finishedJob,output)                                #the model is tasked to provide the needed info to harvest the output
          if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
          for freeSpot in xrange(jobHandler.howManyFreeSpots()):                                     #the harvesting process is done moving forward with the convergence checks
            if inDictionary['Sampler'].amIreadyToProvideAnInput(inLastOutput=inDictionary['TargetEvaluation']):
              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
              inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
        if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
          break
        time.sleep(self.sleepTime)
      else:
        finishedJob = 'empty'
        if inDictionary['Sampler'].amIreadyToProvideAnInput(inLastOutput=inDictionary['TargetEvaluation']):
          newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'])
          inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
          for output in inDictionary['Output']:
            if output.type not in ['OutStreamPlot','OutStreamPrint'] : inDictionary['Model'].collectOutput(inDictionary['jobHandler'],output)
            else: output.addOutput()
        else:
          break
        time.sleep(self.sleepTime)
    #remember to close the rom to decouple the data stroed in the rom from the framework
    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].close()
    
#    print('localTakeAstepRun')
#    converged = False
#    jobHandler = inDictionary['jobHandler']
#    while True:
#      finishedJobs = jobHandler.getFinished()
#      #loop on the finished jobs
#      for finishedJob in finishedJobs:
#        if 'Sampler' in inDictionary.keys():
#          inDictionary['Sampler'].finalizeActualSampling(finishedJob,inDictionary['Model'],inDictionary['Input'])
#        for output in inDictionary['Output']:                                                      #for all expected outputs
#            inDictionary['Model'].collectOutput(finishedJob,output)                                   #the model is tasket to provide the needed info to harvest the output
#        if 'ROM' in inDictionary.keys(): inDictionary['ROM'].trainROM(inDictionary['Output'])      #train the ROM for a new run
#        #the harvesting process is done moving forward with the convergence checks
#        inDictionary['Projector'].evaluate()
#        converged = inDictionary['Tester'].test(inDictionary['Projector'].output)
#        if not converged:
#          for freeSpot in xrange(jobHandler.howManyFreeSpots()):
#            if (jobHandler.getNumSubmitted() < int(self.maxNumberIteration)) and inDictionary['Sampler'].amIreadyToProvideAnInput():
#              newInput = inDictionary['Sampler'].generateInput(inDictionary['Model'],inDictionary['Input'],inDictionary['Projector'])
#              inDictionary['Model'].run(newInput,inDictionary['jobHandler'])
#        elif converged:
#          jobHandler.terminateAll()
#          break
#      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0:
#        break
#      time.sleep(self.sleepTime)
#    if 'ROM' in inDictionary.keys(): inDictionary['ROM'].close()
#
#
#
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
      if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
      if 'Plot' or 'Print' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(inDictionary)
    except AttributeError as ae: print("Error: "+repr(ae))    
    
  def localTakeAstepRun(self,inDictionary):
    for i in xrange(len(inDictionary['Output'])):
      #link the output to the database and construct the Data(s)
      # I have to change it
      if self.actionType[i] == 'HDF5-DATAS':
        inDictionary['Output'][i].addOutput(inDictionary['Input'][i])
        inDictionary['Output'][i].printCSV() # the check on the printing flag is internal
      else: inDictionary['Output'][i].addGroupDatas({'group':inDictionary['Input'][i].name},inDictionary['Input'][i])
    return

  def localAddInitParams(self,tempDict):
    #TODO implement
    pass

  def localInputAndChecks(self,xmlNode):
    #TODO implement
    pass
#
#
#
class RomTrainer(Step):
  '''This step type is used only to train a ROM
    @Input, DataBase (for example, HDF5)
  '''

  def addCurrentSetting(self,originalDict):
    Step.addCurrentSetting(self,originalDict)

  def localInitializeStep(self,inDictionary):
    '''The initialization step  for a ROM is copying the data out to the ROM (it is a copy not a reference) '''
    for i in xrange(len(inDictionary['Output'])):
      inDictionary['Output'][i].initializeTrain(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'][0])
    try: #try is used since files for the moment have no type attribute
      if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
      if 'Plot' or 'Print' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(inDictionary)
    except AttributeError as ae: print("Error: "+repr(ae))

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

  def localAddInitParams(self,tempDict):
    #TODO implement
    pass

  def localInputAndChecks(self,xmlNode):
    #TODO implement
    pass
#
#
#
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
   
  def localAddInitParams(self,tempDict):
    #TODO implement
    pass

  def localInputAndChecks(self,xmlNode):
    #TODO implement
    pass
#
#
#
__interFaceDict                      = {}
__interFaceDict['SingleRun'        ] = SingleRun
__interFaceDict['MultiRun'         ] = MultiRun
#__interFaceDict['SCRun'            ] = SCRun
__interFaceDict['Adaptive'         ] = Adaptive
__interFaceDict['InOutFromDataBase'] = InOutFromDataBase 
__interFaceDict['RomTrainer'       ] = RomTrainer
__interFaceDict['Plotting'         ] = PlottingStep
__base                               = 'Step'

def returnInstance(Type):
  return __interFaceDict[Type]()
  raise NameError('not known '+__base+' type '+Type)
  
