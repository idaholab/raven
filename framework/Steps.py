'''
Module containing the different type of step allowed
Step is called by simulation
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import time
import abc
import cPickle as pickle
#import pickle as cloudpickle
from serialization import cloudpickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import utils
import Models
from OutStreamManager import OutStreamManager
from DataObjects import Data
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
class Step(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  '''
  This class implement one step of the simulation pattern.
  Usage:
  myInstance = Step()                                !Generate the instance
  myInstance.XMLread(xml.etree.ElementTree.Element)  !This method read the xml and perform all the needed checks
  myInstance.takeAstep()                             !This method perform the step

  --Internal chain [in square brackets methods that can be/must be overwritten]
  self.XMLread(xml)-->self._readMoreXML(xml)     -->[self._localInputAndChecks(xmlNode)]
  self.takeAstep() -->self_initializeStep()      -->[self._localInitializeStep()]
                   -->[self._localTakeAstepRun()]
                   -->self._endStepActions()
  --Other external methods--
  myInstance.whoAreYou()                 -see BaseType class-
  myInstance.myInitializzationParams()   -see BaseType class-
  myInstance.myCurrentSetting()          -see BaseType class-
  myInstance.printMe()                   -see BaseType class-

  --Adding a new step subclass--
   **<MyClass> should inherit at least from Step or from another step already presents
   **DO NOT OVERRIDE any of the class method that are not starting with self.local*
   **ADD your class to the dictionary __InterfaceDict at the end of the module

  Overriding the following methods overriding unless you inherit from one of the already existing methods:
  self._localInputAndChecks(xmlNode)      : used to specialize the xml reading and the checks
  self._localAddInitParams(tempDict)      : used to add the local parameters and values to be printed
  self._localInitializeStep(inDictionary) : called after this call the step should be able the accept the call self.takeAstep(inDictionary):
  self._localTakeAstepRun(inDictionary)   : this is where the step happens, after this call the output is ready
  '''

  def __init__(self):
    BaseType.__init__(self)
    self.parList    = []   # List of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.sleepTime  = 0.005  # Waiting time before checking if a run is finished
    #If a step possess re-seeding instruction it is going to ask to the sampler to re-seed according
    #  re-seeding = a number to be used as a new seed
    #  re-seeding = 'continue' the use the already present random environment
    #If there is no instruction (self.initSeed = None) the sampler will reinitialize
    self.initSeed        = None
    self._knownAttribute += ['sleepTime','re-seeding','pauseAtEnd','fromDirectory']
    self._excludeFromModelValidation = ['SolutionExport']
    self.printTag = 'STEPS'

  def _readMoreXML(self,xmlNode):
    '''
    Handles the reading of all the XML describing the step
    Since step are not reused there will not be changes in the parameter describing the step after this reading
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    printString = 'For step of type {0:15} and name {1:15} the attribute {3:10} has been assigned to a not understandable value {2:10}'
    self.raiseADebug('move this tests to base class when it is ready for all the classes')
    if not set(xmlNode.attrib.keys()).issubset(set(self._knownAttribute)):
      self.raiseAnError(IOError,'In step of type {0:15} and name {1:15} there are unknown attributes {2:100}'.format(self.type,self.name,str(xmlNode.attrib.keys())))
    if 're-seeding' in xmlNode.attrib.keys():
      self.initSeed=xmlNode.attrib['re-seeding']
      if self.initSeed.lower()   == "continue": self.initSeed  = "continue"
      else:
        try   : self.initSeed  = int(self.initSeed)
        except: self.raiseAnError(IOError,printString.format(self.type,self.name,self.initSeed,'re-seeding'))
    if 'sleepTime' in xmlNode.attrib.keys():
      try: self.sleepTime = float(xmlNode.attrib['sleepTime'])
      except: self.raiseAnError(IOError,printString.format(self.type,self.name,xmlNode.attrib['sleepTime'],'sleepTime'))
    for child in xmlNode                      : self.parList.append([child.tag,child.attrib['class'],child.attrib['type'],child.text])
    self.pauseEndStep = False
    if 'pauseAtEnd' in xmlNode.attrib.keys():
      if   xmlNode.attrib['pauseAtEnd'].lower() in utils.stringsThatMeanTrue(): self.pauseEndStep = True
      elif xmlNode.attrib['pauseAtEnd'].lower() in utils.stringsThatMeanFalse(): self.pauseEndStep = False
      else: self.raiseAnError(IOError,printString.format(self.type,self.name,xmlNode.attrib['pauseAtEnd'],'pauseAtEnd'))
    self._localInputAndChecks(xmlNode)
    if None in self.parList: self.raiseAnError(IOError,'A problem was found in  the definition of the step '+str(self.name))

  @abc.abstractmethod
  def _localInputAndChecks(self,xmlNode):
    '''
    Place here specialized reading, input consistency check and
    initialization of what will not change during the whole life of the object
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    pass

  def addInitParams(self,tempDict):
    '''Export to tempDict the information that will stay constant during the existence of the instance of this class. Overloaded from BaseType'''
    tempDict['Sleep time'  ] = str(self.sleepTime)
    tempDict['Initial seed'] = str(self.initSeed)
    for List in self.parList:
      tempDict[List[0]] = 'Class: '+str(List[1]) +' Type: '+str(List[2]) + '  Global name: '+str(List[3])
    self._localAddInitParams(tempDict)

  @abc.abstractmethod
  def _localAddInitParams(self,tempDict):
    '''
    Place here a specialization of the exporting of what in the step is added to the initial parameters
    the printing format of tempDict is key: tempDict[key]
    '''
    pass

  def _initializeStep(self,inDictionary):
    '''the job handler is restarted and re-seeding action are performed'''
    inDictionary['jobHandler'].startingNewStep()
    self.raiseADebug('jobHandler initialized')
    self._localInitializeStep(inDictionary)

  @abc.abstractmethod
  def _localInitializeStep(self,inDictionary):
    '''
    This is the API for the local initialization of the children classes of step
    The inDictionary contains the for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
    The role of _localInitializeStep is to call the initialize method instance if needed
    Remember after each initialization to put:
    self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
    '''
    pass

  @abc.abstractmethod
  def _localTakeAstepRun(self,inDictionary):
    '''this is the API for the local run of a step for the children classes'''
    pass

  def _endStepActions(self,inDictionary):
    '''This method is intended for performing actions at the end of a step'''
    if self.pauseEndStep:
      for i in range(len(inDictionary['Output'])):
        #if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
        if inDictionary['Output'][i].type in ['OutStreamPlot']: inDictionary['Output'][i].endInstructions('interactive')

  def takeAstep(self,inDictionary):
    '''
    This should work for everybody just split the step in an initialization and the run itself
    inDictionary[role]=instance or list of instance
    '''
    self.raiseAMessage('***  Beginning initialization ***')
    self._initializeStep(inDictionary)
    self.raiseAMessage('***    Initialization done    ***')
    self.raiseAMessage('***       Beginning run       ***')
    self._localTakeAstepRun(inDictionary)
    self.raiseAMessage('***       Run finished        ***')
    self.raiseAMessage('***     Closing the step      ***')
    self._endStepActions(inDictionary)
    self.raiseAMessage('***        Step closed        ***')
#
#
#
class SingleRun(Step):
  '''This is the step that will perform just one evaluation'''
  def __init__(self):
    Step.__init__(self)
    self.printTag = 'STEP SINGLERUN'

  def _localInputAndChecks(self,xmlNode):
    self.raiseADebug('the mapping used in the model for checking the compatibility of usage should be more similar to self.parList to avoid the double mapping below','FIXME')
    found     = 0
    rolesItem = []
    for index, parameter in enumerate(self.parList):
      if parameter[0]=='Model':
        found +=1
        modelIndex = index
      else: rolesItem.append(parameter[0])
    #test the presence of one and only one model
    if found > 1: self.raiseAnError(IOError,'Only one model is allowed for the step named '+str(self.name))
    elif found == 0: self.raiseAnError(IOError,'No model has been found for the step named '+str(self.name))
    roles      = set(rolesItem)
    toBeTested = {}
    for role in roles: toBeTested[role]=[]
    for  myInput in self.parList:
      if myInput[0] in rolesItem: toBeTested[ myInput[0]].append({'class':myInput[1],'type':myInput[2]})
    #use the models static testing of roles compatibility
    for role in roles:
      if role not in self._excludeFromModelValidation:
        Models.validate(self.parList[modelIndex][2], role, toBeTested[role],self)
    self.raiseADebug('reactivate check on Input as soon as loadCsv gets out from the PostProcessor models!')
    if 'Output' not in roles: self.raiseAnError(IOError,'It is not possible a run without an Output!')

  def _localInitializeStep(self,inDictionary):
    '''this is the initialization for a generic step performing runs '''
    #Model initialization
    modelInitDict={}
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'],{})
    self.raiseADebug('for the role Model  the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Model'].type,inDictionary['Model'].name))
    #HDF5 initialization
    for i in range(len(inDictionary['Output'])):
      #if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
      if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
      elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: inDictionary['Output'][i].initialize(inDictionary)
      self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Output'][i].type,inDictionary['Output'][i].name))

  def _localTakeAstepRun(self,inDictionary):
    '''main driver for a step'''
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]
    model.run(inputs,jobHandler)
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        if finishedJob.getReturnCode() == 0:
          # if the return code is > 0 => means the system code crashed... we do not want to make the statistics poor => we discard this run
          for output in outputs:
            #if type(output).__name__ not in ['str','bytes','unicode']:
            if output.type not in ['OutStreamPlot','OutStreamPrint']: model.collectOutput(finishedJob,output)
            elif output.type in   ['OutStreamPlot','OutStreamPrint']: output.addOutput()
            #else: model.collectOutput(finishedJob,output)
        else:
          self.raiseADebug('the failed jobs are tracked in the JobHandler... we can retrieve and treat them separately. Andrea')
          self.raiseADebug('a job failed... call the handler for this situation')
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0: break
      time.sleep(self.sleepTime)
  def _localAddInitParams(self,tempDict): pass
#
#
#
class MultiRun(SingleRun):
  '''this class implement one step of the simulation pattern' where several runs are needed without being adaptive'''
  def __init__(self):
    SingleRun.__init__(self)
    self._samplerInitDict = {} #this is a dictionary that gets sent as key-worded list to the initialization of the sampler
    self.counter          = 0  #just an handy counter of the runs already performed
    self.printTag = 'STEP MULTIRUN'

  def _localInputAndChecks(self,xmlNode):
    SingleRun._localInputAndChecks(self,xmlNode)
    if 'Sampler' not in [item[0] for item in self.parList]: self.raiseAnError(IOError,'It is not possible a multi-run without a sampler!')

  def _initializeSampler(self,inDictionary):
    if 'SolutionExport' in inDictionary.keys(): self._samplerInitDict['solutionExport']=inDictionary['SolutionExport']

    inDictionary['Sampler'].initialize(**self._samplerInitDict)
    self.raiseADebug('for the role of sampler the item of class '+inDictionary['Sampler'].type+' and name '+inDictionary['Sampler'].name+' has been initialized')
    self.raiseADebug('Sampler initialization dictionary: '+str(self._samplerInitDict))

  def _localInitializeStep(self,inDictionary):
    SingleRun._localInitializeStep(self,inDictionary)
    self.conter                              = 0
    self._samplerInitDict['externalSeeding'] = self.initSeed
    self._initializeSampler(inDictionary)
    #generate lambda function list to collect the output without checking the type
    self._outputCollectionLambda            = []
    for outIndex, output in enumerate(inDictionary['Output']):
      if output.type not in ['OutStreamPlot','OutStreamPrint']:
        if 'SolutionExport' in inDictionary.keys() and output.name == inDictionary['SolutionExport'].name: self._outputCollectionLambda.append((lambda x:None, outIndex))
        else: self._outputCollectionLambda.append( (lambda x: inDictionary['Model'].collectOutput(x[0],x[1]), outIndex) )
      else: self._outputCollectionLambda.append((lambda x: x[1].addOutput(), outIndex))
    self.raiseADebug('Generating input batch of size '+str(inDictionary['jobHandler'].runInfoDict['batchSize']))
    for inputIndex in range(inDictionary['jobHandler'].runInfoDict['batchSize']):
      if inDictionary['Sampler'].amIreadyToProvideAnInput():
        inDictionary["Model"].run(inDictionary['Sampler'].generateInput(inDictionary["Model"],inDictionary['Input']),inDictionary['jobHandler'])
        self.raiseADebug('Submitted input '+str(inputIndex+1))

  def _localTakeAstepRun(self,inDictionary):
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]
    sampler    = inDictionary['Sampler'   ]
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        self.counter +=1
        sampler.finalizeActualSampling(finishedJob,model,inputs)
        if finishedJob.getReturnCode() == 0:
          for myLambda, outIndex in self._outputCollectionLambda:
            myLambda([finishedJob,outputs[outIndex]])
            self.raiseADebug('Just collected output {0:2} of the input {1:6}'.format(outIndex+1,self.counter))
        else:
          self.raiseADebug('the job failed... call the handler for this situation... not yet implemented...')
          self.raiseADebug('the JOBS that failed are tracked in the JobHandler... hence, we can retrieve and treat them separately. skipping here is Ok. Andrea')
        for _ in range(min(jobHandler.howManyFreeSpots(),sampler.endJobRunnable())): # put back this loop (do not take it away again. it is NEEDED for NOT-POINT samplers(aka DET)). Andrea
          self.raiseADebug('Testing the sampler if it is ready to generate a new input')
          #if sampler.amIreadyToProvideAnInput(inLastOutput=self.targetOutput):
          if sampler.amIreadyToProvideAnInput():
            newInput =sampler.generateInput(model,inputs)
            model.run(newInput,jobHandler)
            self.raiseADebug('New input generated')
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0: break
      time.sleep(self.sleepTime)
#
#
#
# class Adaptive(MultiRun):
#   '''this class implement one step of the simulation pattern' where several runs are needed in an adaptive scheme'''
#   def __init__(self):
#     MultiRun.__init__(self)
#     self.printTag = utils.returnPrintTag('STEP ADAPTIVE')
#   def _localInputAndChecks(self,xmlNode):
#     '''we check coherence of Sampler, Functions and Solution Output'''
#     #test sampler information:
#     foundSampler     = False
#     samplCounter     = 0
#     foundTargEval    = False
#     targEvalCounter  = 0
#     solExportCounter = 0
#     functionCounter  = 0
#     foundFunction    = False
#     ROMCounter       = 0
#     #explanation new roles:
#     #Function        : it takes in a dataObjects and generate the value of the goal functions
#     #TargetEvaluation: is the output dataObjects that is used for the evaluation of the goal function. It has to be declared among the outputs
#     #SolutionExport  : if declared it is used to export the location of the  goal functions = 0
#     for role in self.parList:
#       if   role[0] == 'Sampler':
#         foundSampler    =True
#         samplCounter   +=1
#         if not(role[1]=='Samplers' and role[2] in ['Adaptive','AdaptiveDynamicEventTree']): risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '->  The type of sampler used for the step '+str(self.name)+' is not coherent with and adaptive strategy')
#       elif role[0] == 'TargetEvaluation':
#         foundTargEval   = True
#         targEvalCounter+=1
#         if role[1]!='DataObjects'                               : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The data chosen for the evaluation of the adaptive strategy is not compatible,  in the step '+self.name)
#         if not(['Output']+role[1:] in self.parList[:])    : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The data chosen for the evaluation of the adaptive strategy is not in the output list for step '+self.name)
#       elif role[0] == 'SolutionExport'  :
#         solExportCounter  +=1
#         if role[1]!='DataObjects'                               : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The data chosen for exporting the goal function solution is not compatible, in the step '+self.name)
#       elif role[0] == 'Function'       :
#         functionCounter+=1
#         foundFunction   = True
#         if role[1]!='Functions'                           : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> A class function is required as function in an adaptive step, in the step '+self.name)
#       elif role[0] == 'ROM':
#         ROMCounter+=1
#         if not(role[1]=='Models' and role[2]=='ROM')       : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The ROM could be only class=Models and type=ROM. It does not seems so in the step '+self.name)
#     if foundSampler ==False: risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> It is not possible to run an adaptive step without a sampler in step '           +self.name)
#     if foundTargEval==False: risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> It is not possible to run an adaptive step without a target output in step '     +self.name)
#     if foundFunction==False: risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> It is not possible to run an adaptive step without a proper function, in step '  +self.name)
#     if samplCounter    >1  : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> More than one sampler found in step '                                            +self.name)
#     if targEvalCounter >1  : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> More than one target defined for the adaptive sampler found in step '            +self.name)
#     if solExportCounter>1  : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> More than one output to export the solution of the goal function, found in step '+self.name)
#     if functionCounter >1  : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> More than one function defined in the step '                                     +self.name)
#     if ROMCounter      >1  : risea Exception(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> More than one ROM defined in the step '                                          +self.name)
#
#   def _localInitializeStep(self,inDictionary):
#     '''this is the initialization for a generic step performing runs '''
#     #self._samplerInitDict['goalFunction'] = inDictionary['Function']
#     if 'SolutionExport' in inDictionary.keys(): self._samplerInitDict['solutionExport']=inDictionary['SolutionExport']
#     #if 'ROM'            in inDictionary.keys():
#       #self._samplerInitDict['ROM']=inDictionary['ROM']
#       #self._samplerInitDict['ROM'].reset()
#     MultiRun._localInitializeStep(self,inDictionary)
#
#
#
class RomTrainer(Step):
  '''This step type is used only to train a ROM
    @Input, Database (for example, HDF5)
  '''
  def __init__(self):
    Step.__init__(self)
    self.printTag = 'STEP ROM TRAINER'

  def _localInputAndChecks(self,xmlNode):
    if [item[0] for item in self.parList].count('Input')!=1: self.raiseAnError(IOError,'Only one Input and only one is allowed for a training step. Step name: '+str(self.name))
    if [item[0] for item in self.parList].count('Output')<1: self.raiseAnError(IOError,'At least one Output is need in a training step. Step name: '+str(self.name))
    for item in self.parList:
      if item[0]=='Output' and item[2] not in ['ROM']:
        self.raiseAnError(IOError,'Only ROM output class are allowed in a training step. Step name: '+str(self.name))

  def _localAddInitParams(self,tempDict):
    del tempDict['Initial seed'] #this entry in not meaningful for a training step

  def _localInitializeStep(self,inDictionary): pass

  def _localTakeAstepRun(self,inDictionary):
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for ROM in inDictionary['Output']: ROM.train(inDictionary['Input'][0])
#
#
#
# class PostProcess(SingleRun):
#   '''this class implements a PostProcessing (PP) strategy. The driver of this PP action is the model that MUST be of type FILTER'''
#   def __init__(self):
#     SingleRun.__init__(self)
#     self.foundFunction   = False
#     self.functionCounter = 0
#     self.ROMCounter      = 0
#     self.foundROM        = False
#     self.printTag = utils.returnPrintTag('STEP POSTPROCESS')
#
#   def _localInputAndChecks(self,xmlNode):
#     found     = 0
#     rolesItem = []
#     for index, parameter in enumerate(self.parList):
#       if parameter[0]=='Model':
#         found +=1
#         modelIndex = index
#       else: rolesItem.append(parameter[0])
#     #test the presence of one and only one model
#     if found > 1: risea IOError (self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> Only one model is allowed for the step named '+str(self.name))
#     elif found == 0: risea IOError (self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> No model has been found for the step named '+str(self.name))
#     roles      = set(rolesItem)
#     toBeTested = {}
#     for role in roles: toBeTested[role]=[]
#     for  myInput in self.parList:
#       if myInput[0] in rolesItem: toBeTested[ myInput[0]].append({'class':myInput[1],'type':myInput[2]})
#     #use the models static testing of roles compatibility
#     for role in roles: Models.validate(self.parList[modelIndex][2], role, toBeTested[role])
#     #SingleRun._localInputAndChecks(self,xmlNode)
#     for role in self.parList:
#       if role[0] == 'Function':
#         self.functionCounter+=1
#         self.foundFunction   = True
#         if role[1]!='Functions': risea IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The optional function must be of class "Functions", in step ' + self.name)
#       elif role[0] == 'Model' and role[1] == 'Models':
#         if role[2] != 'PostProcessor' : risea IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The required model in "PostProcess" step must be of type PostProcessor, in step ' + self.name)
#       elif role[0] == 'ROM' and role[1] == 'Models':
#         self.ROMCounter+=1
#         self.foundROM   = True
#         if role[2] != 'ROM' : risea IOError(self.printTag+': ' +utils.returnPrintPostTag('ERROR') + '-> The optional ROM in "PostProcess" step must be of type ROM, in step ' + self.name)
#
#   def _localInitializeStep(self,inDictionary):
#     functionExt = None
#     ROMExt      = None
#     if self.foundFunction: functionExt = inDictionary['Function']
#     if self.foundROM: ROMExt = inDictionary['ROM']
#     initDict = {'externalFunction':functionExt,'ROM':ROMExt}
#     inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'],initDict)
#     #HDF5 initialization
#     for i in range(len(inDictionary['Output'])):
#       if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
#         if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
#         elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: inDictionary['Output'][i].initialize(inDictionary)
#
#   def _localTakeAstepRun(self,inDictionary):
#     SingleRun._localTakeAstepRun(self, inDictionary)
#
#
#
class IOStep(Step):
  '''
  This step is used to extract or push information from/into a Database,
  or from a directory, or print out the data to an OutStream
  '''
  def __init__(self):
    Step.__init__(self)
    self.printTag = 'STEP IOCOMBINED'
    self.fromDirectory = None

  def __getOutputs(self, inDictionary):
    outputs         = []
    for out in inDictionary['Output']:
      if not isinstance(out,OutStreamManager): outputs.append(out)
    return outputs

  def _localInitializeStep(self,inDictionary):
    # check if #inputs == #outputs
    # collect the outputs without outstreams
    outputs         = self.__getOutputs(inDictionary)
    databases       = set()
    self.actionType = []
    if len(inDictionary['Input']) != len(outputs) and len(outputs) > 0: self.raiseAnError(IOError,'In Step named ' + self.name + ', the number of Inputs != number of Outputs and the number of Outputs > 0')
    #determine if this is a DATAS->HDF5, HDF5->DATAS or both.
    # also determine if this is an invalid combination
    for i in range(len(outputs)):
      if inDictionary['Input'][i].type == 'HDF5':
        if isinstance(outputs[i],Data): self.actionType.append('HDF5-dataObjects')
        else: utils.raiseAnError(IOError,self,'In Step named ' + self.name + '. This step accepts A DataObjects as Output only, when the Input is an HDF5. Got ' + inDictionary['Output'][i].type)
      elif  isinstance(inDictionary['Input'][i],Data):
        if outputs[i].type == 'HDF5': self.actionType.append('dataObjects-HDF5')
        else: utils.raiseAnError(IOError,self,'In Step named ' + self.name + '. This step accepts ' + 'HDF5' + ' as Output only, when the Input is a DataObjects. Got ' + inDictionary['Output'][i].type)
      elif isinstance(inDictionary['Input'][i],Models.ROM):
        if outputs[i].type == 'FileObject': self.actionType.append('ROM-FILES')
        else: utils.raiseAnError(IOError,self,'In Step named ' + self.name + '. This step accepts A Files as Output only, when the Input is a ROM. Got ' + inDictionary['Output'][i].type)
      elif inDictionary['Input'][i].type == 'FileObject':
        if isinstance(outputs[i],Models.ROM): self.actionType.append('FILES-ROM')
        else: utils.raiseAnError(IOError,self,'In Step named ' + self.name + '. This step accepts A ROM as Output only, when the Input is a Files. Got ' + inDictionary['Output'][i].type)
      else: utils.raiseAnError(IOError,self,'In Step named ' + self.name + '. This step accepts DataObjects, HDF5, ROM and Files as Input only. Got ' + inDictionary['Input'][i].type)

    #Initialize all the HDF5 outputs.
    for i in range(len(outputs)):
      #if type(outputs[i]).__name__ not in ['str','bytes','unicode']:
      if 'HDF5' in inDictionary['Output'][i].type:
        if outputs[i].name not in databases:
          databases.add(outputs[i].name)
          outputs[i].initialize(self.name)
          self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(outputs[i].type,outputs[i].name))

    #if have a fromDirectory and are a dataObjects-*, need to load data
    if self.fromDirectory:
      for i in range(len(inDictionary['Input'])):
        if self.actionType[i].startswith('dataObjects-'):
          inInput = inDictionary['Input'][i]
          inInput.loadXML_CSV(self.fromDirectory)

    #Initialize all the OutStreamPrint and OutStreamPlot outputs
    for output in inDictionary['Output']:
      if type(output).__name__ in ['OutStreamPrint','OutStreamPlot']:
        output.initialize(inDictionary)
        self.raiseADebug('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(output.type,output.name))

  def _localTakeAstepRun(self,inDictionary):
    outputs = self.__getOutputs(inDictionary)
    for i in range(len(outputs)):
      if self.actionType[i] == 'HDF5-dataObjects':
        #inDictionary['Input'][i] is HDF5, outputs[i] is a DataObjects
        outputs[i].addOutput(inDictionary['Input'][i])
      elif self.actionType[i] == 'dataObjects-HDF5':
        #inDictionary['Input'][i] is a dataObjects, outputs[i] is HDF5
        outputs[i].addGroupDataObjects({'group':inDictionary['Input'][i].name},inDictionary['Input'][i])
      elif self.actionType[i] == 'ROM-FILES':
        #inDictionary['Input'][i] is a ROM, outputs[i] is Files
        fileobj = open(outputs[i],'wb+')
        cloudpickle.dump(inDictionary['Input'][i],fileobj)
        fileobj.close()
      elif self.actionType[i] == 'FILES-ROM':
        #inDictionary['Input'][i] is a Files, outputs[i] is ROM
        fileobj = open(inDictionary['Input'][i],'rb+')
        unpickledObj = pickle.load(fileobj)
        outputs[i].train(unpickledObj)
        fileobj.close()
      else:
        self.raiseAnError(IOError,"Unknown action type "+self.actionType[i])
    for output in inDictionary['Output']:
      if output.type in ['OutStreamPrint','OutStreamPlot']: output.addOutput()

  def _localAddInitParams(self,tempDict):
    return tempDict # no inputs

  def _localInputAndChecks(self,xmlNode):
    if 'fromDirectory' in xmlNode.attrib.keys():
      self.fromDirectory = xmlNode.attrib['fromDirectory']

#
#
#
__interFaceDict                      = {}
__interFaceDict['SingleRun'        ] = SingleRun
__interFaceDict['MultiRun'         ] = MultiRun
#__interFaceDict['Adaptive'         ] = Adaptive
__interFaceDict['IOStep'           ] = IOStep
__interFaceDict['IODatabase'       ] = IOStep
__interFaceDict['RomTrainer'       ] = RomTrainer
__interFaceDict['PostProcess'      ] = SingleRun
__interFaceDict['OutStreamStep'    ] = IOStep
__base                               = 'Step'

def returnInstance(Type,caller):
  return __interFaceDict[Type]()
  caller.raiseAnError(NameError,'not known '+__base+' type '+Type)
