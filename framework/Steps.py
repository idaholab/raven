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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseType import BaseType
from utils    import metaclass_insert
import Models
#Internal Modules End--------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
class Step(metaclass_insert(abc.ABCMeta,BaseType)):
  '''
  This class implement one step of the simulation pattern.
  Usage:
  myInstance = Step()                                !Generate the instance
  myInstance.XMLread(xml.etree.ElementTree.Element)  !This method read the xml and perform all the needed checks
  myInstance.takeAstep()                             !This method perform the step
  
  --Internal chain [in square brackets methods that can be/must be overwritten]
  self.XMLread(xml)-->self._readMoreXML(xml)     -->[self._localInputAndChecks()]
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
  self._localInputAndChecks()             : used to specialize the xml reading and the checks
  self._localAddInitParams(tempDict)      : used to add the local parameters and values to be printed
  self._localInitializeStep(inDictionary) : called after this call the step should be able the accept the call self.takeAstep(inDictionary):
  self._localTakeAstepRun(inDictionary)   : this is where the step happens, after this call the output is ready
  '''
  
  def __init__(self):
    self.FIXME = False
    BaseType.__init__(self)
    self.parList    = []   # List of list [[role played in the step, class type, specialization, global name (user assigned by the input)]]
    self.sleepTime  = 0.1  # Waiting time before checking if a run is finished
    #If a step possess re-seeding instruction it is going to ask to the sampler to re-seed according
    #  re-seeding = a number to be used as a new seed
    #  re-seeding = 'continue' the use the already present random environment
    #If there is no instruction (self.initSeed = None) the sampler will reinitialize
    self.initSeed        = None
    self._knownAttribute += ['sleepTime','re-seeding','pauseAtEnd']

  def _readMoreXML(self,xmlNode):
    '''
    Handles the reading of all the XML describing the step
    Since step are not reused there will not be changes in the parameter describing the step after this reading
    @in xmlNode: xml.etree.ElementTree.Element containing the input to construct the step
    '''
    printString = 'For step of type {0:15} and name {1:15} the attribute {3:10} has been assigned to a not understandable value {2:10}'
    if self.FIXME: print('FIXME: move this tests to base class when it is ready for all the classes')
    if not set(xmlNode.attrib.keys()).issubset(set(self._knownAttribute)):
      raise IOError ('IN step of type {0:15} and name {1:15} there are unknown attributes {2:100}'.format(self.type,self.name,str(xmlNode.attrib.keys())))
    if 're-seeding' in xmlNode.attrib.keys():
      self.initSeed=xmlNode.attrib['re-seeding']
      if self.initSeed.lower()   == "continue": self.initSeed  = "continue"
      else: 
        try   : self.initSeed  = int(self.initSeed)
        except: raise IOError (printString.format(self.type,self.name,self.initSeed,'re-seeding'))
    if 'sleepTime' in xmlNode.attrib.keys():
      try: self.sleepTime = float(xmlNode.attrib['sleepTime'])
      except: raise IOError (printString.format(self.type,self.name,xmlNode.attrib['sleepTime'],'sleepTime'))
    for child in xmlNode                      : self.parList.append([child.tag,child.attrib['class'],child.attrib['type'],child.text])
    self.pauseEndStep = False
    if 'pauseAtEnd' in xmlNode.attrib.keys(): 
      if   xmlNode.attrib['pauseAtEnd'].lower() in ['yes','true','t']: self.pauseEndStep = True
      elif xmlNode.attrib['pauseAtEnd'].lower() in ['no','false','f']: self.pauseEndStep = False
      else: raise IOError (printString.format(self.type,self.name,xmlNode.attrib['pauseAtEnd'],'pauseAtEnd'))
    self._localInputAndChecks()
    if None in self.parList: raise IOError ('A problem was found in  the definition of the step '+str(self.name))

  @abc.abstractmethod
  def _localInputAndChecks(self):
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
      tempDict[List[0]] = 'Class: '+str(List[1])+' Type: '+str(List[2])+'  Global name: '+str(List[3])
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
    if self.debug: print('jobHandler initialized')
    self._localInitializeStep(inDictionary)

  
  @abc.abstractmethod
  def _localInitializeStep(self,inDictionary):
    '''
    This is the API for the local initialization of the children classes of step
    The inDictionary contains the for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
    The role of _localInitializeStep is to call the initialize method instance if needed
    Remember after each initialization to put:
    if self.debug: print('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
    or in general after any action put a communication conditional to the debug flag
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
        if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
          if inDictionary['Output'][i].type in ['OutStreamPlot']: inDictionary['Output'][i].endInstructions('interactive')
  
  def takeAstep(self,inDictionary):
    '''
    This should work for everybody just split the step in an initialization and the run itself
    inDictionary[role]=instance or list of instance
    '''
    print('\n-----Beginning initialization')
    self._initializeStep(inDictionary)
    if self.debug: print('-----Initialization done')
    if self.debug: print('-----Beginning run')
    self._localTakeAstepRun(inDictionary)
    if self.debug: print('-----Run finished')
    if self.debug: print('-----Closing the step')
    self._endStepActions(inDictionary)
    if self.debug: print('-----Step closed')
#
#
#
class SingleRun(Step):
  '''This is the step that will perform just one evaluation'''
  def _localInputAndChecks(self):
    if self.FIXME:print('FIXME: the mapping used in the model for checking the compatibility of usage should be more similar to self.parList to avoid the double mapping below')
    found     = 0
    rolesItem = []
    for index, parameter in enumerate(self.parList):
      if parameter[0]=='Model':
        found +=1
        modelIndex = index
      else: rolesItem.append(parameter[0])
    #test the presence of one and only one model
    if found > 1: raise IOError ('Only one model is allowed for the step named '+str(self.name))
    roles      = set(rolesItem)
    toBeTested = {}
    for role in roles: toBeTested[role]=[]
    for  myInput in self.parList:
      if myInput[0] in rolesItem: toBeTested[ myInput[0]].append({'class':myInput[1],'type':myInput[2]})
    #use the models static testing of roles compatibility
    for role in roles: Models.validate(self.parList[modelIndex][2], role, toBeTested[role])
    if 'Input'  not in roles: raise IOError ('It is not possible a run without an Input!!!')
    if 'Output' not in roles: raise IOError ('It is not possible a run without an Output!!!')
    
  def _localInitializeStep(self,inDictionary):
    '''this is the initialization for a generic step performing runs '''
    #Model initialization
    inDictionary['Model'].initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])
    if self.debug: print('for the role Model  the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Model'].type,inDictionary['Model'].name))
    #HDF5 initialization
    for i in range(len(inDictionary['Output'])):
      if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
        if 'HDF5' in inDictionary['Output'][i].type: inDictionary['Output'][i].initialize(self.name)
        elif inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: inDictionary['Output'][i].initialize(inDictionary)
        if self.debug: print('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Output'][i].type,inDictionary['Output'][i].name))
    
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
            if type(output).__name__ not in ['str','bytes','unicode']:
              if output.type not in ['OutStreamPlot','OutStreamPrint']: model.collectOutput(finishedJob,output)
              elif output.type in   ['OutStreamPlot','OutStreamPrint']: output.addOutput() 
            else: model.collectOutput(finishedJob,output)
        else: 
          print('the failed jobs are tracked in the JobHandler... we can retrieve and treat them separately. Andrea')
          print('a job failed... call the handler for this situation')   
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
    
  def _localInputAndChecks(self):
    SingleRun._localInputAndChecks(self)
    if 'Sampler' not in [item[0] for item in self.parList]: raise IOError ('It is not possible a multi-run without a sampler !!!')

  def _initializeSampler(self,inDictionary):
    inDictionary['Sampler'].initialize(**self._samplerInitDict)
    if self.debug: print('for the role of sampler the item of class '+inDictionary['Sampler'].type+' and name '+inDictionary['Sampler'].name+' has been initialized')
    if self.debug: print('Sampler initialization dictionary: '+str(self._samplerInitDict))

  def _localInitializeStep(self,inDictionary):
    SingleRun._localInitializeStep(self,inDictionary)
    self.conter                              = 0 
    self._samplerInitDict['externalSeeding'] = self.initSeed
    self._initializeSampler(inDictionary)
    #generate lambda function list to collect the output without checking the type
    self._outputCollectionLambda            = []
    for outIndex, output in enumerate(inDictionary['Output']):
      if output.type not in ['OutStreamPlot','OutStreamPrint']: self._outputCollectionLambda.append((lambda x: inDictionary['Model'].collectOutput(x[0],x[1]), outIndex))
      else: self._outputCollectionLambda.append((lambda x: x[1].addOutput(), outIndex))
    if 'SolutionExport' in inDictionary.keys():
      self._outputCollectionLambda[inDictionary['Output'].index(inDictionary['SolutionExport'])] = (lambda x: None),inDictionary['Output'].index(inDictionary['SolutionExport'])
    if self.debug:print('Generating input batch of size '+str(inDictionary['jobHandler'].runInfoDict['batchSize']))
    newInputs = inDictionary['Sampler'].generateInputBatch(inDictionary['Input'],inDictionary["Model"],inDictionary['jobHandler'].runInfoDict['batchSize'])
    for inputIndex, newInput in enumerate(newInputs):
      inDictionary["Model"].run(newInput,inDictionary['jobHandler'])
      if self.debug: print('Submitted input '+str(inputIndex+1))

  def _localTakeAstepRun(self,inDictionary):
    jobHandler = inDictionary['jobHandler']
    model      = inDictionary['Model'     ]
    inputs     = inDictionary['Input'     ]
    outputs    = inDictionary['Output'    ]
    sampler    = inDictionary['Sampler'   ]
    if 'TargetEvaluation' in inDictionary.keys(): targetOutput = inDictionary['TargetEvaluation']
    else                                        : targetOutput = None
    while True:
      finishedJobs = jobHandler.getFinished()
      for finishedJob in finishedJobs:
        self.counter +=1
        sampler.finalizeActualSampling(finishedJob,model,inputs)
        if finishedJob.getReturnCode() == 0: 
          for myLambda, outIndex in self._outputCollectionLambda:
            myLambda([finishedJob,outputs[outIndex]])
            if self.debug: print('Just collected output {0:2} of the input {1:6}'.format(outIndex+1,self.counter))
#      for _ in xrange(jobHandler.howManyFreeSpots()):
          if self.debug: print('Testing the sampler if it is ready to generate a new input')
          if sampler.amIreadyToProvideAnInput(inLastOutput=targetOutput):
            newInput =sampler.generateInput(model,inputs)
            model.run(newInput,jobHandler)
            if self.debug: print('New input generated')
        else: 
          print(' the job failed... call the handler for this situation... not yet implemented...')
          print("The JOBS that failed are tracked in the JobHandler... so we can retrieve and treat them separately. skipping here is Ok. Andrea")
      if jobHandler.isFinished() and len(jobHandler.getFinishedNoPop()) == 0: break
      time.sleep(self.sleepTime)
#
#
#
class Adaptive(MultiRun):
  '''this class implement one step of the simulation pattern' where several runs are needed in an adaptive scheme'''
  def _localInputAndChecks(self):
    '''we check coherence of Sampler, Functions and Solution Output'''
    #test sampler information:
    if self.FIXME: print('FIXME: all these test should be done at the beginning in a static fashion being careful since not all goes to the model')
    foundSampler     = False
    samplCounter     = 0
    foundTargEval    = False
    targEvalCounter  = 0
    solExportCounter = 0
    functionCounter  = 0
    foundFunction    = False
    ROMCounter       = 0
    #explanation new roles:
    #Function        : it takes in a datas and generate the value of the goal functions
    #TargetEvaluation: is the output datas that is used for the evaluation of the goal function. It has to be declared among the outputs
    #SolutionExport  : if declared it is used to export the location of the  goal functions = 0
    for role in self.parList:
      if   role[0] == 'Sampler':
        foundSampler    =True
        samplCounter   +=1
        if not(role[1]=='Samplers' and role[2]=='Adaptive'): raise Exception('The type of sampler used for the step '+str(self.name)+' is not coherent with and adaptive strategy')
      elif role[0] == 'TargetEvaluation':
        foundTargEval   = True
        targEvalCounter+=1
        if role[1]!='Datas'                               : raise Exception('The data chosen for the evaluation of the adaptive strategy is not compatible,  in the step '+self.name)
        if not(['Output']+role[1:] in self.parList[:])    : raise Exception('The data chosen for the evaluation of the adaptive strategy is not in the output list for step '+self.name)
      elif role[0] == 'SolutionExport'  :
        solExportCounter  +=1
        if role[1]!='Datas'                               : raise Exception('The data chosen for exporting the goal function solution is not compatible, in the step '+self.name)
      elif role[0] == 'Function'       :
        functionCounter+=1
        foundFunction   = True
        if role[1]!='Functions'                           : raise Exception('A class function is required as function in an adaptive step, in the step '+self.name)
      elif role[0] == 'ROM':
        ROMCounter+=1
        if not(role[1]=='Models' and role[2]=='ROM')       : raise Exception('The ROM could be only class=Models and type=ROM. It does not seems so in the step '+self.name)
    if foundSampler ==False: raise Exception('It is not possible to run an adaptive step without a sampler in step '           +self.name)
    if foundTargEval==False: raise Exception('It is not possible to run an adaptive step without a target output in step '     +self.name)
    if foundFunction==False: raise Exception('It is not possible to run an adaptive step without a proper function, in step '  +self.name)
    if samplCounter    >1  : raise Exception('More than one sampler found in step '                                            +self.name)
    if targEvalCounter >1  : raise Exception('More than one target defined for the adaptive sampler found in step '            +self.name)
    if solExportCounter>1  : raise Exception('More than one output to export the solution of the goal function, found in step '+self.name)
    if functionCounter >1  : raise Exception('More than one function defined in the step '                                     +self.name)
    if ROMCounter      >1  : raise Exception('More than one ROM defined in the step '                                          +self.name)
    
  def _localInitializeStep(self,inDictionary):
    '''this is the initialization for a generic step performing runs '''
    self._samplerInitDict['goalFunction'] = inDictionary['Function']
    if 'SolutionExport' in inDictionary.keys(): self._samplerInitDict['solutionExport']=inDictionary['SolutionExport']
    if 'ROM'            in inDictionary.keys(): self._samplerInitDict['ROM'           ]=inDictionary['ROM']
    MultiRun._localInitializeStep(self,inDictionary)
#
#
#
class IODataBase(Step):
  '''
    This step type is used only to extract or push information from/into a DataBase
    @Input, DataBase (for example, HDF5) or Datas
    @Output,Data(s) (for example, History) or DataBase
  '''
  def _localInitializeStep(self,inDictionary):
    # check if #inputs == #outputs
    if len(inDictionary['Input']) != len(inDictionary['Output']):
      # This condition is an error if the n Inputs > n Outputs. if the n Outputs > n Inputs, it is an error as well except in case the additional outputs are OutStreams => check for this
      if len(inDictionary['Input']) < len(inDictionary['Output']):
        noutputs = len(inDictionary['Output'])
        for i in xrange(len(inDictionary['Output'])): 
          if inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: noutputs -= 1
        if len(inDictionary['Input']) != noutputs: raise IOError('STEPS         : ERROR: In Step named ' + self.name + ', the number of Inputs != number of Outputs')
      else: raise IOError('STEPS         : ERROR -> In Step named ' + self.name + ', the number of Inputs != number of Outputs')
    self.actionType = []
    incnt = -1
    for i in range(len(inDictionary['Output'])):
      try: 
        if inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: 
          incnt -= 1
          continue
        else: incnt += 1
      except AttributeError as ae: pass
      if (inDictionary['Input'][incnt].type != 'HDF5'):
        if (not (inDictionary['Input'][incnt].type in ['TimePoint','TimePointSet','History','Histories'])): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts HDF5 as Input only. Got ' + inDictionary['Input'][incnt].type)
        else:
          if(inDictionary['Output'][i].type != 'HDF5'): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts ' + 'HDF5' + ' as Output only, when the Input is a Datas. Got ' + inDictionary['Output'][i].type)
          else: self.actionType.append('DATAS-HDF5')
      else:
        if (not (inDictionary['Output'][i].type in ['TimePoint','TimePointSet','History','Histories'])): raise IOError('STEPS         : ERROR: In Step named ' + self.name + '. This step accepts A Datas as Output only, when the Input is an HDF5. Got ' + inDictionary['Output'][i].type)
        else: self.actionType.append('HDF5-DATAS')
    databases = []
    for i in range(len(inDictionary['Output'])):
      if type(inDictionary['Output'][i]).__name__ not in ['str','bytes','unicode']:
        if 'HDF5' in inDictionary['Output'][i].type:
          if inDictionary['Output'][i].name not in databases:
            databases.append(inDictionary['Output'][i].name)
            inDictionary['Output'][i].initialize(self.name)
            if self.debug: print('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Output'][i].type,inDictionary['Output'][i].name))
        if inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']:
          inDictionary['Output'][i].initialize(inDictionary)   
          if self.debug: print('for the role Output the item of class {0:15} and name {1:15} has been initialized'.format(inDictionary['Output'][i].type,inDictionary['Output'][i].name))
      
  def _localTakeAstepRun(self,inDictionary):
    incnt = -1
    for i in range(len(inDictionary['Output'])):
      try: 
        if inDictionary['Output'][i].type in ['OutStreamPlot','OutStreamPrint']: 
          incnt -= 1
          continue
        else: incnt += 1
      except AttributeError as ae: pass
      if self.actionType[i] == 'HDF5-DATAS':
        inDictionary['Output'][i].addOutput(inDictionary['Input'][incnt])
      else: inDictionary['Output'][i].addGroupDatas({'group':inDictionary['Input'][incnt].name},inDictionary['Input'][incnt])
    for output in inDictionary['Output']:
      #try: 
      if output.type in ['OutStreamPlot','OutStreamPrint']: output.addOutput() 
      #except AttributeError as ae: print('STEPS         : ERROR -> ' + ae)
  
  def _localAddInitParams(self,tempDict):
    pass # no inputs

  def _localInputAndChecks(self):
    pass 
#
#
#
class RomTrainer(Step):
  '''This step type is used only to train a ROM
    @Input, DataBase (for example, HDF5)
  '''
  def _localInputAndChecks(self):
    if [item[0] for item in self.parList].count('Input')!=1: raise IOError('Only one Input and only one is allowed for a training step. Step name: '+str(self.name))
    if [item[0] for item in self.parList].count('Output')<1: raise IOError('At least one Output is need in a training step. Step name: '+str(self.name))
    for item in self.parList:
      if item[0]=='Output' and item[2]!='ROM': raise IOError('Only ROM output class are allowed in a training step. Step name: '+str(self.name))
  
  def _localAddInitParams(self,tempDict):
    del tempDict['Initial seed'] #this entry in not meaningful for a training step

  def _localInitializeStep(self,inDictionary): pass
        
  def _localTakeAstepRun(self,inDictionary):
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for ROM in inDictionary['Output']:
      ROM.train(inDictionary['Input'][0])
#
#
#
__interFaceDict                      = {}
__interFaceDict['SingleRun'        ] = SingleRun
__interFaceDict['MultiRun'         ] = MultiRun
__interFaceDict['Adaptive'         ] = Adaptive
__interFaceDict['IODataBase'       ] = IODataBase 
__interFaceDict['RomTrainer'       ] = RomTrainer
__base                               = 'Step'

def returnInstance(Type):
  return __interFaceDict[Type]()
  raise NameError('not known '+__base+' type '+Type)
  
