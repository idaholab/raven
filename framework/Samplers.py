'''
Module where the base class and the specialization of different type of sampler are
'''
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import abc
import numpy as np
from scipy import spatial
import xml.etree.ElementTree as ET
import TreeStructure as ETS
from BaseType import BaseType
import Distributions
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import metaclass_insert
#Internal Modules End--------------------------------------------------------------------------------

class Sampler(metaclass_insert(abc.ABCMeta,BaseType)):
  '''
  This is the base class for samplers
  Samplers own the sampling strategy (Type) and they generate the
  input values using the associate distribution. They do not have distributions inside!!!!
  
  --Instance--
  myInstance = Sampler()
  myInstance.XMLread(xml.etree.ElementTree.Element)  This method generates all the information that will be permanent for the object during the simulation
  
  --usage--
  myInstance = Sampler()
  myInstance.XMLread(xml.etree.ElementTree.Element)  This method generate all permanent information of the object from <Simulation>
  myInstance.generateDistributions(dict)             Here the seed for the random engine is started and the distributions are supplied to the sampler and 
                                                     initialized. The method is called come from <Simulation> since it is the only one possess all the distributions.
  myInstance.initialize()                            This method is called from the <Step> before the Step process start. In the base class it reset the counter to 0
  myInstance.amIreadyToProvideAnInput                Requested from <Step> used to verify that the sampler is available to generate a new input
  myInstance.generateInput(self,model,oldInput)      Requested from <Step> to generate a new input. Generate the new values and request to model to modify according the input and returning it back

  --Other inherited methods--
  myInstance.whoAreYou()                            -see BaseType class-
  myInstance.myInitializzationParams()              -see BaseType class-
  myInstance.myCurrentSetting()                     -see BaseType class-

  --Adding a new Sampler subclass--  
  <MyClass> should inherit at least from Sampler or from another step already presents

  DO NOT OVERRIDE any of the class method that are not starting with self.local*
  
  ADD your class to the dictionary __InterfaceDict at the end of the module

  The following method overriding is MANDATORY:  
  self.localGenerateInput(model,oldInput)  : this is where the step happens, after this call the output is ready

  the following methods could be overrode:
  self.localInputAndChecks(xmlNode)
  self.localAddInitParams(tempDict)
  self.localAddCurrentSetting(tempDict)
  self.localInitialize()
  self.localStillReady(ready)
  self.localFinalizeActualSampling(jobObject,model,myInput)
  '''

  def __init__(self):
    BaseType.__init__(self) 
    self.counter      = 0           # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.limit        = sys.maxsize # maximum number of Samples (for example, Monte Carlo = Number of Histories to run, DET = Unlimited)
    self.toBeSampled  = {}          # Sampling mapping dictionary {'Variable Name':['type of distribution to be used', 'name of the distribution']}
    self.distDict     = {}          # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.values       = {}          # for each variable the current value {'var name':value}
    self.inputInfo    = {}          # depending on the sampler several different type of keywarded information could be present only one is mandatory, see below 
    self.initSeed     = None        # if not provided the seed is randomly generated at the istanciation of the sampler, the step can override the seed by sending in another seed
    self.inputInfo['SampledVars'  ] = self.values #this is the location where to get the values of the sampled variables
    self.inputInfo['SampledVarsPb'] = {}          #this is the location where to get the probability of the sampled variables

  def _readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    The text i supposed to contain the info where and which variable to change.
    In case of a code the syntax is specified by the code interface itself
    '''
    try            : self.initSeed = int(xmlNode.attrib['initial_seed'])
    except KeyError: self.initSeed = Distributions.randomIntegers(0,2**31)
    if 'limit' in xmlNode.attrib.keys():
      try: self.limit = int(xmlNode.attrib['limit'])
      except:
        IOError ('reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    for child in xmlNode:
      for childChild in child:
        if childChild.tag =='distribution': 
          if child.tag == 'Distribution':
            #Add <distribution> to name so we know it is not the direct variable
            self.toBeSampled["<distribution>"+child.attrib['name']] = [childChild.attrib['type'],childChild.text]
          elif child.tag == 'variable': self.toBeSampled[child.attrib['name']] = [childChild.attrib['type'],childChild.text]
          else: raise IOError('SAMPLER       : ERROR -> Unknown tag '+child.tag+' .Available are: Distribution and variable!')
    self.localInputAndChecks(xmlNode)

  def localInputAndChecks(self,xmlNode):
    '''place here the additional reading, remember to add initial parameters in the method localAddInitParams'''
    pass
  
  def addInitParams(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is permanent in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary. No information about values that change during the simulation are allowed
    @ In/Out tempDict: {'attribute name':value}
    '''
    for variable in self.toBeSampled.items():
      tempDict[variable[0]+' is sampled using the distribution'] = variable[1][0]+' - '+variable[1][1]
    tempDict['limit' ]        = self.limit
    tempDict['initial seed' ] = self.initSeed
    self.localAddInitParams(tempDict)
    
  def localAddInitParams(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''

  def addCurrentSetting(self,tempDict):
    '''
    This function is called from the base class to print some of the information inside the class.
    Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    tempDict['counter' ]      = self.counter
    tempDict['initial seed' ] = self.initSeed
    for key in self.inputInfo:
      if key!='SampledVars': tempDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys(): tempDict['Variable: '+var+' has value'] = tempDict[key][var]
    self.localAddCurrentSetting(tempDict)

  def localAddCurrentSetting(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''
    pass

  def generateDistributions(self,availableDist):
    '''
    here the needed distribution are made available to the step as also the initializzation 
    of the seeding (the siding could be overriden by the step by calling the initiaize method
    @in availableDist: {'distribution name':instance}
    '''
    if self.initSeed != None:
      Distributions.randomSeed(self.initSeed)
    for key in self.toBeSampled.keys(): self.distDict[key] = availableDist[self.toBeSampled[key][1]]
    
  def initialize(self,externalSeeding=None,solutionExport=None,goalFunction=None,ROM=None):
    '''
    This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
    @in solutionExport: in goal oriented sampling (a.k.a. adaptive sampling this is where the space/point satisfying the constrain)
    @in goalFunction:   in goal oriented sampling this is the function to be used
    '''
    self.counter = 0
    if   externalSeeding==None: Distributions.randomSeed(self.initSeed)            #use the sampler initializzation seed
    elif externalSeeding=='continue': pass                                          #in this case the random sequence wants to be preserved
    else                            : Distributions.randomSeed(externalSeeding)    #the external seeding is used
    for key in self.toBeSampled.keys(): self.distDict[key].initializeDistribution() #now we can initialize the distributions
    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport!=None:
      if   goalFunction==None: raise Exception('not consistent call to the smapler.initialize since the SolutionExport is provided but not the goalFunction')
      else                   : self.localInitialize(solutionExport=solutionExport,goalFunction=goalFunction,ROM=ROM)
    else                     : self.localInitialize()
    
  def localInitialize(self):
    '''
    use this function to add initialization features to the derived class
    it is call at the beginning of each step
    '''
    pass
    
  def amIreadyToProvideAnInput(self,inLastOutput=None):
    '''
    This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
    This method act as a semaphore for generating a new input.
    Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
    @in lastOutput is used for adaptive methodologies when the the last output is used to decide if convergence is achived
    @return Boolean
    '''
    if(self.counter < self.limit): ready = True
    else                         : ready = False
    ready = self.localStillReady(ready,lastOutput=inLastOutput)
    return ready
  
  def localStillReady(self,ready,lastOutput=None):
    '''Use this function to change the ready status'''
    return ready
    
  def generateInput(self,model,oldInput):
    '''
    This method have to be overwrote to provide the specialization for the specific sampler
    The model instance in might be needed since, especially for external codes, 
    only the code interface possesses the dictionary for reading the variable definition syntax
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    @return     : [] containing the new inputs -in reality it is the model that return this the Sampler generate the value to be placed in the intput the model 
    '''
    self.counter +=1                              #since we are creating the input for the next run we increase the counter
    self.inputInfo['prefix'] = str(self.counter)
    self.localGenerateInput(model,oldInput)
    return model.createNewInput(oldInput,self.type,**self.inputInfo)

  @abc.abstractmethod
  def localGenerateInput(self,model,oldInput):
    '''
    This class need to be overwritten since it is here that the magic of the sampler happens.
    After this method call the self.inputInfo should be ready to be sent to the model
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    '''
    pass

  def generateInputBatch(self,myInput,model,batchSize,projector=None):
    '''
    this function provide a mask to create several inputs at the same time
    It call the generateInput function as many time as needed
    @in myInput: [] list containing one input set
    @in model: instance of a model
    @in batchSize: integer the number of input sets required
    @in projector used for adaptive sampling to provide the projection of the solution on the success metric
    @return newInputs: [[]] list of the list of input sets'''
    newInputs = []
    while self.amIreadyToProvideAnInput() and (self.counter < batchSize):
      if projector==None: newInputs.append(self.generateInput(model,myInput))
      else              : newInputs.append(self.generateInput(model,myInput,projector))
    return newInputs


  def finalizeActualSampling(self,jobObject,model,myInput):
    '''just an API'''
    self.localFinalizeActualSampling(jobObject,model,myInput)
  
  def localFinalizeActualSampling(self,jobObject,model,myInput):
    '''
    Overwrite only if you need something special at the end of each run....
    This function is used by samplers that need to collect information from the just ended run
    For example, for a Dynamic Event Tree case, this function can be used to retrieve
    the information from the just finished run of a branch in order to retrieve, for example,
    the distribution name that caused the trigger, etc.
    It is a essentially a place-holder for most of the sampler to remain compatible with the Steps structure
    @in jobObject: an instance of a JobHandler
    @in model    : an instance of a model
    @in myInput  : the generating input    
    '''
    pass
#
#
#
class AdaptiveSampler(Sampler):
  '''This is a general adaptive sampler'''
  def __init__(self):
    Sampler.__init__(self)
    self.goalFunction     = None             #this is the pointer to the function defining the goal
    self.adaptAlgo        = None             #this is a pointer to the adaptive algorithm
    self.adaptAlgoType    = ''               #this is the type of adaptive algorithm
    self.normType         = ''               #this is the norm type used
    self.norm             = None             #this is the pointer to the norm function
    self.tolerance        = None             #this is norm of the error threshold
    self.tolleranceWeight = 'probability'    #this is the a flag that controls if the convergence is checked on the hyper-volume or the probability
    self.persistence      = 5                #this is the number of times the error needs to fell below the tollerance before considering the sim converged
    self.repetition       = 0                #the actual number of time the error was below the requested threshold
    self.forceIteration   = False            #this flag control if at least a self.limit number of iteration should be done
    self.axisName         = None             #this is the ordered list of the variable names (ordering match self.gridStepSize anfd the ordering in the test matrixes)
    self.gridVectors      = {}               # {'name of the variable':numpy.ndarray['the coordinate']}
    self.testGridLenght   = 0                #this the total number of point in the testing grid
    self.testMatrix       = None             #This is the n-dimensional matrix representing the testing grid
    self.oldTestMatrix    = None             #This is the test matrix to use to store the old evaluation of the function
    self.gridShape        = None             #tuple describing the shape of the grid matrix
    self.gridCoorShape    = None             #tuple describing the shape of the grid containing also the coordinate 
    self.functionValue    = {}               #This a dictionary that contains np vectors for each variable and the goal function
    self.solutionExport   = None             #This is the data used to export the solution (it could also not be present)
    self.gridCoord        = None             #this is the matrix that contains for each entry of the grid the coordinate
    self.nVar             = 0                #this is the number of the variable sampled
    self.surfPoint        = None             #coordinate of the points considered on the limit surface
    self.sign             = -1
    
  def localInputAndChecks(self,xmlNode):
    #setting up the adaptive algorithm
    if 'adaptiveAlgorithm' in xmlNode.attrib.keys():
      print('FIXME: we need to build/import the library of adaptive algorithms')
      self.adaptAlgoType = xmlNode.attrib['adaptiveAlgorithm']
      import AdaptiveAlgoLib
      if self.adaptAlgoType in AdaptiveAlgoLib.knonwnTypes(): self.adaptAlgo = AdaptiveAlgoLib.returnInstance(self.adaptAlgoType)
      else                                                  : raise Exception('the '+self.adaptAlgoType+'is not a known type of adaptive search algorithm')
    else: raise Exception('the attribute adaptiveAlgorithm was missed in the definition of the adaptive sampler '+self.name)
    #setting up the Convergence characteristic
    convergenceNode = xmlNode.find('Convergence')
    if convergenceNode==None:raise Exception('the node Convergence was missed in the definition of the adaptive sampler '+self.name)
    self.tolerance=float(convergenceNode.text)     
    if 'norm'          in convergenceNode.attrib.keys():
      print('FIXME: we need to build/import the library of adaptive algorithms')
      self.normType = convergenceNode.attrib['norm']
      import NormLib
      if self.normType in NormLib.knonwnTypes()             : self.norm             = NormLib.returnInstance(self.normType)
      else: raise Exception('the '+self.normType+'is not a known type of norm')
    if 'limit'          in convergenceNode.attrib.keys()    : self.limit            = int (convergenceNode.attrib['limit'      ])
    if 'persistence'    in convergenceNode.attrib.keys()    : self.persistence      = int (convergenceNode.attrib['persistence'])
    if 'weight'         in convergenceNode.attrib.keys()    : self.tolleranceWeight = str (convergenceNode.attrib['weight'     ])
    if 'forceIteration' in convergenceNode.attrib.keys()    :
      if   convergenceNode.attrib['forceIteration']=='True' : self.forceIteration   = True
      elif convergenceNode.attrib['forceIteration']=='False': self.forceIteration   = False
      else: raise Exception('in reading the convergence setting for the adaptive sampler '+self.name+' the forceIteration keyword had an unknown value: '+str(convergenceNode.attrib['forceIteration'])) 
      
  def localAddInitParams(self,tempDict):
    tempDict['The adaptive algorithm type is '                ] = self.adaptAlgoType
    tempDict['The norm type is '                              ] = self.normType
    tempDict['Force the sampler to reach the iteration limit '] = str(self.forceIteration)
    tempDict['The norm tolerance is '                         ] = str(self.tolerance)
    tempDict['The type of weighting for the error is '        ] = str(self.tolleranceWeight)
    tempDict['The number of no error repetition requested is '] = str(self.repetition)
         
  def localAddCurrentSetting(self,tempDict):
    if self.solutionExport!=None:
      tempDict['The solution is exported in '    ] = 'Name: ' + self.solutionExport.name + 'Type: ' + self.solutionExport.type
    if self.goalFunction!=None:
      tempDict['The function used is '] = self.goalFunction.name
    for varName in self.distDict.keys():
      tempDict['The coordinate for the convergence test grid on variable '+str(varName)+' are'] = str(self.gridVectors[varName])
  
  def _cKDTreeInterface(self,action,data):
    m = len(list(data.keys()))
    n = len(data[list(data.keys())[0]])
    dataMatrix = np.zeros((n,m-1))
    if action=='train':
      self._mappingList = []
      index = 0
      for key in data.keys():
        if key != self.goalFunction.name:
          self._mappingList.append(key)
          dataMatrix[:,index] = data[key]
          index +=1
      self._tree = spatial.cKDTree(copy.copy(dataMatrix),leafsize=18)
    elif action=='evaluate':
      print('FIXME: here rather than using self.gridCoord a conversion of data would be more coherent')
      distance, outId    = self._tree.query(self.gridCoord)
      return [self.functionValue[self.goalFunction.name][myID] for myID in outId]
    elif action=='confidence':
      distance, outId    = self._tree.query(self.surfPoint)
      return distance, outId
   
  def localInitialize(self,goalFunction=None,solutionExport=None,ROM=None):
    self.goalFunction   = goalFunction
    self.solutionExport = solutionExport
    #build a lambda function to masquerade the ROM <-> cKDTree presence
    if ROM==None:
      class ROM(object):
        def __init__(self,cKDTreeInterface):
          self.amItrained = False
          self._cKDTreeInterface = cKDTreeInterface
        def train(self,trainSet):
          self._cKDTreeInterface('train',trainSet)
          self.amItrained = True
        def evaluate(self,coordinateVect): return self._cKDTreeInterface('evaluate',coordinateVect)
        def confidence(self,coordinateVect): return self._cKDTreeInterface('confidence',coordinateVect)[0]
      self.ROM = ROM(self._cKDTreeInterface)
    else: self.ROM = ROM
    #check if convergence is not on probability if all variables are bounded in value otherwise the problem is unbounded
    if self.tolleranceWeight!='probability':
      for varName in self.distDict.keys():
        if not(self.distDict[varName].upperBoundUsed and self.distDict[varName].lowerBoundUsed):
          raise Exception('It is impossible to converge on an unbounded domain (variable '+varName+' with distribution '+self.distDict[varName].name+') as requested to the sampler '+self.name)
    #setup the grid. The grid is build such as each element has a volume equal to the tolerance
    #the grid is build in such a way that an unit change in each node within the grid correspond to a change equal to the tolerance
    self.nVar        = len(self.distDict.keys())              #Total number of variables 
    stepLenght        = self.tolerance**(1./float(self.nVar)) #build the step size in 0-1 range such as the differential volume is equal to the tolerance 
    self.axisName     = []                                     #this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    #here we build lambda function to return the coordinate of the grid point depending if the tolerance is on probability or on volume
    if self.tolleranceWeight!='probability':
      stepParam = lambda x: [stepLenght*(self.distDict[x].upperBound-self.distDict[x].lowerBound), self.distDict[x].lowerBound, self.distDict[x].upperBound]
    else:
      stepParam = lambda _: [stepLenght, 0.0, 1.0]
    #moving forward building all the information set
    pointByVar = [None]*self.nVar                              #list storing the number of point by cooridnate
    #building the grid point coordinates
    for varId, varName in enumerate(self.distDict.keys()):
      self.axisName.append(varName)     
      [myStepLenght, start, end]  = stepParam(varName)
      start                      += 0.5*myStepLenght
      if self.tolleranceWeight=='probability': self.gridVectors[varName] = np.asarray([self.distDict[varName].ppf(pbCoord) for pbCoord in  np.arange(start,end,myStepLenght)])
      else                                   : self.gridVectors[varName] = np.arange(start,end,myStepLenght)
      pointByVar[varId]           = np.shape(self.gridVectors[varName])[0]
    self.gridShape                = tuple   (pointByVar)          #tuple of the grid shape
    self.testGridLenght           = np.prod (pointByVar)          #total number of point on the grid
    self.testMatrix               = np.zeros(self.gridShape)      #grid where the values of the goalfunction are stored
    self.oldTestMatrix            = np.zeros(self.gridShape)      #swap matrix fro convergence test
    self.gridCoorShape            = tuple(pointByVar+[self.nVar]) #shape of the matrix containing all coordinate of all points in the grid
    self.gridCoord                = np.zeros(self.gridCoorShape)  #the matrix containing all coordinate of all points in the grid
    #filling the coordinate on the grid
    myIterator = np.nditer(self.gridCoord,flags=['multi_index'])
    while not myIterator.finished:
      coordinateID  = myIterator.multi_index[-1]
      axisName      = self.axisName[coordinateID]
      valuePosition = myIterator.multi_index[coordinateID]
      self.gridCoord[myIterator.multi_index] = self.gridVectors[axisName][valuePosition] 
      myIterator.iternext()
    self.axisStepSize = {}
    for varName in self.distDict.keys():
      self.axisStepSize[varName] = np.asarray([self.gridVectors[varName][myIndex+1]-self.gridVectors[varName][myIndex] for myIndex in range(len(self.gridVectors[varName])-1)])
      
    #printing
    if self.debug:
      print('self.gridShape '+str(self.gridShape))
      print('self.testGridLenght '+str(self.testGridLenght))
      print('self.gridCoorShape '+str(self.gridCoorShape))
      for key in self.gridVectors.keys():
        print('the variable '+key+' has coordinate: '+str(self.gridVectors[key]))
      myIterator          = np.nditer(self.testMatrix,flags=['multi_index'])
      while not myIterator.finished:
        print ('Indexes: '+str(myIterator.multi_index)+'    coordinate: '+str(self.gridCoord[myIterator.multi_index]))
        myIterator.iternext()
    print('Initializzation done')

  def localStillReady(self,ready,lastOutput=None):
    '''
    first perform some check to understand what it needs to be done possibly perform an early return
    ready is returned
    lastOutput should be present when the next point should be chosen on previous iteration and convergence checked
    lastOutput it is not considered to be present during the test performed for generating an input batch
    ROM if passed in it is used to construct the test matrix otherwise the nearest neightburn value is used
    '''
    self.debug=True
    if self.debug: print('From method localStillReady...')
    #test on what to do
    if ready      == False : return ready #if we exceeded the limit just return that we are done
    if lastOutput == None and self.ROM.amItrained==False: return ready #if the last output is not provided I am still generating an input batch, if the rom was not trained before we need to start clean
    #first evaluate the goal function on the newly sampled points and store them in mapping description self.functionValue
    if lastOutput !=None:
      print('Initiate training')
      self.functionValue.update(lastOutput.getParametersValues('input'))
      self.functionValue.update(lastOutput.getParametersValues('output'))
      if self.goalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.goalFunction.name])-1
      else:                                                   indexLast = -1
      indexEnd  = len(self.functionValue[self.axisName[0]])-1
      tempDict  = {}
      if self.goalFunction.name in self.functionValue.keys():
        self.functionValue[self.goalFunction.name] = np.append( self.functionValue[self.goalFunction.name], np.zeros(indexEnd-indexLast))
      else: self.functionValue[self.goalFunction.name] = np.zeros(indexEnd+1)
      for myIndex in range(indexLast+1,indexEnd+1):
        for key, value in self.functionValue.items(): tempDict[key] = value[myIndex]
        self.functionValue[self.goalFunction.name][myIndex] =  self.goalFunction.evaluate('residuumSign',tempDict)
      #printing----------------------
      if self.debug: print('Mapping of the goal function evaluation done')
      if self.debug:
        print('already evaluated points and function value')
        keyList = list(self.functionValue.keys())
        print(','.join(keyList))
        for index in range(indexEnd+1):
          print(','.join([str(self.functionValue[key][index]) for key in keyList]))
      #printing----------------------
      tempDict = {}
      print('FIXME: please find a more elegant way to remove the output variables from the training set')
      for name in self.axisName: tempDict[name] = self.functionValue[name]
      tempDict[self.goalFunction.name] = self.functionValue[self.goalFunction.name]    
      self.ROM.train(tempDict) 
      print('Training done')
    if self.debug: print('Training finished')                                    #happy thinking :)
    np.copyto(self.oldTestMatrix,self.testMatrix)                                #copy the old solution for convergence check
    self.testMatrix.shape     = (self.testGridLenght)                            #rearrange the grid matrix such as is an array of values
    self.gridCoord.shape      = (self.testGridLenght,self.nVar)                  #rearrange the grid coordinate matrix such as is an array of coordinate values
    tempDict ={}
    for  varId, varName in enumerate(self.axisName): tempDict[varName] = self.gridCoord[:,varId]
    self.testMatrix[:]        = self.ROM.evaluate(tempDict)                      #get the prediction on the testing grid
    self.testMatrix.shape     = self.gridShape                                   #bring back the grid structure
    self.gridCoord.shape      = self.gridCoorShape                               #bring back the grid structure
    if self.debug: print('Prediction finished')      
    testError                 = np.sum(np.abs(np.subtract(self.testMatrix,self.oldTestMatrix)))#compute the error
    if (testError > 0): ready, self.repetition = True, 0                        #we still have error
    else              : self.repetition +=1                                     #we are increasing persistence
    if self.persistence<self.repetition : ready = False                         #we are done
    print('counter: '+str(self.counter)+'       Error: ' +str(testError)+' Repetition: '+str(self.repetition))
    #here next the points that are close to any change are detected by a gradient (it is a pre-screener)
    toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix)),axis=0))))
    #printing----------------------
    if self.debug:
      print('Limit surface candidate points')
      for coordinate in np.rollaxis(toBeTested,0):
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    #check which one of the preselected points is really on the limit surface
    listsurfPoint = []
    myIdList      = np.zeros(self.nVar)
    for coordinate in np.rollaxis(toBeTested,0):
      myIdList[:] = copy.deepcopy(coordinate)
      if int(self.testMatrix[tuple(coordinate)])<0: #we seek the frontier sitting on the -1 side
        for iVar in range(self.nVar):
          if coordinate[iVar]+1<self.gridShape[iVar]: #coordinate range from 0 to n-1 while shape is equal to n
            myIdList[iVar]+=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]-=1
          if coordinate[iVar]>0:
            myIdList[iVar]-=1
            if self.testMatrix[tuple(myIdList)]>=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]+=1
    #printing----------------------
    if self.debug:
      print('Limit surface points')
      for coordinate in listsurfPoint:
        myStr = ''
        for iVar, varnName in enumerate(self.axisName): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    
    #if the number of point on the limit surface is > than zero than save it
    if len(listsurfPoint)>0:
      self.surfPoint = np.ndarray((len(listsurfPoint),self.nVar))
      for pointID, coordinate in enumerate(listsurfPoint): self.surfPoint[pointID,:] = self.gridCoord[tuple(coordinate)]
      if self.solutionExport!=None:
        for varName in self.solutionExport.getParaKeys('inputs'):
          for varIndex in range(len(self.axisName)):
            if varName == self.axisName[varIndex]:
              self.solutionExport.removeInputValue(varName)
              self.solutionExport.updateInputValue(varName,self.surfPoint[:,varIndex])
    
    return ready
    
  def localGenerateInput(self,model,oldInput):
    #self.adaptAlgo.nextPoint(self.dataContainer,self.goalFunction,self.values,self.distDict)
    # create values dictionary
    '''compute the direction normal to the surface, compute the derivative normal to the surface of the probability,
     check the points where the derivative probability is the lowest'''  
    if self.debug: print('generating input')
    if self.surfPoint!=None and len(self.surfPoint)>0:
      tempDict = {}
      for name in self.axisName: tempDict[name] = self.functionValue[name]
      tempDict[self.goalFunction.name] = self.functionValue[self.goalFunction.name]    
      self._cKDTreeInterface('train',tempDict)
      tempDict = {}
      for varIndex, varName in enumerate(self.axisName):
        tempDict[varIndex] = self.surfPoint[:,varIndex]
      distance, _ = self._cKDTreeInterface('confidence',tempDict)
      minIndex = np.argmax(distance)
      for varIndex, varName in enumerate(self.axisName):
        self.values[varName] = copy.copy(float(self.surfPoint[minIndex,varIndex]))
     
      
#This is the normal derivation to be used later on
#      pbMapPointCoord = np.zeros((len(self.surfPoint),self.nVar*2+1,self.nVar))
#      for pointIndex, point in enumerate(self.surfPoint):
#        temp = copy.copy(point)
#        pbMapPointCoord[pointIndex,2*self.nVar,:] = temp
#        for varIndex, varName in enumerate(self.axisName):
#          temp[varIndex] -= np.max(self.axisStepSize[varName])
#          pbMapPointCoord[pointIndex,varIndex,:] = temp
#          temp[varIndex] += 2.*np.max(self.axisStepSize[varName])
#          pbMapPointCoord[pointIndex,varIndex+self.nVar,:] = temp
#          temp[varIndex] -= np.max(self.axisStepSize[varName])
#      print('Indexing of close point to the limit surface done')
#      #getting the coordinate ready to be evaluated by the ROM
#      pbMapPointCoord.shape = (len(self.surfPoint)*(self.nVar*2+1),self.nVar)
#      tempDict = {}
#      for varIndex, varName in enumerate(self.axisName):
#        tempDict[varName] = pbMapPointCoord.T[varIndex,:]
#      print('ready to request pb')
#      #acquiring Pb evaluation
#      pbPoint       = self.ROM.confidence(tempDict)
#      pbPoint.shape = (len(self.surfPoint),self.nVar*2+1,2)
#      pbMapPointCoord.shape = (len(self.surfPoint),self.nVar*2+1,self.nVar)
#      #computing gradient
#      modGrad   = np.zeros((len(self.surfPoint)))
#      gradVect  = np.zeros((len(self.surfPoint),self.nVar))
#      for pointIndex in range(len(self.surfPoint)):
#        centralCoor = pbMapPointCoord[pointIndex,2*self.nVar,:]
#        centraPb    = pbPoint[pointIndex,2*self.nVar][0]
#        sum = 0.0
#        for varIndex in range(self.nVar):
#          d1Down     = (centraPb-pbPoint[pointIndex,varIndex][0])/(centralCoor[varIndex]-pbMapPointCoord[pointIndex,varIndex,varIndex])
#          d1Up       = (pbPoint[pointIndex,varIndex+self.nVar][0]-centraPb)/(pbMapPointCoord[pointIndex,varIndex+self.nVar,varIndex]-centralCoor[varIndex])
#          if np.abs(d1Up)>np.abs(d1Down): d1Avg = d1Up
#          else                          : d1Avg = d1Down          
#          gradVect[pointIndex,varIndex] = d1Avg
#          sum +=d1Avg
#          modGrad[pointIndex] += d1Avg**2
#        modGrad[pointIndex] = np.sqrt(modGrad[pointIndex])*np.abs(sum)/sum
#        #concavityPb[pointIndex] = concavityPb[pointIndex]/float(self.nVar)
#      print('concavity computed')
#      print(self.axisName)
#      for pointIndex, point in enumerate(self.surfPoint):
#        myStr  = ''
#        myStr  += '['       
#        for varIndex in range(self.nVar):
#          myStr += '{:+6.4f}'.format(pbMapPointCoord[pointIndex,2*self.nVar,varIndex])
#        myStr += '] '+'{:+6.4f}'.format(pbPoint[pointIndex,2*self.nVar,0])+'   '
#        for varIndex in range(2*self.nVar):
#          myStr += '['
#          for varIndex2 in range(self.nVar):
#            myStr += '{:+6.4f}'.format(pbMapPointCoord[pointIndex,varIndex,varIndex2])+' '
#          myStr += '] '+'{:+6.4f}'.format(pbPoint[pointIndex,varIndex,0])+'   '
#        myStr += '   gradient  ['
#        for varIndex in range(self.nVar):
#          myStr += '{:+6.4f}'.format(gradVect[pointIndex,varIndex])+'  '
#        myStr += ']'
#        myStr += '    Module '+'{:+6.4f}'.format(modGrad[pointIndex])
#        print(myStr)
#      print('probability acquired')
#                                 
#      minIndex = np.argmin(np.abs(modGrad))
#      print('index on the limit surface of the smallest gradient '+ str(minIndex)+'corresponding gradient module '+str(modGrad[minIndex])+' and probability '+str(pbPoint[minIndex,2*self.nVar][0]))
#      pdDist = self.sign*(pbPoint[minIndex,2*self.nVar][0]-0.5-10*self.tolerance)/modGrad[minIndex]
#      print('extrapolation length' +str(pdDist))
#      for varIndex, varName in enumerate(self.axisName):
#        self.values[varName] = copy.copy(float(pbMapPointCoord[minIndex,2*self.nVar,varIndex]+pdDist*gradVect[minIndex,varIndex]))
#      gradVect = np.ndarray(self.nVar)
#      centraPb = pbPoint[minIndex,2*self.nVar]
#      centralCoor = pbMapPointCoord[minIndex,2*self.nVar,:]
#      for varIndex in range(self.nVar):
#        d1Down = (centraPb-pbPoint[minIndex,varIndex])/(centralCoor[varIndex]-pbMapPointCoord[minIndex,varIndex,varIndex])
#        d1Up   = (pbPoint[minIndex,varIndex+self.nVar]-centraPb)/(pbMapPointCoord[minIndex,varIndex+self.nVar,varIndex]-centralCoor[varIndex])
#        d1Avg   = (d1Up+d1Down)/2.0
#        gradVect[varIndex] = d1Avg
#      gradVect = gradVect*pdDist
#      gradVect = gradVect+centralCoor
#      for varIndex, varName in enumerate(self.axisName):
#        self.values[varName] = copy.copy(float(gradVect[varIndex]))
    else:
      #here we are still generating the batch
      for key in self.distDict.keys():
        if self.tolleranceWeight=='probability':
          self.values[key]= self.distDict[key].ppf(float(Distributions.random()))
        else:
          self.values[key]= self.distDict[key].lowerBound+(self.distDict[key].upperBound-self.distDict[key].lowerBound)*float(Distributions.random())
    #self.debug=True
    if self.debug:
      print('At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
    self.debug=False
    self.sign = -1*self.sign
  def localFinalizeActualSampling(self,jobObject,model,myInput):
    '''generate representation of goal function'''
    pass
#
#
#
class MonteCarlo(Sampler):
  '''MONTE CARLO Sampler'''
  
  def localInputAndChecks(self,xmlNode):
    if 'limit' not in  xmlNode.attrib.keys(): raise IOError(' Monte Carlo sampling needs the attribute limit (number of samplings)')

  def localGenerateInput(self,model,myInput):
    '''set up self.inputInfo before being sent to the model'''
    # create values dictionary
    for key in self.distDict: self.values[key]=self.distDict[key].rvs()
#
#
#
class Grid(Sampler):
  '''
  Samples the model on a given (by input) set of points
  '''
  def __init__(self):
    Sampler.__init__(self) 
    self.gridCoordinate       = [] #the grid point to be used for each distribution (changes at each step)
    self.axisName             = [] #the name of each axis (variable)
    self.gridInfo             = {} # {'name of the variable':('Type',Construction,[values])} gridType: Probability/Value, gridConstruction:Custom/Equal    
    #gridInfo[var][0] is type, ...[1] is construction, ...[2] is values

  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    self.limit = 1
    for child in xmlNode:
      if child.tag == "Distribution":
        #Add <distribution> to name so we know it is not a direct variable
        varName = "<distribution>"+child.attrib['name']
      else:
        varName = child.attrib['name']

      for childChild in child:
        if childChild.tag =='grid':
          self.axisName.append(varName)
          constrType = childChild.attrib['construction']
          if constrType == 'custom':
            tempList = [float(i) for i in childChild.text.split()]
            tempList.sort()
            self.gridInfo[varName] = (childChild.attrib['type'],constrType,tempList)
            self.limit = len(tempList)*self.limit
          elif constrType == 'equal':
            self.limit = self.limit*(int(childChild.attrib['steps'])+1)
            if   'lowerBound' in childChild.attrib.keys():
              self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['lowerBound']) + float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
            elif 'upperBound' in childChild.attrib.keys():
              self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['upperBound']) - float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])    
            else: raise IOError('no upper or lower bound has been declared for '+str(child.tag)+' in sampler '+str(self.name))
          else: raise IOError('not specified the grid construction type')
    if len(self.toBeSampled.keys()) != len(self.gridInfo.keys()): raise IOError('inconsistency between number of variables and grid specification')
    self.gridCoordinate = [None]*len(self.axisName)

  def localAddInitParams(self,tempDict):
    for variable in self.gridInfo.items():
      tempList = [str(i) for i in variable[1][2]]
      tempDict[variable[0]+' is sampled using the grid'] = variable[1][0]+' with spacing '+variable[1][1]+', points: '+' '.join(tempList)
            
  def localAddCurrentSetting(self,tempDict):
    for var, value in zip(self.axisName, self.gridCoordinate):
      tempDict['coordinate '+var+' has value'] = value

  def localInitialize(self):
    '''This is used to check if the points and bounds are compatible with the distribution provided'''
    for varName in self.gridInfo.keys():
      if self.gridInfo[varName][0]=='value':
        if self.distDict[varName].upperBoundUsed:
          if max(self.gridInfo[varName][2])>self.distDict[varName].upperBound:
            raise Exception('the variable '+varName+'can not be sampled at '+str(max(self.gridInfo[varName][2]))+' since outside the upper bound of the chosen distribution')
        if self.distDict[varName].lowerBoundUsed:
          if min(self.gridInfo[varName][2])<self.distDict[varName].lowerBound:
            raise Exception('the variable '+varName+'can not be sampled at '+str(min(self.gridInfo[varName][2]))+' since outside the upper bound of the chosen distribution')
        
  def localGenerateInput(self,model,myInput):
    remainder = self.counter - 1 #used to keep track as we get to smaller strides
    stride = self.limit+1 #How far apart in the 1D array is the current gridCoordinate
    #self.inputInfo['distributionInfo'] = {}
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    for i in range(len(self.gridCoordinate)):
      varName = self.axisName[i]
      #self.inputInfo['distributionInfo'][varName] = self.gridInfo[varName]
      #print(varName,self.toBeSampled[varName])
      self.inputInfo['distributionName'][varName] = self.toBeSampled[varName][1]
      self.inputInfo['distributionType'][varName] = self.toBeSampled[varName][0]
      stride = stride // len(self.gridInfo[varName][2]) 
      #index is the index into the array self.gridInfo[varName][2]
      index, remainder = divmod(remainder, stride )
      self.gridCoordinate[i] = index
      if self.gridInfo[varName][0]=='CDF':
        self.values[varName] = self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]])
      elif self.gridInfo[varName][0]=='value':
        self.values[varName] = self.gridInfo[varName][2][self.gridCoordinate[i]]
#
#
#
class LHS(Grid):
  '''
  Latin hyper Cube based sampler. Currently no special filling method are implemented
  '''
  def __init__(self):
    Grid.__init__(self) 
    self.sampledCoordinate    = [] # a list of list for i=0,..,limit a list of the coordinate to be used this is needed for the LHS

  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)
    pointByVar  = [len(self.gridInfo[variable][2]) for variable in self.gridInfo.keys()]
    if len(set(pointByVar))!=1: raise IOError('the latin Hyper Cube requires the same number of point in each dimension')
    self.pointByVar = pointByVar[0]
    self.inputInfo['upper'] = {}
    self.inputInfo['lower'] = {}
    self.limit = (self.pointByVar-1)

  def localInitialize(self):
    '''
    the local initialize is used to generate test the box being within the distribution upper/lower bound
    and filling mapping of the hyper cube.
    '''
    Grid.localInitialize(self)
    tempFillingCheck = [None]*len(self.axisName) #for all variables
    for i in range(len(tempFillingCheck)):
      tempFillingCheck[i] = [None]*(self.pointByVar-1) #intervals are n-points-1
      tempFillingCheck[i][:] = Distributions.randomPermutation(list(range(self.pointByVar-1))) #pick a random interval sequence
    self.sampledCoordinate = [None]*(self.pointByVar-1)
    for i in range(self.pointByVar-1):
      self.sampledCoordinate[i] = [None]*len(self.axisName)
      self.sampledCoordinate[i][:] = [tempFillingCheck[j][i] for j in range(len(tempFillingCheck))]

  def localGenerateInput(self,model,myInput):
    j=0
    #self.inputInfo['distributionInfo'] = {}
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    for varName in self.axisName:
      upper = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]+1]
      lower = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]  ]
      j +=1
      intervalFraction = Distributions.random()
      coordinate = lower + (upper-lower)*intervalFraction
      #self.inputInfo['distributionInfo'][varName] = self.gridInfo[varName]
      self.inputInfo['distributionName'][varName] = self.toBeSampled[varName][1]
      self.inputInfo['distributionType'][varName] = self.toBeSampled[varName][0]
      if self.gridInfo[varName][0] =='CDF':
        self.values[varName] = self.distDict[varName].ppf(coordinate)
        self.inputInfo['upper'][varName] = self.distDict[varName].ppf(max(upper,lower))
        self.inputInfo['lower'][varName] = self.distDict[varName].ppf(min(upper,lower))
      elif self.gridInfo[varName][0]=='value':
        self.values[varName] = coordinate
        self.inputInfo['upper'][varName] = max(upper,lower)
        self.inputInfo['lower'][varName] = min(upper,lower)
#
#
#
class DynamicEventTree(Sampler):
  '''
  DYNAMIC EVEN TREE Sampler - "ANalysis of Dynamic REactor Accident evolution" module (DET      ) :D
  '''
  def __init__(self):
    Sampler.__init__(self)
    # Working directory (Path of the directory in which all the outputs,etc. are stored)
    self.workingDir = ""       
    # (optional) if not present, the sampler will not change the relative keyword in the input file
    self.maxSimulTime            = None  
    # print the xml tree representation of the dynamic event tree calculation
    # see variable 'self.TreeInfo'
    self.print_end_xml           = False 
    # Dictionary of the probability bins for each distribution that have been 
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    self.branchProbabilities     = {}
    # Dictionary of the Values' bins for each distribution that have been 
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    # these are the invCDFs of the PBs inputted in branchProbabilities (if ProbabilityThresholds have been inputted) 
    self.branchValues     = {}
    # List of Dictionaries of the last probability bin level (position in the array) reached for each distribution ('distName':IntegerValue)
    # This container is a working dictionary. The branchedLevels are stored in the xml tree "self.TreeInfo" since they must track
    # the evolution of the dynamic event tree
    self.branchedLevel           = []
    # Counter for the branch needs to be run after a calculation branched (it is a working variable)
    self.branchCountOnLevel      = 0
    # Dictionary tha contains the actual branching info 
    # (i.e. distribution that triggered, values of the variables that need to be changed, etc)
    self.actualBranchInfo        = {}
    # Parent Branch end time (It's a working variable used to set up the new branches need to be run.
    #   The new branches' start time will be the end time of the parent branch )
    self.actual_end_time         = 0.0
    # Parent Branch end time step (It's a working variable used to set up the new branches need to be run.
    #  The end time step is used to construct the filename of the restart files needed for restart the new branch calculations)
    self.actual_end_ts           = 0
    # Xml tree object. It stored all the info regarding the DET. It is in continue evolution during a DET calculation
    self.TreeInfo                = None
    # List of Dictionaries. It is a working variable used to store the information needed to create branches from a Parent Branch
    self.endInfo                 = []
    # Queue system. The inputs are waiting to be run are stored in this queue dictionary
    self.RunQueue                = {}
    # identifiers of the inputs in queue (name of the history... for example DET_1,1,1)
    self.RunQueue['identifiers'] = []
    # Corresponding inputs
    self.RunQueue['queue'      ] = []

  def amIreadyToProvideAnInput(self):
    '''
    Function that inquires if there is at least an input the in the queue that needs to be run
    @ In, None
    @ Out, boolean 
    '''
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0): return True
    else: 
      if self.print_end_xml: self.TreeInfo.writeNodeTree(self.workingDir+"/"+self.name + "_output_summary.xml")
      return False

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    '''
    General function (available to all samplers) that finalize the sampling calculation just ended 
    In this case (DET), The function reads the information from the ended calculation, updates the
    working variables, and creates the new inputs for the next branches
    @ In, jobObject: JobHandler Instance of the job (run) just finished
    @ In, model    : Model Instance... It may be a Code Instance, a ROM, etc.
    @ In, myInput  : List of the original input files
    @ Out, None 
    '''  
    self.workingDir = model.workingDir
    # Read the branch info from the parent calculation (just ended calculation)
    # This function stores the information in the dictionary 'self.actualBranchInfo'
    # If no branch info, this history is concluded => return
    if not self.__readBranchInfo(jobObject.output): return
    # Collect the branch info in a multi-level dictionary
    endInfo = {'end_time':self.actual_end_time,'end_ts':self.actual_end_ts,'branch_dist':list(self.actualBranchInfo.keys())[0]}
    endInfo['branch_changed_params'] = self.actualBranchInfo[endInfo['branch_dist']]
    # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    if(jobObject.identifier == self.TreeInfo.getrootnode().name): endInfo['parent_node'] = self.TreeInfo.getrootnode()
    else: endInfo['parent_node'] = list(self.TreeInfo.getrootnode().iter(jobObject.identifier))[0] 
    # get the branchedLevel dictionary 
    branchedLevel = endInfo['parent_node'].get('branchedLevel')
    if not branchedLevel: raise Exception('SAMPLER DET   : ERROR -> branchedLevel of node '+jobObject.identifier+'not found!!!!')  
    # Loop of the parameters that have been changed after a trigger gets activated
    for key in endInfo['branch_changed_params']:
      endInfo['n_branches'] = 1 + int(len(endInfo['branch_changed_params'][key]['actual_value']))
      if(len(endInfo['branch_changed_params'][key]['actual_value']) > 1):
        #  Multi-Branch mode => the resulting branches from this parent calculation (just ended)
        # will be more then 2
        # unchanged_pb = probability (not conditional probability yet) that the event does not occur  
        unchanged_pb = 0.0
        try:
          # changed_pb = probability (not conditional probability yet) that the event A occurs and the final state is 'alpha' ''' 
          for pb in xrange(len(endInfo['branch_changed_params'][key]['associated_pb'])): unchanged_pb = unchanged_pb + endInfo['branch_changed_params'][key]['associated_pb'][pb]
        except KeyError: pass
        if(unchanged_pb <= 1): endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0-unchanged_pb
      else:
        # Two-Way mode => the resulting branches from this parent calculation (just ended) = 2
        if branchedLevel[endInfo['branch_dist']] > len(self.branchProbabilities[endInfo['branch_dist']])-1: pb = 1.0
        else: pb = self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]
        endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0 - pb
        endInfo['branch_changed_params'][key]['associated_pb'] = [pb]
  
    self.branchCountOnLevel = 0
    # set runEnded and running to true and false respectively
    endInfo['parent_node'].add('runEnded',True)
    endInfo['parent_node'].add('running',False)
    endInfo['parent_node'].add('end_time',self.actual_end_time)
    # The branchedLevel counter is updated
    if branchedLevel[endInfo['branch_dist']] < len(self.branchProbabilities[endInfo['branch_dist']]): branchedLevel[endInfo['branch_dist']] += 1
    # Append the parent branchedLevel (updated for the new branch/es) in the list tha contains them 
    # (it is needed in order to avoid overlapping among info coming from different parent calculations)
    # When this info is used, they are popped out
    self.branchedLevel.append(branchedLevel) 
    # Append the parent end info in the list tha contains them 
    # (it is needed in order to avoid overlapping among info coming from different parent calculations)
    # When this info is used, they are popped out
    self.endInfo.append(endInfo)
    # Compute conditional probability 
    self.computeConditionalProbability()
    # Create the inputs and put them in the runQueue dictionary
    self.__createRunningQueue(model,myInput)
    return

  def computeConditionalProbability(self,index=None):
    '''
    Function to compute Conditional probability of the branches that are going to be run.
    The conditional probabilities are stored in the self.endInfo object
    @ In, index: position in the self.endInfo list (optional). Default = 0
    '''
    if not index: index = len(self.endInfo)-1
    # parent_cond_pb = associated conditional probability of the Parent branch 
    #parent_cond_pb = 0.0  
    try:
      parent_cond_pb = self.endInfo[index]['parent_node'].get('conditional_pb')
      if not parent_cond_pb: parent_cond_pb = 1.0
    except KeyError: parent_cond_pb = 1.0
    # for all the branches the conditional pb is computed 
    # unchanged_cond_pb = Conditional Probability of the branches in which the event has not occurred
    # changed_cond_pb   = Conditional Probability of the branches in which the event has occurred  
    for key in self.endInfo[index]['branch_changed_params']:
      #try:
      self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'] = []
      self.endInfo[index]['branch_changed_params'][key]['unchanged_cond_pb'] = parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['unchanged_pb'])
      for pb in range(len(self.endInfo[index]['branch_changed_params'][key]['associated_pb'])): self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'].append(parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['associated_pb'][pb]))
      #except? pass
    return  

  def __readBranchInfo(self,out_base=None):
    '''
    Function to read the Branching info that comes from a Model
    The branching info (for example, distribution that triggered, parameters must be changed, etc)
    are supposed to be in a xml format
    @ In, out_base: is the output root that, if present, is used to construct the file name the function is going
                    to try reading.
    @ Out, boolean: true if the info are present (a set of new branches need to be run), false if the actual parent calculation reached an end point
    '''
    
    # Remove all the elements from the info container 
    del self.actualBranchInfo
    self.actualBranchInfo = {}
    # Construct the file name adding the out_base root if present
    if out_base: filename = out_base + "_actual_branch_info.xml"
    else: filename = "actual_branch_info.xml"
    if not os.path.isabs(filename): filename = os.path.join(self.workingDir,filename)
    if not os.path.exists(filename):
      print('SAMPLER DET   : branch info file ' + filename +' has not been found. => No Branching.')
      branch_present = False
      return branch_present
    # Parse the file and create the xml element tree object
    #try:
    branch_info_tree = ET.parse(filename)
    print('SAMPLER DET   : Done parsing '+filename)
    #except? raise IOError ('not able to parse ' + filename)
    root = branch_info_tree.getroot()  
    # Check if end_time and end_ts (time step)  are present... In case store them in the relative working vars 
    #try: #Branch info written out by program, so should always exist.
    self.actual_end_time = float(root.attrib['end_time'])
    self.actual_end_ts   = int(root.attrib['end_ts'])
    #except? pass
    # Store the information in a dictionary that has as keywords the distributions that triggered
    for node in root:
      if node.tag == "Distribution_trigger":
        dist_name = node.attrib['name'].strip()
        self.actualBranchInfo[dist_name] = {}
        for child in node:
          self.actualBranchInfo[dist_name][child.text.strip()] = {'varType':child.attrib['type'].strip(),'actual_value':child.attrib['actual_value'].strip().split(),'old_value':child.attrib['old_value'].strip()}
          if 'probability' in child.attrib:
            as_pb = child.attrib['probability'].strip().split()
            self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'] = []
            #self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb)) 
            for index in range(len(as_pb)): self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb[index]))
      # we exit the loop here, because only one trigger at the time can be handled  right now 
      break
    # remove the file
    os.remove(filename)
    branch_present = True
    return branch_present 

  def __createRunningQueue(self,model,myInput):
    '''
    Function to create and append new inputs to the queue. It uses all the containers have been updated by the previous functions
    @ In, model  : Model instance. It can be a Code type, ROM, etc.
    @ In, myInput: List of the original inputs
    @ Out, None
    '''
    # Check if the number of calculation that have been run is greater than 1. If so, the simulation is already in the tree 
    if self.counter >= 1:
      # The first DET calculation branch has already been run'
      # Start the manipulation:
      #  Pop out the last endInfo information and the branchedLevel
      branchedLevelG = copy.deepcopy(self.branchedLevel.pop(0))
      endInfo = self.endInfo.pop(0)
      # n_branches = number of branches need to be run
      n_branches = endInfo['n_branches']
      # Check if the distribution that just triggered hitted the last probability threshold . 
      # In case we create a number of branches = endInfo['n_branches'] - 1 => the branch in 
      # which the event did not occur is not going to be tracked
      if branchedLevelG[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']]):
        print('SAMPLER DET   : Branch ' + endInfo['parent_node'].get('name') + ' hit last Threshold for distribution ' + endInfo['branch_dist']) 
        print('SAMPLER DET   : Branch ' + endInfo['parent_node'].get('name') + ' is dead end.')
        self.branchCountOnLevel = 1
        n_branches = endInfo['n_branches'] - 1
      # Loop over the branches for which the inputs must be created
      for _ in range(n_branches):
        del self.inputInfo
        self.counter += 1
        self.branchCountOnLevel += 1
        branchedLevel = copy.deepcopy(branchedLevelG)
        # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
        rname = copy.deepcopy(endInfo['parent_node'].get('name') + '-' + str(self.branchCountOnLevel))
        # create a subgroup that will be appended to the parent element in the xml tree structure 
        subGroup = ETS.Node(rname)
        subGroup.add('parent', endInfo['parent_node'].get('name'))
        subGroup.add('name', rname)
        # cond_pb_un = conditional probability event not occur
        # cond_pb_c  = conditional probability event/s occur/s
        cond_pb_un = 0.0
        cond_pb_c  = 0.0
        # Loop over  branch_changed_params (events) and start storing information, 
        # such as conditional pb, variable values, into the xml tree object
        for key in endInfo['branch_changed_params'].keys():
          subGroup.add('branch_changed_param',key)
          if self.branchCountOnLevel != 1:
            subGroup.add('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2]))
            subGroup.add('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2]))
            #try: 
            cond_pb_c = cond_pb_c + copy.deepcopy(endInfo['branch_changed_params'][key]['changed_cond_pb'][self.branchCountOnLevel-2])
            #except? pass
          else:
            subGroup.add('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['old_value']))
            subGroup.add('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_pb']))
            #try:
            cond_pb_un =  cond_pb_un + copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_cond_pb'])
            #except? pass
        # add conditional probability
        if self.branchCountOnLevel != 1: subGroup.add('conditional_pb',copy.deepcopy(cond_pb_c))
        else: subGroup.add('conditional_pb',copy.deepcopy(cond_pb_un)) 
        # add initiator distribution info, start time, etc. 
        subGroup.add('initiator_distribution',copy.deepcopy(endInfo['branch_dist']))
        subGroup.add('start_time', copy.deepcopy(endInfo['parent_node'].get('end_time')))
        # initialize the end_time to be equal to the start one... It will modified at the end of this branch
        subGroup.add('end_time', copy.deepcopy(endInfo['parent_node'].get('end_time')))
        # add the branchedLevel dictionary to the subgroup
        if self.branchCountOnLevel != 1: branchedLevel[endInfo['branch_dist']] = branchedLevel[endInfo['branch_dist']] - 1
        
        subGroup.add('branchedLevel', copy.deepcopy(branchedLevel))
        # branch calculation info... running, queue, etc are set here
        subGroup.add('runEnded',False)
        subGroup.add('running',False)
        subGroup.add('queue',True)
#        subGroup.set('restartFileRoot',endInfo['restartRoot'])
        # Append the new branch (subgroup) info to the parent_node in the xml tree object
        endInfo['parent_node'].appendBranch(subGroup)
        
        # Fill the values dictionary that will be passed into the model in order to create an input
        # In this dictionary the info for changing the original input is stored
        self.inputInfo = {'prefix':copy.deepcopy(rname),'end_ts':copy.deepcopy(endInfo['end_ts']),
                  'branch_changed_param':copy.deepcopy([subGroup.get('branch_changed_param')]),
                  'branch_changed_param_value':copy.deepcopy([subGroup.get('branch_changed_param_value')]),
                  'conditional_prb':copy.deepcopy([subGroup.get('conditional_pb')]),
                  'start_time':copy.deepcopy(endInfo['parent_node'].get('end_time')),
                  'parent_id':subGroup.get('parent')}

        # Check if the distribution that just triggered hitted the last probability threshold . 
        #  In this case there is not a probability threshold that needs to be added in the input
        #  for this particular distribution 
        if not (branchedLevel[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']])):
          self.inputInfo['initiator_distribution'] = copy.deepcopy([endInfo['branch_dist']])
          self.inputInfo['PbThreshold'           ] = copy.deepcopy([self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]])
          self.inputInfo['ValueThreshold'        ] = copy.deepcopy([self.branchValues[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]])
          
        #  For the other distributions, we put the unbranched thresholds
        #  Before adding these thresholds, check if the keyword 'initiator_distribution' is present... 
        #  (In the case the previous if statement is true, this keyword is not present yet
        #  Add it otherwise 
        if not ('initiator_distribution' in self.inputInfo.keys()):
          self.inputInfo['initiator_distribution'] = []
          self.inputInfo['PbThreshold'           ] = []
          self.inputInfo['ValueThreshold'        ] = []
          
        # Add the unbranched thresholds
        for key in self.branchProbabilities.keys():
          if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])): self.inputInfo['initiator_distribution'].append(copy.deepcopy(key.encode()))
        for key in self.branchProbabilities.keys():
          if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])):
            self.inputInfo['PbThreshold'   ].append(copy.deepcopy(self.branchProbabilities[key][branchedLevel[key]]))
            self.inputInfo['ValueThreshold'].append(copy.deepcopy(self.branchValues[key][branchedLevel[key]]))
        self.inputInfo['SampledVars']   = {}
        self.inputInfo['SampledVarsPb'] = {}
        for varname in self.toBeSampled.keys():
          self.inputInfo['SampledVars'][varname]   = copy.deepcopy(self.branchValues[self.toBeSampled[varname][1]][branchedLevel[self.toBeSampled[varname][1]]])
          self.inputInfo['SampledVarsPb'][varname] = copy.deepcopy(self.branchProbabilities[self.toBeSampled[varname][1]][branchedLevel[self.toBeSampled[varname][1]]])
        # Call the model function "createNewInput" with the "values" dictionary just filled.
        # Add the new input path into the RunQueue system  
        self.RunQueue['queue'].append(copy.deepcopy(model.createNewInput(myInput,self.type,**self.inputInfo)))
        self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
        del branchedLevel

    else:
      # We construct the input for the first DET branch calculation'
      # Increase the counter
      # The root name of the xml element tree is the starting name for all the branches 
      # (this root name = the user defined sampler name)
      rname = self.TreeInfo.getrootnode().name 
      # Get the initial branchedLevel dictionary (=> the list gets empty)
      branchedLevelG = copy.deepcopy(self.branchedLevel.pop(0))
      branchedLevel = copy.deepcopy(branchedLevelG)
      # Fill th values dictionary in
      self.inputInfo['prefix'                    ] = copy.deepcopy(rname)
      self.inputInfo['initiator_distribution'    ] = []
      self.inputInfo['PbThreshold'               ] = []
      self.inputInfo['ValueThreshold'            ] = []
      self.inputInfo['branch_changed_param'      ] = copy.deepcopy([b'None'])
      self.inputInfo['branch_changed_param_value'] = copy.deepcopy([b'None'])
      self.inputInfo['start_time'                ] = copy.deepcopy(b'Initial')
      self.inputInfo['end_ts'                    ] = copy.deepcopy(0)
      self.inputInfo['parent_id'                 ] = copy.deepcopy(b'root')
      self.inputInfo['conditional_prb'           ] = copy.deepcopy([1.0])
      for key in self.branchProbabilities.keys():self.inputInfo['initiator_distribution'].append(copy.deepcopy(key.encode()))
      for key in self.branchProbabilities.keys():self.inputInfo['PbThreshold'].append(copy.deepcopy(self.branchProbabilities[key][branchedLevel[key]]))
      for key in self.branchProbabilities.keys():self.inputInfo['ValueThreshold'].append(copy.deepcopy(self.branchValues[key][branchedLevel[key]]))
      for varname in self.toBeSampled.keys():
        self.inputInfo['SampledVars'  ][varname] = copy.deepcopy(self.branchValues[self.toBeSampled[varname][1]][branchedLevel[self.toBeSampled[varname][1]]])
        self.inputInfo['SampledVarsPb'][varname] = copy.deepcopy(self.branchProbabilities[self.toBeSampled[varname][1]][branchedLevel[self.toBeSampled[varname][1]]])
  
      if(self.maxSimulTime): self.inputInfo['end_time'] = self.maxSimulTime
      # Call the model function "createNewInput" with the "values" dictionary just filled.
      # Add the new input path into the RunQueue system  
      newInputs = model.createNewInput(myInput,self.type,**self.inputInfo)
      self.RunQueue['queue'].append(copy.deepcopy(newInputs))
      self.RunQueue['identifiers'].append(copy.deepcopy(self.inputInfo['prefix']))
      del newInputs
      del branchedLevel
      self.counter += 1
    del branchedLevelG
    return  
  
  def __getQueueElement(self):
    '''
    Function to get an input from the internal queue system
    @ In, None  
    @ Out, jobInput: First input in the queue 
    '''
    if len(self.RunQueue['queue']) == 0:
      # There are no more runs must be run
      #  we set the self.limit == self.counter
      #  => the simulation ends
      self.limit = self.counter
      # If the user specified to print the xml representation of the calculation
      #  Print it out
      if self.print_end_xml: self.TreeInfo.writeNodeTree(self.workingDir+"/"+self.name + "_output_summary.xml")
      return None
    else:
      # Pop out the first input in queue
      jobInput = self.RunQueue['queue'].pop(0)
      jobId     = self.RunQueue['identifiers'].pop(0)
      #set running flags in self.TreeInfo
      root = self.TreeInfo.getrootnode()
      # Update the run information flags
      if (root.name == jobId):
        root.add('runEnded',str(False))
        root.add('running',str(True)) 
        root.add('queue',str(False))
      else:
        subElm = list(root.iter(jobId))[0]
        if(subElm):
          subElm.add('runEnded',str(False))
          subElm.add('running',str(True))
          subElm.add('queue',str(False))
    return jobInput

  def generateInput(self,model,oldInput):
    '''
    This method needs to be overwritten by the Dynamic Event Tree Sampler, since the input creation strategy is completely different with the respect the other samplers
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    @return     : [] containing the new inputs -in reality it is the model that returns this, the Sampler generates the values to be placed in the model input
    ''' 
    return self.localGenerateInput(model, oldInput)

  def localGenerateInput(self,model,myInput):
    if self.counter <= 1:
      # If first branch input, create the queue 
      self.__createRunningQueue(model, myInput)
    # retrieve the input from the queue
    newerinput = self.__getQueueElement()
    if not newerinput:
      # If no inputs are present in the queue => a branch is finished 
      print("SAMPLER DET   : A Branch ended!!!!")
    return newerinput

  def localInputAndChecks(self,xmlNode):

    try:    self.print_end_xml = (xmlNode.attrib['print_end_xml'].lower() in ['true','t','yes','si','y'])
    except KeyError: self.print_end_xml = False

    # retrieve max simulation time, if input
    try:    self.maxSimulTime = xmlNode.attrib['maxSimulationTime']
    except (KeyError,NameError): self.maxSimulTime = None
    
    # Read branching settings
    #children = xmlNode.find("BranchingSettings")
    # this branching levels may be either probability thresholds or value thresholds
    branchedLevel = {}
    error_found = False
    for child in xmlNode:
      for childChild in child:
        if childChild.tag =='distribution':
          branchedLevel[childChild.text] = 0
          if 'ProbabilityThresholds' in childChild.attrib: 
            self.branchProbabilities[childChild.text] = [float(x) for x in childChild.attrib['ProbabilityThresholds'].split()]
            self.branchProbabilities[childChild.text].sort(key=float)
            if max(self.branchProbabilities[childChild.text]) > 1:
              print("SAMPLER DET   : ERROR -> One of the Thresholds for distribution " + str(child.attrib['distName']) + " is > 1")
              error_found = True
            for index in range(len(sorted(self.branchProbabilities[childChild.text], key=float))):
              if sorted(self.branchProbabilities[childChild.text], key=float).count(sorted(self.branchProbabilities[childChild.text], key=float)[index]) > 1:
                print("SAMPLER DET   : ERROR -> In distribution " + str(childChild.text) + " the Threshold " + str(sorted(self.branchProbabilities[childChild.text], key=float)[index])+" appears multiple times!!")
                error_found = True
            # the invCDF of the branchProbabilities are computed in the initialize method (called right before the Sampler gets used)
          elif 'ValueThresholds' in childChild.attrib:
            self.branchValues[childChild.text] = [float(x) for x in childChild.attrib['ValueThresholds'].split()]
            self.branchValues[childChild.text].sort(key=float)
            for index in range(len(sorted(self.branchValues[childChild.text], key=float))):
              if sorted(self.branchValues[childChild.text], key=float).count(sorted(self.branchValues[childChild.text], key=float)[index]) > 1:
                print("SAMPLER DET   : ERROR -> In distribution " + str(childChild.text) + " the Threshold " + str(sorted(self.branchValues[childChild.text], key=float)[index])+" appears multiple times!!")
                error_found = True
            # the associated CDF of the branchValues are computed in the initialize method (called right before the Sampler gets used)
          else: raise IOError('The DynamicEventTree sampler needs that all distributions have either Probability Thresholds or Value Thresholds!!!!')
    if error_found: raise IOError("In Sampler " + self.name+' ERRORS have been found!!!' )     
    # Append the branchedLevel dictionary in the proper list
    self.branchedLevel.append(branchedLevel)
  
  def localAddInitParams(self,tempDict):
    
    for key in self.branchProbabilities.keys():
      tempDict['Probability Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchProbabilities[key]]
    for key in self.branchValues.keys():
      tempDict['Values Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchValues[key]]
      
  def localAddCurrentSetting(self,tempDict):
    tempDict['actual threshold levels are '] = self.branchedLevel[0]
  
  def localInitialize(self):
    elm = ETS.Node(self.name + '_1')
  
    elm.add('name', self.name + '_1')
    elm.add('start_time', str(0.0))
    # Initialize the end_time to be equal to the start one... 
    # It will modified at the end of each branch
    elm.add('end_time', str(0.0))
    elm.add('runEnded',str(False))
    elm.add('running',str(True))
    elm.add('queue',str(False))   
    # The dictionary branchedLevel is stored in the xml tree too. That's because
    # the advancement of the thresholds must follow the tree structure
    elm.add('branchedLevel', self.branchedLevel[0])
    # Here it is stored all the info regarding the DET => we create the info for all the
    # branchings and we store them
    self.TreeInfo = ETS.NodeTree(elm) 
    
    if (len(self.branchProbabilities.keys()) != 0):
      #compute the associated invCDF values
      for key in self.branchProbabilities.keys():
        self.branchValues[key] = [copy.deepcopy(self.distDict[key].ppf(float(self.branchProbabilities[key][index]))) for index in range(len(self.branchProbabilities[key]))]
    else:
      #compute the associated CDF values
      for key in self.branchValues.keys():
        self.branchProbabilities[key] = [copy.deepcopy(self.distDict[key].cdf(float(self.branchValues[key][index]))) for index in range(len(self.branchValues[key]))]
    return

'''
 Interface Dictionary (factory) (private)
'''
__base = 'Sampler'
__interFaceDict = {}
__interFaceDict['MonteCarlo'            ] = MonteCarlo
__interFaceDict['DynamicEventTree'      ] = DynamicEventTree
__interFaceDict['LHS'                   ] = LHS
__interFaceDict['Grid'                  ] = Grid
__interFaceDict['Adaptive'              ] = AdaptiveSampler
__knownTypes = list(__interFaceDict.keys())

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
  function used to generate a Sampler class
  @ In, Type : Sampler type
  @ Out,Instance of the Specialized Sampler class
  '''
  try: return __interFaceDict[Type]()
  except KeyError: raise NameError('not known '+__base+' type '+Type)

def optionalInputs(Type):
  pass

def mandatoryInputs(Type):
  pass

