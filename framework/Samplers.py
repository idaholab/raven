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
import json
from operator import mul
from functools import reduce
from scipy import spatial
from scipy.interpolate import InterpolatedUnivariateSpline
import xml.etree.ElementTree as ET
import itertools
from sklearn import neighbors
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import metaclass_insert,find_le,index,find_le_index,returnPrintTag,returnPrintPostTag,stringsThatMeanTrue
from BaseClasses import BaseType
from Assembler import Assembler
import Distributions
import TreeStructure as ETS
import SupervisedLearning
import pyDOE as doe
#Internal Modules End--------------------------------------------------------------------------------

class Sampler(metaclass_insert(abc.ABCMeta,BaseType),Assembler):
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
  myInstance.whatDoINeed()                           -see Assembler class-
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
    self.counter                       = 0                         # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.auxcnt                        = 0                         # Aux counter of samples performed (for its usage check initialize method)
    self.limit                         = sys.maxsize               # maximum number of Samples (for example, Monte Carlo = Number of Histories to run, DET = Unlimited)
    self.toBeSampled                   = {}                        # Sampling mapping dictionary {'Variable Name':'name of the distribution'}
    self.distDict                      = {}                        # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.values                        = {}                        # for each variable the current value {'var name':value}
    self.inputInfo                     = {}                        # depending on the sampler several different type of keywarded information could be present only one is mandatory, see below
    self.initSeed                      = None                      # if not provided the seed is randomly generated at the istanciation of the sampler, the step can override the seed by sending in another seed
    self.inputInfo['SampledVars'     ] = self.values               # this is the location where to get the values of the sampled variables
    self.inputInfo['SampledVarsPb'   ] = {}                        # this is the location where to get the probability of the sampled variables
    self.inputInfo['PointProbability'] = None                      # this is the location where the point wise probability is stored (probability associated to a sampled point)
    self.inputInfo['crowDist']         = {}                        # Stores a dictionary that contains the information to create a crow distribution.  Stored as a json object
    self.reseedAtEachIteration         = False                     # Logical flag. True if every newer evaluation is perfermed after a new reseeding
    self.FIXME                         = False                     # FIXME flag
    self.printTag                      = returnPrintTag(self.type) # prefix for all prints (sampler type)
    self._endJobRunnable               = sys.maxsize               # max number of inputs creatable by the sampler right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)
    
    self.assemblerObjects  = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    self.requiredAssObject = (True,(['Distribution'],['n']))       # tuple. first entry boolean flag. True if the XML parser must look for assembler objects;
                                                      # second entry tuple.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))

#  def whatDoINeed(self):
#    '''
#    This method is used mainly by the Simulation class at the Step construction stage.
#    It is used for inquiring the Sampler about the kind of objects the Sampler needs to
#    be initialize. It is an abstract method that comes from the base class Assembler(see BaseClasses.py)
#    @ In , None, None
#    @ Out, needDict, dictionary of objects needed (class:list(tuple(object type{if None, Simulation does not check the type}, object name))). (eg. {'Distributions':[(type1,distname1),(type2,distname2)]} )
#    '''
#    # call the local method for getting additional needed objects
#    needDict = self._localWhatDoINeed()
#    # the distributions are the common things that are needed by each sampler
#    if 'Distributions' not in needDict.keys(): needDict['Distributions'] = []
#    for dist in self.toBeSampled.values(): needDict['Distributions'].append((None,dist))
#    return needDict

  def _localWhatDoINeed(self):
    '''
    This method is a local mirrow of the general whatDoINeed method.
    It is implmented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    '''
    needDict = {}
    needDict['Distributions'] = []
    for dist in self.toBeSampled.values(): needDict['Distributions'].append((None,dist))
    return needDict

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
    if 'reseedAtEachIteration' in xmlNode.attrib.keys():
      if xmlNode.attrib['reseedAtEachIteration'].lower() in stringsThatMeanTrue(): self.reseedAtEachIteration = True
    for child in xmlNode:
      for childChild in child:
        if childChild.tag =='distribution':
          if child.tag == 'Distribution':
            #Add <distribution> to name so we know it is not the direct variable
            self.toBeSampled["<distribution>"+child.attrib['name']] = childChild.text
          elif child.tag == 'variable': self.toBeSampled[child.attrib['name']] = childChild.text
          else: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Unknown tag '+child.tag+' .Available are: Distribution and variable!')
          if len(list(childChild.attrib.keys())) > 0: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Unknown attributes for distribution node '+childChild.text+'. Got '+str(childChild.attrib.keys()).replace('[', '').replace(']',''))
    self.localInputAndChecks(xmlNode)

  def endJobRunnable(self): return self._endJobRunnable

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
      tempDict[variable[0]] = 'is sampled using the distribution ' +variable[1]
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
    tempDict['counter'       ] = self.counter
    tempDict['initial seed'  ] = self.initSeed
    for key in self.inputInfo:
      if key!='SampledVars': tempDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys(): tempDict['Variable: '+var+' has value'] = tempDict[key][var]
    self.localAddCurrentSetting(tempDict)

  def localAddCurrentSetting(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''
    pass

#  def generateAssembler(self,initDict):
#    availableDist = initDict['Distributions']
#    self._generateDistributions(availableDist)
#    self._localGenerateAssembler(initDict)

  def _localGenerateAssembler(self,initDict): 
    availableDist = initDict['Distributions']
    self._generateDistributions(availableDist)

  def _generateDistributions(self,availableDist):
    '''
    here the needed distribution are made available to the step as also the initialization
    of the seeding (the siding could be overriden by the step by calling the initialize method
    @in availableDist: {'distribution name':instance}
    '''
    if self.initSeed != None:
      Distributions.randomSeed(self.initSeed)
    for key in self.toBeSampled.keys():
      if self.toBeSampled[key] not in availableDist.keys(): IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Distribution '+self.toBeSampled[key]+' not found among available distributions (check input)!!!')
      self.distDict[key] = availableDist[self.toBeSampled[key]]
      self.inputInfo['crowDist'][key] = json.dumps(self.distDict[key].getCrowDistDict())

  def initialize(self,externalSeeding=None,solutionExport=None):
    '''
    This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
    @in solutionExport: in goal oriented sampling (a.k.a. adaptive sampling this is where the space/point satisfying the constrains)
    '''
    self.counter = 0
    if   not externalSeeding          :
      Distributions.randomSeed(self.initSeed)       #use the sampler initialization seed
      self.auxcnt = self.initSeed
    elif externalSeeding=='continue'  : pass        #in this case the random sequence needs to be preserved
    else                              :
      Distributions.randomSeed(externalSeeding)     #the external seeding is used
      self.auxcnt = externalSeeding
    for key in self.toBeSampled.keys():
        print(key)
        self.distDict[key].initializeDistribution()   #now we can initialize the distributions
    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport != None : self.localInitialize(solutionExport=solutionExport)
    else                      : self.localInitialize()

  def localInitialize(self):
    '''
    use this function to add initialization features to the derived class
    it is call at the beginning of each step
    '''
    pass

  def amIreadyToProvideAnInput(self): #inLastOutput=None):
    '''
    This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
    This method act as a "traffic light" for generating a new input.
    Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
    @ In, None, None
    @ Out, ready, Boolean
    '''
    if(self.counter < self.limit): ready = True
    else                         : ready = False
    ready = self.localStillReady(ready)
    return ready

  def localStillReady(self,ready): #,lastOutput=None
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
    self.counter +=1                              #since we are creating the input for the next run we increase the counter and global counter
    self.auxcnt  +=1
    if self.reseedAtEachIteration: Distributions.randomSeed(self.auxcnt-1)
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

  def generateInputBatch(self,myInput,model,batchSize,projector=None): #,lastOutput=None
    '''
    this function provide a mask to create several inputs at the same time
    It call the generateInput function as many time as needed
    @in myInput: [] list containing one input set
    @in model: instance of a model
    @in batchSize: integer the number of input sets required
    @in projector used for adaptive sampling to provide the projection of the solution on the success metric
    @return newInputs: [[]] list of the list of input sets'''
    newInputs = []
    #inlastO = None
    #if lastOutput:
    #  if not lastOutput.isItEmpty(): inlastO = lastOutput
    #while self.amIreadyToProvideAnInput(inlastO) and (self.counter < batchSize):
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
    It is a essentially a place-holder for most of the sampler to remain compatible with the StepsCR structure
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
    self.assemblerObjects = {}               #this dictionary contains information about the object needed by the adaptive sampler in order to work (ROM,targetEvaluation, etc)
    self.goalFunction     = None             #this is the pointer to the function defining the goal
    self.tolerance        = None             #this is norm of the error threshold
    self.subGridTol       = None             #This is the tolerance used to construct the testing sub grid
    self.ROM              = None             #This contains a pointer to the ROM instance
    self.toleranceWeight  = 'probability'    #this is the a flag that controls if the convergence is checked on the hyper-volume or the probability
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
    self.functionValue    = {}               #This a dictionary that contains np vectors with the value for each variable and for the goal function
    self.solutionExport   = None             #This is the data used to export the solution (it could also not be present)
    self.gridCoord        = None             #this is the matrix that contains for each entry of the grid the coordinate
    self.nVar             = 0                #this is the number of the variable sampled
    self.surfPoint        = None             #coordinate of the points considered on the limit surface
    self.hangingPoints    = []               #list of the points already submitted for evaluation for which the result is not yet available
    self.printTag         = returnPrintTag('SAMPLER ADAPTIVE')

    self.assemblerObjects  = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    self.requiredAssObject = (True,(['Distribution',],['n']))       # tuple. first entry boolean flag. True if the XML parser must look for assembler objects;

  def _localWhatDoINeed(self):
    '''
    This method is a local mirrow of the general whatDoINeed method.
    It is implmented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    '''
#    needDict = {}
#    for value in self.assemblerObjects.values():
#      if value[0] not in needDict.keys(): needDict[value[0]] = []
#      needDict[value[0]].append((value[1],value[2]))
#    return needDict
    return {}

  def _localGenerateAssembler(self,initDict):
    for key, value in self.assemblerObjects.items():
      if key in 'TargetEvaluation' : self.lastOutput = initDict[value[0]][value[2]]
      if key in 'ROM'              : self.ROM = initDict[value[0]][value[2]]
      if key in 'Function'         : self.goalFunction = initDict[value[0]][value[2]]
    if self.ROM==None:
      mySrting= ','.join(list(self.distDict.keys()))
      self.ROM = SupervisedLearning.returnInstance('SciKitLearn',**{'SKLtype':'neighbors|KNeighborsClassifier','Features':mySrting,'Target':self.goalFunction.name})
    self.ROM.reset()

  def localInputAndChecks(self,xmlNode):
    if 'limit' in xmlNode.attrib.keys():
      try: self.limit = int(xmlNode.attrib['limit'])
      except ValueError: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    # convergence Node
    convergenceNode = xmlNode.find('Convergence')
    if convergenceNode==None:raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> the node Convergence was missed in the definition of the adaptive sampler '+self.name)
    try   : self.tolerance=float(convergenceNode.text)
    except: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Failed to convert '+convergenceNode.text+' to a meaningful number for the convergence')
    attribList = list(convergenceNode.attrib.keys())
    if 'limit'          in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('limit'))
      try   : self.limit = int (convergenceNode.attrib['limit'])
      except: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Failed to convert the limit value '+convergenceNode.attrib['limit']+' to a meaningful number for the convergence')
    if 'persistence'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('persistence'))
      try   : self.persistence = int (convergenceNode.attrib['persistence'])
      except: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Failed to convert the persistence value '+convergenceNode.attrib['persistence']+' to a meaningful number for the convergence')
    if 'weight'         in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('weight'))
      try   : self.toleranceWeight = str(convergenceNode.attrib['weight'])
      except: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Failed to convert the weight type '+convergenceNode.attrib['weight']+' to a meaningful string for the convergence')
    if 'subGridTol'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('subGridTol'))
      try   : self.subGridTol = float (convergenceNode.attrib['subGridTol'])
      except: raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Failed to convert the subGridTol '+convergenceNode.attrib['subGridTol']+' to a meaningful float for the convergence')
    if 'forceIteration' in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('forceIteration'))
      if   convergenceNode.attrib['forceIteration']=='True' : self.forceIteration   = True
      elif convergenceNode.attrib['forceIteration']=='False': self.forceIteration   = False
      else: raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Reading the convergence setting for the adaptive sampler '+self.name+' the forceIteration keyword had an unknown value: '+str(convergenceNode.attrib['forceIteration']))
    #assembler node: Hidden from User
    targEvalNode = xmlNode.find('TargetEvaluation')
    if targEvalNode == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> TargetEvaluation object is required. Not found in Sampler '+self.name + '!')
    self.assemblerObjects[targEvalNode.tag] = [targEvalNode.attrib['class'],targEvalNode.attrib['type'],targEvalNode.text]
    functionNode = xmlNode.find('Function')
    if functionNode == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Function object is required. Not Found in Sampler '+self.name + '!')
    self.assemblerObjects[functionNode.tag] = [functionNode.attrib['class'],functionNode.attrib['type'],functionNode.text]
    romNode = xmlNode.find('ROM')
    if romNode != None: self.assemblerObjects[romNode.tag] = [romNode.attrib['class'],romNode.attrib['type'],romNode.text]
    targEvalCounter  = 0
    romCounter       = 0
    functionCounter  = 0
    for subNode in xmlNode:
      if 'TargetEvaluation' in subNode.tag:
        targEvalCounter += 1
      if 'ROM'              in subNode.tag:
        romCounter += 1
      if 'Function'         in subNode.tag:
        functionCounter += 1
    if targEvalCounter != 1: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> One TargetEvaluation object is required. Sampler '+self.name + ' got '+str(targEvalCounter) + '!')
    if functionCounter != 1: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> One Function object is required. Sampler '+self.name + ' got '+str(functionCounter) + '!')
    if romCounter      >  1: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Only one ROM object is required. Sampler '+self.name + ' got '+str(romCounter) + '!')
    # set subgrid
    if self.subGridTol == None: self.subGridTol = self.tolerance
    if self.subGridTol> self.tolerance: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> The sub grid tolerance '+str(self.subGridTol)+' must be smaller than the tolerance: '+str(self.tolerance))
    if len(attribList)>0: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> There are unknown keywords in the convergence specifications: '+str(attribList))

  def localAddInitParams(self,tempDict):
    tempDict['Iter. forced'    ] = str(self.forceIteration)
    tempDict['Norm tolerance'  ] = str(self.tolerance)
    tempDict['Sub grid size'   ] = str(self.subGridTol)
    tempDict['Error Weight'    ] = str(self.toleranceWeight)
    tempDict['Persistence'     ] = str(self.repetition)

  def localAddCurrentSetting(self,tempDict):
    if self.solutionExport!=None:
      tempDict['The solution is exported in '    ] = 'Name: ' + self.solutionExport.name + 'Type: ' + self.solutionExport.type
    if self.goalFunction!=None:
      tempDict['The function used is '] = self.goalFunction.name
    for varName in self.distDict.keys():
      tempDict['The coordinate for the convergence test grid on variable '+str(varName)+' are'] = str(self.gridVectors[varName])

  def localInitialize(self,solutionExport=None):
    self.memoryStep        = 5               # number of step for which the memory is kept
    self.solutionExport    = solutionExport
    # check if solutionExport is actually a "Datas" type "TimePointSet"
    if type(solutionExport).__name__ != "TimePointSet": raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> solutionExport type is not a TimePointSet. Got '+ type(solutionExport).__name__+'!')
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.testMatrix        = None             #This is the n-dimensional matrix representing the testing grid
    self.oldTestMatrix     = None             #This is the test matrix to use to store the old evaluation of the function
    self.functionValue     = {}               #This a dictionary that contains np vectors with the value for each variable and for the goal function
    self.persistenceMatrix = None             #this is a matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.surfPoint         = None
    if self.goalFunction.name not in self.solutionExport.getParaKeys('output'): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Goal function name does not match solution export data output.')
    self._endJobRunnable   = 1
    #build a lambda function to masquerade the ROM <-> cKDTree presence
    #if not goalFunction: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Gaol Function not provided!!')
    #set up the ROM for the acceleration
    #mySrting= ','.join(list(self.distDict.keys()))
    #if ROM==None: self.ROM = SupervisedLearning.returnInstance('SciKitLearn',**{'SKLtype':'neighbors|KNeighborsClassifier','Features':mySrting,'Target':self.goalFunction.name})
    #else        : self.ROM = ROM
    #check if convergence is not on probability if all variables are bounded in value otherwise the problem is unbounded
    if self.toleranceWeight=='none':
      for varName in self.distDict.keys():
        if not(self.distDict[varName].upperBoundUsed and self.distDict[varName].lowerBoundUsed):
          raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> It is impossible to converge on an unbounded domain (variable '+varName+' with distribution '+self.distDict[varName].name+') as requested to the sampler '+self.name)
    elif self.toleranceWeight=='probability': pass
    else: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Unknown weight string descriptor: '+self.toleranceWeight)
    #setup the grid. The grid is build such as each element has a volume equal to the sub grid tolerance
    #the grid is build in such a way that an unit change in each node within the grid correspond to a change equal to the tolerance
    self.nVar        = len(self.distDict.keys())               #Total number of variables
    stepLenght        = self.subGridTol**(1./float(self.nVar)) #build the step size in 0-1 range such as the differential volume is equal to the tolerance
    self.axisName     = []                                     #this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    #here we build lambda function to return the coordinate of the grid point depending if the tolerance is on probability or on volume
    if self.toleranceWeight!='probability':
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
      if self.toleranceWeight=='probability': self.gridVectors[varName] = np.asarray([self.distDict[varName].ppf(pbCoord) for pbCoord in  np.arange(start,end,myStepLenght)])
      elif self.toleranceWeight=='none'     : self.gridVectors[varName] = np.arange(start,end,myStepLenght)
      pointByVar[varId]           = np.shape(self.gridVectors[varName])[0]
    self.gridShape                = tuple   (pointByVar)          #tuple of the grid shape
    self.testGridLenght           = np.prod (pointByVar)          #total number of point on the grid
    self.testMatrix               = np.zeros(self.gridShape)      #grid where the values of the goalfunction are stored
    self.oldTestMatrix            = np.zeros(self.gridShape)      #swap matrix fro convergence test
    self.gridCoorShape            = tuple(pointByVar+[self.nVar]) #shape of the matrix containing all coordinate of all points in the grid
    self.gridCoord                = np.zeros(self.gridCoorShape)  #the matrix containing all coordinate of all points in the grid
    self.persistenceMatrix        = np.zeros(self.gridShape)      #matrix that for each point of the testing grid tracks the persistence of the limit surface position
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
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> self.gridShape '+str(self.gridShape))
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> self.testGridLenght '+str(self.testGridLenght))
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> self.gridCoorShape '+str(self.gridCoorShape))
      for key in self.gridVectors.keys():
        print(self.printTag+': ' +returnPrintPostTag('Message') + '-> the variable '+key+' has coordinate: '+str(self.gridVectors[key]))
      myIterator          = np.nditer(self.testMatrix,flags=['multi_index'])
      while not myIterator.finished:
        print (self.printTag+': ' +returnPrintPostTag('Message') + '-> Indexes: '+str(myIterator.multi_index)+'    coordinate: '+str(self.gridCoord[myIterator.multi_index]))
        myIterator.iternext()
    self.hangingPoints    = np.ndarray((0, self.nVar))
    #if ROM==None: self.ROM = SupervisedLearning.returnInstance('SciKitLearn',**{'SKLtype':'neighbors|KNeighborsClassifier','Features':','.join(self.axisName),'Target':self.goalFunction.name})
    #else        : self.ROM = ROM
    print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Initialization done')

  def _trainingROM(self,lastOutput):
    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Initiate training')
    if type(lastOutput) == dict:
      self.functionValue.update(lastOutput['inputs' ])
      self.functionValue.update(lastOutput['outputs'])
    else:
      self.functionValue.update(lastOutput.getParametersValues('inputs',nodeid='RecontructEnding'))
      self.functionValue.update(lastOutput.getParametersValues('outputs',nodeid='RecontructEnding'))
    #recovery the index of the last function evaluation performed
    if self.goalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.goalFunction.name])-1
    else                                                  : indexLast = -1
    #index of last set of point tested and ready to perform the function evaluation
    indexEnd  = len(self.functionValue[self.axisName[0].replace('<distribution>','')])-1
    tempDict  = {}
    if self.goalFunction.name in self.functionValue.keys(): self.functionValue[self.goalFunction.name] = np.append( self.functionValue[self.goalFunction.name], np.zeros(indexEnd-indexLast))
    else                                                  : self.functionValue[self.goalFunction.name] = np.zeros(indexEnd+1)
    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.functionValue.items(): tempDict[key] = value[myIndex]
      if len(self.hangingPoints) > 0: self.hangingPoints = self.hangingPoints[~(self.hangingPoints==np.array([tempDict[varName] for varName in [key.replace('<distribution>','') for key in self.axisName]])).all(axis=1)][:]
      self.functionValue[self.goalFunction.name][myIndex] =  self.goalFunction.evaluate('residuumSign',tempDict)
      if type(lastOutput) == dict:
        # if a dictionary, the check must be outside!!!A.
        if self.goalFunction.name not in lastOutput['outputs'].keys(): lastOutput['outputs'][self.goalFunction.name] = np.atleast_1d(self.functionValue[self.goalFunction.name][myIndex])
        else                                                         : lastOutput['outputs'][self.goalFunction.name] = np.concatenate((lastOutput['outputs'][self.goalFunction.name],np.atleast_1d(self.functionValue[self.goalFunction.name][myIndex])))
      else:
        if self.goalFunction.name in lastOutput.getParaKeys('inputs') : lastOutput.updateInputValue (self.goalFunction.name,self.functionValue[self.goalFunction.name][myIndex])
        if self.goalFunction.name in lastOutput.getParaKeys('outputs'): lastOutput.updateOutputValue(self.goalFunction.name,self.functionValue[self.goalFunction.name][myIndex])
    #printing----------------------
    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Mapping of the goal function evaluation done')
    if self.debug:
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> already evaluated points and function value')
      print(','.join(list(self.functionValue.keys())))
      for index in range(indexEnd+1): print(','.join([str(self.functionValue[key][index]) for key in list(self.functionValue.keys())]))
    #printing----------------------
    tempDict = {}
    for name in [key.replace('<distribution>','') for key in self.axisName]: tempDict[name] = self.functionValue[name]
    tempDict[self.goalFunction.name] = self.functionValue[self.goalFunction.name]
    self.ROM.train(tempDict)

  def localStillReady(self,ready): #,lastOutput=None
    '''
    first perform some check to understand what it needs to be done possibly perform an early return
    ready is returned
    lastOutput should be present when the next point should be chosen on previous iteration and convergence checked
    lastOutput it is not considered to be present during the test performed for generating an input batch
    ROM if passed in it is used to construct the test matrix otherwise the nearest neightburn value is used
    '''

    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> From method localStillReady...')
    #test on what to do
    if ready      == False : return ready #if we exceeded the limit just return that we are done
    if type(self.lastOutput) == dict:
      if self.lastOutput == None and self.ROM.amITrained==False: return ready
    else:
      if self.lastOutput.isItEmpty() and self.ROM.amITrained==False: return ready #if the last output is not provided I am still generating an input batch, if the rom was not trained before we need to start clean
    #first evaluate the goal function on the newly sampled points and store them in mapping description self.functionValue RecontructEnding
    if type(self.lastOutput) == dict:
      if self.lastOutput != None: self._trainingROM(self.lastOutput)
    else:
      if not self.lastOutput.isItEmpty(): self._trainingROM(self.lastOutput)
    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Training finished')           #happy thinking :)
    np.copyto(self.oldTestMatrix,self.testMatrix)                                #copy the old solution for convergence check
    self.testMatrix.shape     = (self.testGridLenght)                            #rearrange the grid matrix such as is an array of values
    self.gridCoord.shape      = (self.testGridLenght,self.nVar)                  #rearrange the grid coordinate matrix such as is an array of coordinate values
    tempDict ={}
    for  varId, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]): tempDict[varName] = self.gridCoord[:,varId]
    self.testMatrix[:]        = self.ROM.evaluate(tempDict)                      #get the prediction on the testing grid
    self.testMatrix.shape     = self.gridShape                                   #bring back the grid structure
    self.gridCoord.shape      = self.gridCoorShape                               #bring back the grid structure
    self.persistenceMatrix   += self.testMatrix
    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Prediction finished')
    testError                 = np.sum(np.abs(np.subtract(self.testMatrix,self.oldTestMatrix)))#compute the error
    if (testError > self.tolerance/self.subGridTol): ready, self.repetition = True, 0                        #we still have error
    else              : self.repetition +=1                                     #we are increasing persistence
    if self.persistence<self.repetition: ready =  False                         #we are done
    print(self.printTag+': ' +returnPrintPostTag('Message') + '-> counter: '+str(self.counter)+'       Error: ' +str(testError)+' Repetition: '+str(self.repetition))
    #here next the points that are close to any change are detected by a gradient (it is a pre-screener)
    toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix)),axis=0))))
    #printing----------------------
    if self.debug:
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Limit surface candidate points')
      for coordinate in np.rollaxis(toBeTested,0):
        myStr = ''
        for iVar, varnName in enumerate([key.replace('<distribution>','') for key in self.axisName]): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    listsurfPoint=self.__localLimitStateSearch__(toBeTested,-1)         #it returns the list of points belonging to the limit state surface and resulting in a negative response by the ROM
    nNegPoints=len(listsurfPoint)
    listsurfPoint.extend(self.__localLimitStateSearch__(toBeTested,1))  #it returns the list of points belonging to the limit state surface and resulting in a positive response by the ROM
    nTotPoints=len(listsurfPoint)
    #printing----------------------
    if self.debug:
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Limit surface points')
      for coordinate in listsurfPoint:
        myStr = ''
        for iVar, varnName in enumerate([key.replace('<distribution>','') for key in self.axisName]): myStr +=  varnName+': '+str(coordinate[iVar])+'      '
        print(myStr+'  value: '+str(self.testMatrix[tuple(coordinate)]))
    #printing----------------------
    #if the number of point on the limit surface is > than zero than save it
    if len(listsurfPoint)>0:
      self.surfPoint = np.ndarray((len(listsurfPoint),self.nVar))
      self.invPointPersistence = np.ndarray(len(listsurfPoint))
      for pointID, coordinate in enumerate(listsurfPoint):
        self.surfPoint[pointID,:] = self.gridCoord[tuple(coordinate)]
        self.invPointPersistence[pointID]=abs(self.persistenceMatrix[tuple(coordinate)])
      maxPers = np.max(self.invPointPersistence)
      self.invPointPersistence = (maxPers-self.invPointPersistence)/maxPers
      if self.solutionExport!=None:
        for varName in self.solutionExport.getParaKeys('inputs'):
          for varIndex in range(len(self.axisName)):
            if varName == [key.replace('<distribution>','') for key in self.axisName][varIndex]:
              self.solutionExport.removeInputValue(varName)
              for value in self.surfPoint[:,varIndex]: self.solutionExport.updateInputValue(varName,copy.copy(value))
        evaluations=np.concatenate((-np.ones(nNegPoints),np.ones(nTotPoints-nNegPoints)), axis=0)
        # to be fixed
        self.solutionExport.removeOutputValue(self.goalFunction.name)
        for index in range(len(evaluations)):
          self.solutionExport.updateOutputValue(self.goalFunction.name,copy.copy(evaluations[index]))
    return ready

  def __localLimitStateSearch__(self,toBeTested,sign):
    '''
    It returns the list of points belonging to the limit state surface and resulting in positive or negative responses by the ROM, depending on whether ''sign'' equals either -1 or 1, respectively.
    '''
    listsurfPoint=[]
    myIdList= np.zeros(self.nVar)
    for coordinate in np.rollaxis(toBeTested,0):
      myIdList[:]=copy.deepcopy(coordinate)
      if self.testMatrix[tuple(coordinate)]*sign>0:
        for iVar in range(self.nVar):
          if coordinate[iVar]+1<self.gridShape[iVar]:
            myIdList[iVar]+=1
            if self.testMatrix[tuple(myIdList)]*sign<=0:
              listsurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar]-=1
            if coordinate[iVar]>0:
              myIdList[iVar]-=1
              if self.testMatrix[tuple(myIdList)]*sign<=0:
                listsurfPoint.append(copy.copy(coordinate))
                break
              myIdList[iVar]+=1
    return listsurfPoint

  def localGenerateInput(self,model,oldInput):
    #self.adaptAlgo.nextPoint(self.dataContainer,self.goalFunction,self.values,self.distDict)
    # create values dictionary
    '''compute the direction normal to the surface, compute the derivative normal to the surface of the probability,
     check the points where the derivative probability is the lowest'''

    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    if self.debug: print('generating input')
    varSet=False
    if self.surfPoint!=None and len(self.surfPoint)>0:

      sampledMatrix = np.zeros((len(self.functionValue[self.axisName[0].replace('<distribution>','')])+len(self.hangingPoints[:,0]),len(self.axisName)))
      for varIndex, name in enumerate([key.replace('<distribution>','') for key in self.axisName]): sampledMatrix [:,varIndex] = np.append(self.functionValue[name],self.hangingPoints[:,varIndex])
      distanceTree = spatial.cKDTree(copy.copy(sampledMatrix),leafsize=12)
      tempDict = {}
      #the hanging point are added to the list of the already explored points so not to pick the same when in //
#      lastPoint = [self.functionValue[name][-1] for name in [key.replace('<distribution>','') for key in self.axisName]]
#      for varIndex, name in enumerate([key.replace('<distribution>','') for key in self.axisName]): tempDict[name] = np.append(self.functionValue[name],self.hangingPoints[:,varIndex])
      tempDict = {}
      #distLast = np.zeros(self.surfPoint.shape[0])
      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
        tempDict[varName]     = self.surfPoint[:,varIndex]
        #distLast[:] += np.square(tempDict[varName]-lastPoint[varIndex])
        self.inputInfo['distributionName'][self.axisName[varIndex]] = self.toBeSampled[self.axisName[varIndex]]
        self.inputInfo['distributionType'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].type
      #distLast = np.sqrt(distLast)
      distance, _ = distanceTree.query(self.surfPoint)
      #distance = np.multiply(distance,distLast,self.invPointPersistence)
      distance = np.multiply(distance,self.invPointPersistence)
      if np.max(distance)>0.0:
        for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
          self.values[self.axisName[varIndex]] = copy.copy(float(self.surfPoint[np.argmax(distance),varIndex]))
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True
      else: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> np.max(distance)=0.0')
    if not varSet:
      #here we are still generating the batch
      for key in self.distDict.keys():
        if self.toleranceWeight=='probability':
          self.values[key]                      = self.distDict[key].ppf(float(Distributions.random()))
        else:
          self.values[key]                      = self.distDict[key].lowerBound+(self.distDict[key].upperBound-self.distDict[key].lowerBound)*float(Distributions.random())
        self.inputInfo['distributionName'][key] = self.toBeSampled[key]
        self.inputInfo['distributionType'][key] = self.distDict[key].type
        self.inputInfo['SampledVarsPb'   ][key] = self.distDict[key].pdf(self.values[key])
    self.inputInfo['PointProbability'    ]      = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    # the probability weight here is not used, the post processor is going to recreate the grid associated and use a ROM for the probability evaluation
    self.inputInfo['ProbabilityWeight']         = 1.0
    self.hangingPoints                          = np.vstack((self.hangingPoints,copy.copy(np.array([self.values[axis] for axis in self.axisName]))))
    #print(self.hangingPoints)
    if self.debug: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
    self.inputInfo['SamplerType'] = 'Adaptive'
    self.inputInfo['subGridTol' ] = self.subGridTol

#This is the normal derivation to be used later on
#      pbMapPointCoord = np.zeros((len(self.surfPoint),self.nVar*2+1,self.nVar))
#      for pointIndex, point in enumerate(self.surfPoint):
#        temp = copy.copy(point)
#        pbMapPointCoord[pointIndex,2*self.nVar,:] = temp
#        for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
#          temp[varIndex] -= np.max(self.axisStepSize[varName])
#          pbMapPointCoord[pointIndex,varIndex,:] = temp
#          temp[varIndex] += 2.*np.max(self.axisStepSize[varName])
#          pbMapPointCoord[pointIndex,varIndex+self.nVar,:] = temp
#          temp[varIndex] -= np.max(self.axisStepSize[varName])
#      print('Indexing of close point to the limit surface done')
#      #getting the coordinate ready to be evaluated by the ROM
#      pbMapPointCoord.shape = (len(self.surfPoint)*(self.nVar*2+1),self.nVar)
#      tempDict = {}
#      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
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
#      print([key.replace('<distribution>','') for key in self.axisName])
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
#      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
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
#      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
#        self.values[varName] = copy.copy(float(gradVect[varIndex]))


  def localFinalizeActualSampling(self,jobObject,model,myInput):
    '''generate representation of goal function'''
    pass
#
#
#
class MonteCarlo(Sampler):
  '''MONTE CARLO Sampler'''
  def __init__(self):
    Sampler.__init__(self)
    self.printTag = returnPrintTag('SAMPLER MONTECARLO')

  def localInputAndChecks(self,xmlNode):
    if 'limit' in xmlNode.attrib.keys():
      try: self.limit = int(xmlNode.attrib['limit'])
      except ValueError:
        IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '-> reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    else:
      raise IOError(' Monte Carlo sampling needs the attribute limit (number of samplings)')

  def localGenerateInput(self,model,myInput):
    '''set up self.inputInfo before being sent to the model'''
    # create values dictionary
    for key in self.distDict:
      # check if the key is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
      rvsnum = self.distDict[key].rvs()
      for kkey in key.strip().split(','):
        self.values[kkey] = copy.deepcopy(rvsnum)
        self.inputInfo['SampledVarsPb'][kkey] = self.distDict[key].pdf(self.values[kkey])
      #self.values[key] = self.distDict[key].rvs()
      #self.inputInfo['SampledVarsPb'][key] = self.distDict[key].cdf(self.values[key])
    if len(self.inputInfo['SampledVarsPb'].keys()) > 0:
      self.inputInfo['PointProbability'  ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
      #self.inputInfo['ProbabilityWeight' ] = 1.0 #MC weight is 1/N => weight is one
    self.inputInfo['SamplerType'] = 'MC'
#
#
#
class Grid(Sampler):
  '''
  Samples the model on a given (by input) set of points
  '''
  def __init__(self):
    Sampler.__init__(self)
    self.printTag = returnPrintTag('SAMPLER GRID')
    self.gridCoordinate       = []    # the grid point to be used for each distribution (changes at each step)
    self.axisName             = []    # the name of each axis (variable)
    self.gridInfo             = {}    # {'name of the variable':('Type',Construction,[values])} gridType: Probability/Value, gridConstruction:Custom/Equal
    self.externalgGridCoord   = False # boolean attribute. True if the coordinate list has been filled by external source (see factorial sampler)
    #gridInfo[var][0] is type, ...[1] is construction, ...[2] is values

  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    if 'limit' in xmlNode.attrib.keys(): raise IOError('limit is not used in Grid sampler')
    self.limit = 1
    if not self.axisName: self.axisName = []
    for child in xmlNode:
      if child.tag == "Distribution":
        #Add <distribution> to name so we know it is not a direct variable
        varName = "<distribution>"+child.attrib['name']
      elif child.tag == "variable":
        varName = child.attrib['name']
      for childChild in child:
        if childChild.tag =='grid':
          self.axisName.append(varName)
          constrType = childChild.attrib['construction']
          if constrType == 'custom':
            tempList = [float(i) for i in childChild.text.split()]
            tempList.sort()
            self.gridInfo[varName] = (childChild.attrib['type'],constrType,tempList)
            if self.gridInfo[varName][0]!='value' and self.gridInfo[varName][0]!='CDF': raise IOError (self.printTag+': ' +returnPrintPostTag('ERROR') + '->The type of grid is neither value nor CDF')
            self.limit = len(tempList)*self.limit
          elif constrType == 'equal':
            self.limit = self.limit*(int(childChild.attrib['steps'])+1)
            if   'lowerBound' in childChild.attrib.keys():
              self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['lowerBound']) + float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
              self.gridInfo[varName][2].sort()
            elif 'upperBound' in childChild.attrib.keys():
              self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['upperBound']) - float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
              self.gridInfo[varName][2].sort()
            else: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> no upper or lower bound has been declared for '+str(child.tag)+' in sampler '+str(self.name))
          else: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> not specified the grid construction type')
    if len(self.toBeSampled.keys()) != len(self.gridInfo.keys()): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> inconsistency between number of variables and grid specification')
    self.gridCoordinate = [None]*len(self.axisName)

  def localAddInitParams(self,tempDict):
    for variable in self.gridInfo.items():
      tempList = [str(i) for i in variable[1][2]]
      tempDict[variable[0]+' is sampled using the grid'] = variable[1][0]+' with spacing '+variable[1][1]+', points: '+' '.join(tempList)

  def localAddCurrentSetting(self,tempDict):
    for var, value in zip(self.axisName, self.gridCoordinate):
      tempDict['coordinate '+var+' has value'] = value

  def localInitialize(self):
    '''
    This is used to check if the points and bounds are compatible with the distribution provided.
    It could not have been done earlier since the distribution might not have been initialized first
    '''
    for varName in self.gridInfo.keys():
      if self.gridInfo[varName][0]=='value':
        valueMax, indexMax = max(self.gridInfo[varName][2]), self.gridInfo[varName][2].index(max(self.gridInfo[varName][2]))
        valueMin, indexMin = min(self.gridInfo[varName][2]), self.gridInfo[varName][2].index(min(self.gridInfo[varName][2]))
        if self.distDict[varName].upperBoundUsed:
          if valueMax>self.distDict[varName].upperBound and valueMax-2.0*np.finfo(valueMax).eps>self.distDict[varName].upperBound:
            raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> the variable '+varName+'can not be sampled at '+str(valueMax)+' since outside the upper bound of the chosen distribution,Distripution Upper Bound = '+ str(self.distDict[varName].upperBound))
          if valueMax>self.distDict[varName].upperBound and valueMax-2.0*np.finfo(valueMax).eps<=self.distDict[varName].upperBound:
            valueMax = valueMax-2.0*np.finfo(valueMax).eps
        if self.distDict[varName].lowerBoundUsed:
          if valueMin<self.distDict[varName].lowerBound and valueMin+2.0*np.finfo(valueMin).eps<self.distDict[varName].lowerBound:
            raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> the variable '+varName+'can not be sampled at '+str(valueMin)+' since outside the lower bound of the chosen distribution,Distripution Lower Bound = '+str(self.distDict[varName].lowerBound))
          if valueMin<self.distDict[varName].lowerBound and valueMin+2.0*np.finfo(valueMax).eps>=self.distDict[varName].lowerBound:
            valueMin = valueMin-2.0*np.finfo(valueMin).eps
        self.gridInfo[varName][2][indexMax], self.gridInfo[varName][2][indexMin] = valueMax, valueMin

  def localGenerateInput(self,model,myInput):
    remainder = self.counter - 1 #used to keep track as we get to smaller strides
    stride = self.limit+1 #How far apart in the 1D array is the current gridCoordinate
    #self.inputInfo['distributionInfo'] = {}
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    weight = 1.0
    for i in range(len(self.gridCoordinate)):
      varName = self.axisName[i]
      if not self.externalgGridCoord:
        stride = stride // len(self.gridInfo[varName][2])
        #index is the index into the array self.gridInfo[varName][2]
        index, remainder = divmod(remainder, stride )
        self.gridCoordinate[i] = index
      # check if the varName is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
      for kkey in varName.strip().split(','):
        self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
        self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
        if self.gridInfo[varName][0]=='CDF':
          self.values[kkey] = self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]])
          self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
        elif self.gridInfo[varName][0]=='value':
          self.values[kkey] = self.gridInfo[varName][2][self.gridCoordinate[i]]
          self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
        else: raise IOError (self.gridInfo[varName][0]+' is not know as value keyword for type. Sampler: '+self.name)
      if self.gridInfo[varName][0]=='CDF':
        if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
        if self.gridCoordinate[i] == 0: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(0))/2.0)
        if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(1))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
      else:
        if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
        if self.gridCoordinate[i] == 0: weight *= self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].lowerBound)/2.0)
        if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].upperBound)/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = copy.deepcopy(weight)
    self.inputInfo['SamplerType'] = 'Grid'
#
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
    self.printTag = returnPrintTag('SAMPLER LHS')
  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)
    pointByVar  = [len(self.gridInfo[variable][2]) for variable in self.gridInfo.keys()]
    if len(set(pointByVar))!=1: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> the latin Hyper Cube requires the same number of point in each dimension')
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
    weight = 1.0
    for varName in self.axisName:
      upper = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]+1]
      lower = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]  ]
      j +=1
      intervalFraction = Distributions.random()
      coordinate = lower + (upper-lower)*intervalFraction
      # check if the varName is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
      if self.gridInfo[varName][0] =='CDF':
        ppfvalue = self.distDict[varName].ppf(coordinate)
        ppflower = self.distDict[varName].ppf(min(upper,lower))
        ppfupper = self.distDict[varName].ppf(max(upper,lower))
      for kkey in varName.strip().split(','):
        self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
        self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
        if self.gridInfo[varName][0] =='CDF':
          self.values[kkey] = copy.deepcopy(ppfvalue)
          self.inputInfo['upper'][kkey] = copy.deepcopy(ppfupper)
          self.inputInfo['lower'][kkey] = copy.deepcopy(ppflower)
          self.inputInfo['SampledVarsPb'][varName] = coordinate
          weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
        elif self.gridInfo[varName][0]=='value':
          self.values[varName] = coordinate
          self.inputInfo['upper'][kkey] = max(upper,lower)
          self.inputInfo['lower'][kkey] = min(upper,lower)
          self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
      if self.gridInfo[varName][0] =='CDF': weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
      else: weight *= self.distDict[varName].cdf(upper) - self.distDict[varName].cdf(lower)
    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = copy.deepcopy(weight)
    self.inputInfo['SamplerType'] = 'Stratified'
#
#
#
#
class DynamicEventTree(Grid):
  '''
  DYNAMIC EVENT TREE Sampler - "ANalysis of Dynamic REactor Accident evolution" module (DET      ) :D
  '''
  def __init__(self):
    Grid.__init__(self)
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
    # mapping from jobID to rootname in TreeInfo {jobID:rootName}
    self.rootToJob               = {}
    # dictionary of preconditioner sampler available
    self.preconditionerAvail = {}
    self.preconditionerAvail['MonteCarlo'] = MonteCarlo      # MC
    self.preconditionerAvail['Stratified'] = LHS             # Stratified
    self.preconditionerAvail['Grid'      ] = Grid            # Grid
    # dictionary of inputted preconditioners need to be applied
    self.preconditionerToApply             = {}
    # total number of preconditioner samples (combination of all different preconditioner strategy)
    self.precNumberSamplers                = 0
    self.printTag = returnPrintTag('SAMPLER DYNAMIC ET')

  def _localWhatDoINeed(self):
    needDict = {}
    for preconditioner in self.preconditionerToApply.values():
      preneedDict = preconditioner.whatDoINeed()
      for key,value in preneedDict.items():
        if key not in needDict.keys(): needDict[key] = []
        needDict[key] = needDict[key] + value
    return needDict

  def localStillReady(self, ready): #,lastOutput=None
    '''
    Function that inquires if there is at least an input the in the queue that needs to be run
    @ In, None
    @ Out, boolean
    '''
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0): ready = True
    else:
      if self.print_end_xml:
        myFile = open(os.path.join(self.workingDir,self.name + "_output_summary.xml"),'w')
        for treeNode in self.TreeInfo.values(): treeNode.writeNodeTree(myFile)
        myFile.close()
      ready = False
    return ready

  def _retrieveParentNode(self,idj):
    if(idj == self.TreeInfo[self.rootToJob[idj]].getrootnode().name): parentNode = self.TreeInfo[self.rootToJob[idj]].getrootnode()
    else: parentNode = list(self.TreeInfo[self.rootToJob[idj]].getrootnode().iter(idj))[0]
    return parentNode

  def localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=True):
    '''
    General function (available to all samplers) that finalize the sampling calculation just ended
    In this case (DET), The function reads the information from the ended calculation, updates the
    working variables, and creates the new inputs for the next branches
    @ In, jobObject: JobHandler Instance of the job (run) just finished
    @ In, model        : Model Instance... It may be a Code Instance, a ROM, etc.
    @ In, myInput      : List of the original input files
    @ In, genRunQueue  : bool, generated Running queue at the end of the finalization?
    @ Out, None
    '''
    self.workingDir = model.workingDir

#     returnBranchInfo = self.__readBranchInfo(jobObject.output)
    # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    parentNode = self._retrieveParentNode(jobObject.identifier)
    # set runEnded and running to true and false respectively
    parentNode.add('runEnded',True)
    parentNode.add('running',False)
    parentNode.add('end_time',self.actual_end_time)
    # Read the branch info from the parent calculation (just ended calculation)
    # This function stores the information in the dictionary 'self.actualBranchInfo'
    # If no branch info, this history is concluded => return
    if not self.__readBranchInfo(jobObject.output):
      parentNode.add('completedHistory', True)
      return False
    # Collect the branch info in a multi-level dictionary
    endInfo = {'end_time':self.actual_end_time,'end_ts':self.actual_end_ts,'branch_dist':list(self.actualBranchInfo.keys())[0]}
    endInfo['branch_changed_params'] = self.actualBranchInfo[endInfo['branch_dist']]
    parentNode.add('actual_end_ts',copy.deepcopy(self.actual_end_ts))
#     # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
#     if(jobObject.identifier == self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().name): endInfo['parent_node'] = self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode()
#     else: endInfo['parent_node'] = list(self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().iter(jobObject.identifier))[0]
    endInfo['parent_node'] = parentNode
    # get the branchedLevel dictionary
    branchedLevel = {}
    for distk, distpb in zip(endInfo['parent_node'].get('initiator_distribution'),endInfo['parent_node'].get('PbThreshold')): branchedLevel[distk] = index(self.branchProbabilities[distk],distpb)
    if not branchedLevel: raise Exception(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> branchedLevel of node '+jobObject.identifier+'not found!!!!')
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
#     # set runEnded and running to true and false respectively
#     endInfo['parent_node'].add('runEnded',True)
#     endInfo['parent_node'].add('running',False)
#     endInfo['parent_node'].add('end_time',self.actual_end_time)
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
    # Create the inputs and put them in the runQueue dictionary (if genRunQueue is true)
    if genRunQueue: self._createRunningQueue(model,myInput)
    return True

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
    branch_present = False
    self.actualBranchInfo = {}
    # Construct the file name adding the out_base root if present
    if out_base: filename = out_base + "_actual_branch_info.xml"
    else: filename = "actual_branch_info.xml"
    if not os.path.isabs(filename): filename = os.path.join(self.workingDir,filename)
    if not os.path.exists(filename):
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> branch info file ' + os.path.basename(filename) +' has not been found. => No Branching.')
      return branch_present
    # Parse the file and create the xml element tree object
    #try:
    branch_info_tree = ET.parse(filename)
    print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Done parsing '+filename)
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

  def _createRunningQueueBeginOne(self,rootTree,branchedLevel, model,myInput):
    precSampled = rootTree.getrootnode().get('preconditionerSampled')
    rootnode    =  rootTree.getrootnode()
    rname       = rootnode.name
    rootnode.add('completedHistory', False)
    # Fill th values dictionary in
    if precSampled: self.inputInfo['preconditionerCoordinate'  ] = copy.deepcopy(precSampled)
    self.inputInfo['prefix'                    ] = copy.deepcopy(rname.encode())
    self.inputInfo['initiator_distribution'    ] = []
    self.inputInfo['PbThreshold'               ] = []
    self.inputInfo['ValueThreshold'            ] = []
    self.inputInfo['branch_changed_param'      ] = [b'None']
    self.inputInfo['branch_changed_param_value'] = [b'None']
    self.inputInfo['start_time'                ] = b'Initial'
    self.inputInfo['end_ts'                    ] = 0
    self.inputInfo['parent_id'                 ] = 'root'
    self.inputInfo['conditional_prb'           ] = [1.0]
    self.inputInfo['conditional_pb'            ] = 1.0
    for key in self.branchProbabilities.keys():self.inputInfo['initiator_distribution'].append(copy.deepcopy(key.encode()))
    for key in self.branchProbabilities.keys():self.inputInfo['PbThreshold'].append(copy.deepcopy(self.branchProbabilities[key][branchedLevel[key]]))
    for key in self.branchProbabilities.keys():self.inputInfo['ValueThreshold'].append(copy.deepcopy(self.branchValues[key][branchedLevel[key]]))
    for varname in self.toBeSampled.keys():
      self.inputInfo['SampledVars'  ][varname] = copy.deepcopy(self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]])
      self.inputInfo['SampledVarsPb'][varname] = copy.deepcopy(self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]])
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]

    if(self.maxSimulTime): self.inputInfo['end_time'] = self.maxSimulTime
    # Call the model function "createNewInput" with the "values" dictionary just filled.
    # Add the new input path into the RunQueue system
    newInputs = model.createNewInput(myInput,self.type,**self.inputInfo)
    for key,value in self.inputInfo.items(): rootnode.add(key,copy.deepcopy(value))
    self.RunQueue['queue'].append(copy.deepcopy(newInputs))
    self.RunQueue['identifiers'].append(copy.deepcopy(self.inputInfo['prefix'].encode()))
    self.rootToJob[copy.deepcopy(self.inputInfo['prefix'])] = copy.deepcopy(rname)
    del newInputs
    self.counter += 1

  def _createRunningQueueBegin(self,model,myInput):
    # We construct the input for the first DET branch calculation'
    # Increase the counter
    # The root name of the xml element tree is the starting name for all the branches
    # (this root name = the user defined sampler name)
    # Get the initial branchedLevel dictionary (=> the list gets empty)
    branchedLevel = copy.deepcopy(self.branchedLevel.pop(0))
    for rootTree in self.TreeInfo.values(): self._createRunningQueueBeginOne(rootTree,branchedLevel, model,myInput)
    return

  def _createRunningQueueBranch(self,model,myInput):
    # The first DET calculation branch has already been run'
    # Start the manipulation:

    #  Pop out the last endInfo information and the branchedLevel
    branchedLevelParent     = self.branchedLevel.pop(0)
    endInfo                 = self.endInfo.pop(0)
    self.branchCountOnLevel = 0 #?
    # n_branches = number of branches need to be run
    n_branches = endInfo['n_branches']
    # Check if the distribution that just triggered hitted the last probability threshold .
    # In case we create a number of branches = endInfo['n_branches'] - 1 => the branch in
    # which the event did not occur is not going to be tracked
    if branchedLevelParent[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']]):
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Branch ' + endInfo['parent_node'].get('name') + ' hit last Threshold for distribution ' + endInfo['branch_dist'])
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Branch ' + endInfo['parent_node'].get('name') + ' is dead end.')
      self.branchCountOnLevel = 1
      n_branches = endInfo['n_branches'] - 1

    # Loop over the branches for which the inputs must be created
    for _ in range(n_branches):
      del self.inputInfo
      self.counter += 1
      self.branchCountOnLevel += 1
      branchedLevel = copy.deepcopy(branchedLevelParent)
      # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
      rname = copy.deepcopy(endInfo['parent_node'].get('name') + '-' + str(self.branchCountOnLevel))

      # create a subgroup that will be appended to the parent element in the xml tree structure
      subGroup = ETS.Node(rname.encode())
      subGroup.add('parent', endInfo['parent_node'].get('name'))
      subGroup.add('name', rname)
      subGroup.add('completedHistory', False)
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
          cond_pb_c = cond_pb_c + copy.deepcopy(endInfo['branch_changed_params'][key]['changed_cond_pb'][self.branchCountOnLevel-2])
        else:
          subGroup.add('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['old_value']))
          subGroup.add('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_pb']))
          cond_pb_un =  cond_pb_un + copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_cond_pb'])
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
      # branch calculation info... running, queue, etc are set here
      subGroup.add('runEnded',False)
      subGroup.add('running',False)
      subGroup.add('queue',True)
#        subGroup.set('restartFileRoot',endInfo['restartRoot'])
      # Append the new branch (subgroup) info to the parent_node in the tree object
      endInfo['parent_node'].appendBranch(subGroup)
      # Fill the values dictionary that will be passed into the model in order to create an input
      # In this dictionary the info for changing the original input is stored
      self.inputInfo = {'prefix':copy.deepcopy(rname.encode()),'end_ts':copy.deepcopy(endInfo['end_ts']),
                'branch_changed_param':copy.deepcopy([subGroup.get('branch_changed_param')]),
                'branch_changed_param_value':copy.deepcopy([subGroup.get('branch_changed_param_value')]),
                'conditional_prb':copy.deepcopy([subGroup.get('conditional_pb')]),
                'start_time':copy.deepcopy(endInfo['parent_node'].get('end_time')),
                'parent_id':subGroup.get('parent')}
      # add the newer branch name to the map
      self.rootToJob[copy.deepcopy(rname)] = copy.deepcopy(self.rootToJob[subGroup.get('parent')])
      # check if it is a preconditioned DET sampling, if so add the relative information
      precSampled = endInfo['parent_node'].get('preconditionerSampled')
      if precSampled:
        self.inputInfo['preconditionerCoordinate'] = copy.deepcopy(precSampled)
        subGroup.add('preconditionerSampled', precSampled)
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
        self.inputInfo['SampledVars'][varname]   = copy.deepcopy(self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]])
        self.inputInfo['SampledVarsPb'][varname] = copy.deepcopy(self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]])
      self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())*subGroup.get('conditional_pb')
      self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
      # Call the model function "createNewInput" with the "values" dictionary just filled.
      # Add the new input path into the RunQueue system
      self.RunQueue['queue'].append(copy.deepcopy(model.createNewInput(myInput,self.type,**self.inputInfo)))
      self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
      for key,value in self.inputInfo.items(): subGroup.add(key,copy.deepcopy(value))
      popped = endInfo.pop('parent_node')
      subGroup.add('endInfo',copy.deepcopy(endInfo))
      endInfo['parent_node'] = popped
      del branchedLevel

  def _createRunningQueue(self,model,myInput):
    '''
    Function to create and append new inputs to the queue. It uses all the containers have been updated by the previous functions
    @ In, model  : Model instance. It can be a Code type, ROM, etc.
    @ In, myInput: List of the original inputs
    @ Out, None
    '''
    if self.counter >= 1:
      # The first DET calculation branch has already been run
      # Start the manipulation:
      #  Pop out the last endInfo information and the branchedLevel
      self._createRunningQueueBranch(model, myInput)
    else:
      # We construct the input for the first DET branch calculation'
      self._createRunningQueueBegin(model, myInput)
    return

  def __getQueueElement(self):
    '''
    Function to get an input from the internal queue system
    @ In, None
    @ Out, jobInput: First input in the queue
    '''
    # Pop out the first input in queue
    jobInput  = self.RunQueue['queue'      ].pop(0)
    jobId     = self.RunQueue['identifiers'].pop(0)
    #set running flags in self.TreeInfo
    root = self.TreeInfo[self.rootToJob[jobId]].getrootnode()
    # Update the run information flags
    if (root.name == jobId):
      root.add('runEnded',False)
      root.add('running',True)
      root.add('queue',False)
    else:
      subElm = list(root.iter(jobId))[0]
      if(subElm):
        subElm.add('runEnded',False)
        subElm.add('running',True)
        subElm.add('queue',False)
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
      self._createRunningQueue(model, myInput)
    # retrieve the input from the queue
    newerinput = self.__getQueueElement()
    if not newerinput:
      # If no inputs are present in the queue => a branch is finished
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> A Branch ended!!!!')
    return newerinput

  def _generateDistributions(self,availableDist):
    Grid._generateDistributions(self,availableDist)
    for preconditioner in self.preconditionerToApply.values(): preconditioner._generateDistributions(availableDist)

  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)
    self.limit = sys.maxsize
    if 'print_end_xml' in xmlNode.attrib.keys():
      if xmlNode.attrib['print_end_xml'].lower() in stringsThatMeanTrue(): self.print_end_xml = True
      else: self.print_end_xml = False
    if 'maxSimulationTime' in xmlNode.attrib.keys():
      try:    self.maxSimulTime = float(xmlNode.attrib['maxSimulationTime'])
      except (KeyError,NameError): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Can not convert maxSimulationTime in float number!!!')
    for child in xmlNode:
      if child.tag == 'PreconditionerSampler':
        if not 'type' in child.attrib.keys()                          : raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Not found attribute type in PreconditionerSampler block!')
        if child.attrib['type'] in self.preconditionerToApply.keys()  : raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> PreconditionerSampler type '+child.attrib['type'] + ' already inputted!')
        if child.attrib['type'] not in self.preconditionerAvail.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> PreconditionerSampler type' +child.attrib['type'] + 'unknown. Available are '+ str(self.preconditionerAvail.keys()).replace("[","").replace("]",""))
        self.precNumberSamplers = 1
        # the user can decided how to preconditionate
        self.preconditionerToApply[child.attrib['type']] = self.preconditionerAvail[child.attrib['type']]()
        # make the preconditioner sampler read  its own xml block
        self.preconditionerToApply[child.attrib['type']]._readMoreXML(child)
    branchedLevel = {}
    error_found = False
    for keyk in self.axisName:
      branchedLevel[self.toBeSampled[keyk]] = 0
      if self.gridInfo[keyk][0] == 'CDF':
        self.branchProbabilities[self.toBeSampled[keyk]] = self.gridInfo[keyk][2]
        self.branchProbabilities[self.toBeSampled[keyk]].sort(key=float)
        if max(self.branchProbabilities[self.toBeSampled[keyk]]) > 1:
          print(self.printTag+": ERROR -> One of the Thresholds for distribution " + str(self.gridInfo[keyk][2]) + " is > 1")
          error_found = True
          for index in range(len(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float))):
            if sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float).count(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index]) > 1:
              print(self.printTag+": ERROR -> In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
              error_found = True
      else:
        self.branchValues[self.toBeSampled[keyk]] = self.gridInfo[keyk][2]
        self.branchValues[self.toBeSampled[keyk]].sort(key=float)
        for index in range(len(sorted(self.branchValues[self.toBeSampled[keyk]], key=float))):
          if sorted(self.branchValues[self.toBeSampled[keyk]], key=float).count(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index]) > 1:
            print(self.printTag+": ERROR -> In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
            error_found = True
    if error_found: raise IOError(self.printTag+": ERROR -> In sampler named " + self.name+' ERRORS have been found!!!' )
    # Append the branchedLevel dictionary in the proper list
    self.branchedLevel.append(branchedLevel)

  def localAddInitParams(self,tempDict):
    for key in self.branchProbabilities.keys(): tempDict['Probability Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchProbabilities[key]]
    for key in self.branchValues.keys()       : tempDict['Values Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchValues[key]]

  def localAddCurrentSetting(self,tempDict):
    tempDict['actual threshold levels are '] = self.branchedLevel[0]

  def localInitialize(self):
    if len(self.preconditionerToApply.keys()) > 0: precondlistoflist = []
    for cnt, preckey  in enumerate(self.preconditionerToApply.keys()):
      preconditioner =  self.preconditionerToApply[preckey]
      precondlistoflist.append([])
      preconditioner.initialize()
      self.precNumberSamplers *= preconditioner.limit
      while preconditioner.amIreadyToProvideAnInput():
        preconditioner.counter +=1
        preconditioner.localGenerateInput(None,None)
        preconditioner.inputInfo['prefix'] = copy.deepcopy(preconditioner.counter)
        precondlistoflist[cnt].append(copy.deepcopy(preconditioner.inputInfo))
    if self.precNumberSamplers > 0:
      print(self.printTag+': ' +returnPrintPostTag('Message') + '-> Number of Preconditioner Samples are ' + str(self.precNumberSamplers) + '!')
      precNumber = self.precNumberSamplers
      combinations = list(itertools.product(*precondlistoflist))
    else: precNumber = 1
    self.TreeInfo = {}
    for precSample in range(precNumber):
      elm = ETS.Node(self.name + '_' + str(precSample+1))
      elm.add('name', self.name + '_'+ str(precSample+1))
      elm.add('start_time', str(0.0))
      # Initialize the end_time to be equal to the start one...
      # It will modified at the end of each branch
      elm.add('end_time', str(0.0))
      elm.add('runEnded',False)
      elm.add('running',True)
      elm.add('queue',False)
      # if preconditioned DET, add the sampled from preconditioner samplers
      if self.precNumberSamplers > 0: elm.add('preconditionerSampled', combinations[precSample])
      # The dictionary branchedLevel is stored in the xml tree too. That's because
      # the advancement of the thresholds must follow the tree structure
      elm.add('branchedLevel', self.branchedLevel[0])
      # Here it is stored all the info regarding the DET => we create the info for all the
      # branchings and we store them
      self.TreeInfo[self.name + '_' + str(precSample+1)] = ETS.NodeTree(copy.deepcopy(elm))

    for key in self.branchProbabilities.keys():
      #kk = self.toBeSampled.values().index(key)
      self.branchValues[key] = [copy.deepcopy(self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].ppf(float(self.branchProbabilities[key][index]))) for index in range(len(self.branchProbabilities[key]))]
    for key in self.branchValues.keys():
      #kk = self.toBeSampled.values().index(key)
      self.branchProbabilities[key] = [copy.deepcopy(self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].cdf(float(self.branchValues[key][index]))) for index in range(len(self.branchValues[key]))]
    return
#
#
#
#
class AdaptiveDET(DynamicEventTree, AdaptiveSampler):
  def __init__(self):
    DynamicEventTree.__init__(self)  # init DET
    AdaptiveSampler.__init__(self)   # init Adaptive
    self.detAdaptMode         = 1    # Adaptive Dynamic Event Tree method (=1 -> DynamicEventTree as preconditioner and subsequent Adaptive,=2 -> DynamicEventTree online adaptive)
    self.noTransitionStrategy = 1    # Strategy in case no transitions have been found by DET (1 = 'Probability MC', 2 = Increase the grid exploration)
    self.insertAdaptBPb       = True # Add Probabability THs requested by adaptive in the initial grid (default = False)
    self.startAdaptive = False
    self.printTag = returnPrintTag('SAMPLER ADAPTIVE DET')
    self.adaptiveReady = False
    self.investigatedPoints = []
    self.completedHistCnt   = 1
    self.actualLastOutput   = None
  @staticmethod
  def _checkIfRunnint(treeValues): return not treeValues['runEnded']
  @staticmethod
  def _checkEnded(treeValues): return treeValues['runEnded']
  @staticmethod
  def _checkCompleteHistory(treeValues): return treeValues['completedHistory']

  def _checkIfStartAdaptive(self):
    '''
    Function that checks if the adaptive needs to be started (mode 1)
    @ In, None
    @ Out, None
    '''
    if not self.startAdaptive:
      self.startAdaptive = True
      for treer in self.TreeInfo.values():
        for _ in treer.iterProvidedFunction(self._checkIfRunnint):
          self.startAdaptive = False
          break
        if not self.startAdaptive: break

  def _checkClosestBranch(self):
    '''
    Function that checks the closest branch already evaluated
    @ In, None
    @ Out, dict, key:gridPosition
    '''
    # compute cdf of sampled vars
    lowerCdfValues = {}
    cdfValues         = {}
    for key,value in self.values.items():
      cdfValues[key] = self.distDict[key].cdf(value)
      lowerCdfValues[key] = find_le(self.branchProbabilities[self.toBeSampled[key]],cdfValues[key])[0]
      if self.debug:
        print(self.printTag+': ' +returnPrintPostTag('Message') + '-> ' + str(self.toBeSampled[key]))
        print(self.printTag+': ' +returnPrintPostTag('Message') + '-> ' + str(value))
        print(self.printTag+': ' +returnPrintPostTag('Message') + '-> ' + str(cdfValues[key]))
        print(self.printTag+': ' +returnPrintPostTag('Message') + '-> ' + str(lowerCdfValues[key]))
    # check if in the adaptive points already explored (if not pushed into the grid)
    if not self.insertAdaptBPb:
      candidatesBranch = []
      # check if adaptive point is better choice -> TODO: improve efficiency
      for invPoint in self.investigatedPoints:
        pbth = [invPoint[self.toBeSampled[key]] for key in cdfValues.keys()]
        if all(i <= pbth[cnt] for cnt,i in enumerate(cdfValues.values())): candidatesBranch.append(invPoint)
      if len(candidatesBranch) > 0:
        if None in lowerCdfValues.values(): lowerCdfValues = candidatesBranch[0]
        for invPoint in candidatesBranch:
          pbth = [invPoint[self.toBeSampled[key]] for key in cdfValues.keys()]
          if all(i >= pbth[cnt] for cnt,i in enumerate(lowerCdfValues.values())): lowerCdfValues = invPoint
    # Check if The adaptive point requested is outside the so far run grid; in case return None
    if None in lowerCdfValues.values(): return None,cdfValues
    nntrain = None
    mapping = {}
    for treer in self.TreeInfo.values(): # this needs to be solved
      for ending in treer.iterProvidedFunction(self._checkEnded):
        #already ended branches, create training set for nearest algorithm (take coordinates <= of cdfValues) -> TODO: improve efficiency
        pbth = [ending.get('SampledVarsPb')[key] for key in lowerCdfValues.keys()]
        if all(pbth[cnt] <= i for cnt,i in enumerate(lowerCdfValues.values())):
          if nntrain == None:
            nntrain = np.zeros((1,len(cdfValues.keys())))
            nntrain[0,:] = np.array(copy.copy(pbth))
          else          :
            nntrain = np.concatenate((nntrain,np.atleast_2d(np.array(copy.copy(pbth)))),axis=0)
            #nntrain = np.append(nntrain, np.atleast_1d(np.array(copy.copy(pbth))), axis=0)
          mapping[nntrain.shape[0]] = ending
    if nntrain != None:
      neigh = neighbors.NearestNeighbors(n_neighbors=1)
      neigh.fit(nntrain)
      return mapping[neigh.kneighbors(lowerCdfValues.values())[1][0][0]+1],cdfValues
    else: return None,cdfValues

  def _retrieveBranchInfo(self,branch):
    '''
     Function that retrieves the key information from a branch to start a newer calculation
     @ In, branch
     @ Out, dictionary with those information
    '''
    info = branch.getValues()
    info['actualBranchOnLevel'] = branch.numberBranches()
    info['parent_node']         = branch
    return info

  def _constructEndInfoFromBranch(self,model, myInput, info, cdfValues):
    endInfo = info['parent_node'].get('endInfo')
    #branchedLevel = {}
    #for distk, distpb in zip(info['initiator_distribution'],info['PbThreshold']): branchedLevel[distk] = index(self.branchProbabilities[distk],distpb)
    del self.inputInfo
    self.counter           += 1
    self.branchCountOnLevel = info['actualBranchOnLevel']+1
    # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
    rname = copy.deepcopy(info['parent_node'].get('name') + '-' + str(self.branchCountOnLevel))
    info['parent_node'].add('completedHistory', False)
    print(rname)
    bcnt = copy.deepcopy(self.branchCountOnLevel)
    while info['parent_node'].isAnActualBranch(rname):
      bcnt += 1
      rname = copy.deepcopy(info['parent_node'].get('name') + '-' + str(bcnt))
    # create a subgroup that will be appended to the parent element in the xml tree structure
    subGroup = ETS.Node(rname)
    subGroup.add('parent', info['parent_node'].get('name'))
    subGroup.add('name', rname)
    print('cond pb = '+str(info['parent_node'].get('conditional_pb')))
    cond_pb_c  = copy.deepcopy(float(info['parent_node'].get('conditional_pb')))

    # Loop over  branch_changed_params (events) and start storing information,
    # such as conditional pb, variable values, into the xml tree object
    if endInfo:
      for key in endInfo['branch_changed_params'].keys():
        subGroup.add('branch_changed_param',key)
        subGroup.add('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['old_value'][0]))
        subGroup.add('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['associated_pb'][0]))
    else:
      pass
    #cond_pb_c = cond_pb_c + copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_cond_pb'])
    # add conditional probability
    subGroup.add('conditional_pb',copy.deepcopy(cond_pb_c))
    # add initiator distribution info, start time, etc.
    #subGroup.add('initiator_distribution',copy.deepcopy(endInfo['branch_dist']))
    subGroup.add('start_time', copy.deepcopy(info['parent_node'].get('end_time')))
    # initialize the end_time to be equal to the start one... It will modified at the end of this branch
    subGroup.add('end_time', copy.deepcopy(info['parent_node'].get('end_time')))
    # add the branchedLevel dictionary to the subgroup
    #branchedLevel[endInfo['branch_dist']] = branchedLevel[endInfo['branch_dist']] - 1
    # branch calculation info... running, queue, etc are set here
    subGroup.add('runEnded',False)
    subGroup.add('running',False)
    subGroup.add('queue',True)
    subGroup.add('completedHistory', False)
    # Append the new branch (subgroup) info to the parent_node in the tree object
    info['parent_node'].appendBranch(subGroup)
    # Fill the values dictionary that will be passed into the model in order to create an input
    # In this dictionary the info for changing the original input is stored
    self.inputInfo = {'prefix':copy.deepcopy(rname),'end_ts':copy.deepcopy(info['parent_node'].get('actual_end_ts')),
              'branch_changed_param':copy.deepcopy([subGroup.get('branch_changed_param')]),
              'branch_changed_param_value':copy.deepcopy([subGroup.get('branch_changed_param_value')]),
              'conditional_prb':copy.deepcopy([subGroup.get('conditional_pb')]),
              'start_time':copy.deepcopy(info['parent_node'].get('end_time')),
              'parent_id':subGroup.get('parent')}
    # add the newer branch name to the map
    self.rootToJob[copy.deepcopy(rname)] = copy.deepcopy(self.rootToJob[subGroup.get('parent')])
#     check if it is a preconditioned DET sampling, if so add the relative information
#     precSampled = endInfo['parent_node'].get('preconditionerSampled')
#     if precSampled:
#       self.inputInfo['preconditionerCoordinate'] = copy.deepcopy(precSampled)
#       subGroup.add('preconditionerSampled', precSampled)
    # The probability Thresholds are stored here in the cdfValues dictionary... We are sure that they are whitin the ones defined in the grid
    # check is not needed
    self.inputInfo['initiator_distribution'] = copy.deepcopy([self.toBeSampled[key] for key in cdfValues.keys()])
    self.inputInfo['PbThreshold'           ] = copy.deepcopy(cdfValues.values())
    self.inputInfo['ValueThreshold'        ] = copy.deepcopy([self.distDict[key].ppf(value) for key,value in cdfValues.items()])
    self.inputInfo['SampledVars'           ] = {}
    self.inputInfo['SampledVarsPb'         ] = {}
    for varname in self.toBeSampled.keys():
      self.inputInfo['SampledVars'][varname]   = copy.deepcopy(self.distDict[varname].ppf(cdfValues[varname]))
      self.inputInfo['SampledVarsPb'][varname] = copy.deepcopy(cdfValues[varname])
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())*subGroup.get('conditional_pb')
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
    # Call the model function "createNewInput" with the "values" dictionary just filled.
    # Add the new input path into the RunQueue system
    self.RunQueue['queue'].append(copy.deepcopy(model.createNewInput(myInput,self.type,**self.inputInfo)))
    self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
    for key,value in self.inputInfo.items(): subGroup.add(key,copy.deepcopy(value))
    if endInfo: subGroup.add('endInfo',copy.deepcopy(endInfo))
    # Call the model function "createNewInput" with the "values" dictionary just filled.
    return

  def localStillReady(self,ready): #, lastOutput= None
    '''
    Function that inquires if there is at least an input the in the queue that needs to be run
    @ In, None
    @ Out, boolean
    '''
    if(self.counter == 0): return True
    if len(self.RunQueue['queue']) != 0: detReady = True
    else: detReady = False
    # since the RunQueue is empty, let's check if there are still branches running => if not => start the adaptive search
    self._checkIfStartAdaptive()
    if self.startAdaptive:
      #if self._endJobRunnable != 1: self._endJobRunnable = 1
      # retrieve the endHistory branches
      completedHistNames = []
      for treer in self.TreeInfo.values(): # this needs to be solved
        for ending in treer.iterProvidedFunction(self._checkCompleteHistory):
          completedHistNames.append(self.lastOutput.getParam(typeVar='inout',keyword='none',nodeid=ending.get('name'),serialize=False))
      # assemble a dictionary
      if len(completedHistNames) > 0:
        if len(completedHistNames[-1].values()) > 0:
          lastOutDict = {'inputs':{},'outputs':{}}
          for histd in completedHistNames:
            histdict = histd.values()[-1]
            for key in histdict['inputs' ].keys():
              if key not in lastOutDict['inputs'].keys(): lastOutDict['inputs'][key] = copy.deepcopy(np.atleast_1d(histdict['inputs'][key]))
              else                                      : lastOutDict['inputs'][key] = np.concatenate((np.atleast_1d(lastOutDict['inputs'][key]),copy.deepcopy(np.atleast_1d(histdict['inputs'][key]))))
            for key in histdict['outputs'].keys():
              if key not in lastOutDict['outputs'].keys(): lastOutDict['outputs'][key] = copy.deepcopy(np.atleast_1d(histdict['outputs'][key]))
              else                                       : lastOutDict['outputs'][key] = np.concatenate((np.atleast_1d(lastOutDict['outputs'][key]),copy.deepcopy(np.atleast_1d(histdict['outputs'][key]))))
        else: print(self.printTag+': ' +returnPrintPostTag('Warning') + '-> No Completed histories! No possible to start an adaptive search! Something went wrongly!')
      if len(completedHistNames) > self.completedHistCnt:
        self.actualLastOutput = self.lastOutput
        self.lastOutput       = self.actualLastOutput
        ready = AdaptiveSampler.localStillReady(self,ready)
        self.completedHistCnt = len(completedHistNames)
      else: ready = False
      self.adaptiveReady = ready
      if ready or detReady and self.persistence > self.repetition : return True
      else: return False
    return detReady

  def localGenerateInput(self,model,myInput):
    if self.startAdaptive:
      AdaptiveSampler.localGenerateInput(self,model,myInput)
      #the adaptive sampler created the next point sampled vars
      #find the closest branch
      closestBranch, cdfValues = self._checkClosestBranch()
      if not closestBranch: print(self.printTag+': ' +returnPrintPostTag('Message') + '-> An usable branch for next candidate has not been found => create a parallel branch!')
      # add pbthresholds in the grid
      investigatedPoint = {}
      for key,value in cdfValues.items():
#         if self.insertAdaptBPb:
#           ind = find_le_index(self.branchProbabilities[self.toBeSampled[key]],value)
#           if not ind: ind = 0
#           if value not in self.branchProbabilities[self.toBeSampled[key]]:
#             self.branchProbabilities[self.toBeSampled[key]].insert(ind,value)
#             self.branchValues[self.toBeSampled[key]].insert(ind,self.distDict[key].ppf(value))

        ind = find_le_index(self.branchProbabilities[self.toBeSampled[key]],value)
        if not ind: ind = 0
        if value not in self.branchProbabilities[self.toBeSampled[key]]:
          self.branchProbabilities[self.toBeSampled[key]].insert(ind,value)
          self.branchValues[self.toBeSampled[key]].insert(ind,self.distDict[key].ppf(value))
        investigatedPoint[self.toBeSampled[key]] = value
      # collect investigated point
      self.investigatedPoints.append(copy.deepcopy(investigatedPoint))

      if closestBranch:
        info = self._retrieveBranchInfo(closestBranch)
        self._constructEndInfoFromBranch(model, myInput, info, cdfValues)
      else:
        # create a new tree, since there are no branches that are close enough to the adaptive request
        elm = ETS.Node(self.name + '_' + str(len(self.TreeInfo.keys())+1))
        elm.add('name', self.name + '_'+ str(len(self.TreeInfo.keys())+1))
        elm.add('start_time', 0.0)

        # Initialize the end_time to be equal to the start one...
        # It will modified at the end of each branch
        elm.add('end_time', 0.0)
        elm.add('runEnded',False)
        elm.add('running',True)
        elm.add('queue',False)
        elm.add('completedHistory', False)
        branchedLevel = {}
        for key,value in cdfValues.items():
          branchedLevel[self.toBeSampled[key]] = index(self.branchProbabilities[self.toBeSampled[key]],value)
        # The dictionary branchedLevel is stored in the xml tree too. That's because
        # the advancement of the thresholds must follow the tree structure
        elm.add('branchedLevel', branchedLevel)
        # Here it is stored all the info regarding the DET => we create the info for all the branchings and we store them
        self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys())+1)] = ETS.NodeTree(copy.deepcopy(elm))
        #self.branchedLevel.append(branchedLevel)
        self._createRunningQueueBeginOne(self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys()))],branchedLevel, model,myInput)
        #self._createRunningQueueBegin(model,myInput)
    return DynamicEventTree.localGenerateInput(self,model,myInput)

  def localInputAndChecks(self,xmlNode):
    DynamicEventTree.localInputAndChecks(self,xmlNode)
    AdaptiveSampler.localInputAndChecks(self,xmlNode)
    if 'mode' in xmlNode.attrib.keys():
      if xmlNode.attrib['mode'].lower() == 'online': self.detAdaptMode = 2
      elif xmlNode.attrib['mode'].lower() == 'post': self.detAdaptMode = 1
      else:  raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> unknown mode '+xmlNode.attrib['mode']+'. Available are "online" and "post"!')
    if 'noTransitionStrategy' in xmlNode.attrib.keys():
      if xmlNode.attrib['noTransitionStrategy'].lower() == 'mc'    : self.noTransitionStrategy = 1
      elif xmlNode.attrib['noTransitionStrategy'].lower() == 'grid': self.noTransitionStrategy = 2
      else:  raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> unknown noTransitionStrategy '+xmlNode.attrib['noTransitionStrategy']+'. Available are "mc" and "grid"!')
    if 'updateGrid' in xmlNode.attrib.keys():
      if xmlNode.attrib['updateGrid'].lower() in stringsThatMeanTrue(): self.insertAdaptBPb = True

  def _generateDistributions(self,availableDist):
    DynamicEventTree._generateDistributions(self,availableDist)

  def localInitialize(self,solutionExport = None,goalFunction = None,ROM = None):
    if self.detAdaptMode == 2: self.startAdaptive = True
    DynamicEventTree.localInitialize(self)
    AdaptiveSampler.localInitialize(self,solutionExport=solutionExport,goalFunction=goalFunction,ROM=ROM)
    self._endJobRunnable    = sys.maxsize

  def generateInput(self,model,oldInput):
    return DynamicEventTree.generateInput(self, model, oldInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    returncode = DynamicEventTree.localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=False)
    if returncode: self._createRunningQueue(model,myInput)
#
#
#
#
class FactorialDesign(Grid):
  '''
  Samples the model on a given (by input) set of points
  '''
  def __init__(self):
    Grid.__init__(self)
    self.printTag = returnPrintTag('SAMPLER FACTORIAL DESIGN')
    # accepted types. full = full factorial, 2levelfract = 2-level fracional factorial, pb = Plackett-Burman design. NB. full factorial is equivalent to Grid sampling
    self.acceptedTypes = ['full','2levelfract','pb'] # accepted factorial types
    self.factOpt       = {}                          # factorial options (type,etc)
    self.designMatrix  = None                        # matrix container

  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    Grid.localInputAndChecks(self,xmlNode)
    factsettings = xmlNode.find("FactorialSettings")
    if factsettings == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'FactorialSettings xml node not found!!!')
    facttype = factsettings.find("type")
    if facttype == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'node "type" not found in FactorialSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedTypes:raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +' "type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedTypes))
    self.factOpt['type'] = facttype.text.lower()
    if self.factOpt['type'] == '2levelfract':
      self.factOpt['options'] = {}
      self.factOpt['options']['gen'] = factsettings.find("gen")
      self.factOpt['options']['genMap'] = factsettings.find("genMap")
      if self.factOpt['options']['gen'] == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'node "gen" not found in FactorialSettings xml node!!!')
      if self.factOpt['options']['genMap'] == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'node "genMap" not found in FactorialSettings xml node!!!')
      self.factOpt['options']['gen'] = self.factOpt['options']['gen'].text.split(',')
      self.factOpt['options']['genMap'] = self.factOpt['options']['genMap'].text.split(',')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'number of variable in genMap != number of variables !!!')
      if len(self.factOpt['options']['gen']) != len(self.gridInfo.keys())   : raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'number of variable in gen != number of variables !!!')
      rightOrder = [None]*len(self.gridInfo.keys())
      if len(self.factOpt['options']['genMap']) != len(self.factOpt['options']['gen']): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> gen and genMap different size!')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> number of gen attributes and variables different!')
      for ii,var in enumerate(self.factOpt['options']['genMap']):
        if var not in self.gridInfo.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +' variable "'+var+'" defined in genMap block not among the inputted variables!')
        rightOrder[self.axisName.index(var)] = self.factOpt['options']['gen'][ii]
      self.factOpt['options']['orderedGen'] = rightOrder
    if self.factOpt['type'] != 'full':
      self.externalgGridCoord = True
      for varname in self.gridInfo.keys():
        if len(self.gridInfo[varname][2]) != 2:
          raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +' The number of levels for type '+
                        self.factOpt['type'] +' must be 2! In variable '+varname+ ' got number of levels = ' +
                        str(len(self.gridInfo[varname][2])))
    else: self.externalgGridCoord = False

  def localAddInitParams(self,tempDict):
    Grid.localAddInitParams(self,tempDict)
    for key,value in self.factOpt.items():
      if key != 'options': tempDict['Factorial '+key] = value
      else:
        for kk,val in value.items(): tempDict['Factorial options '+kk] = val

  def localInitialize(self):
    '''
    This method initialize the factorial matrix. No actions are taken for full-factorial since it is equivalent to the Grid sampling this sampler is based on
    '''
    Grid.localInitialize(self)
    if   self.factOpt['type'] == '2levelfract': self.designMatrix = doe.fracfact(' '.join(self.factOpt['options']['orderedGen'])).astype(int)
    elif self.factOpt['type'] == 'pb'         : self.designMatrix = doe.pbdesign(len(self.gridInfo.keys())).astype(int)
    if self.designMatrix != None:
      # convert all -1 in 0 => we can access to the grid info directly
      self.designMatrix[self.designMatrix == -1] = 0
      # the limit is the number of rows
      self.limit = self.designMatrix.shape[0]

  def localGenerateInput(self,model,myInput):
    if self.factOpt['type'] == 'full':  Grid.localGenerateInput(self,model, myInput)
    else:
      self.gridCoordinate = self.designMatrix[self.counter - 1][:].tolist()
      Grid.localGenerateInput(self,model, myInput)
#
#
#
#
class ResponseSurfaceDesign(Grid):
  '''
  Samples the model on a given (by input) set of points
  '''
  def __init__(self):
    Grid.__init__(self)
    self.limit    = 1
    self.printTag = returnPrintTag('SAMPLER RESPONSE SURF DESIGN')
    self.respOpt         = {}                                    # response surface design options (type,etc)
    self.designMatrix    = None                                  # matrix container
    self.bounds          = {}                                    # dictionary of lower and upper
    self.mapping         = {}                                    # mapping between designmatrix coordinates and position in grid
    self.minNumbVars     = {'boxbehnken':3,'centralcomposite':2} # minimum number of variables
    # dictionary of accepted types and options (required True, optional False)
    self.acceptedOptions = {'boxbehnken':['ncenters'],
                            'centralcomposite':['centers','alpha','face']}

  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    # here we call the input reader of the grid, even if the grid is definded in a different way, just to collect the variable names
    # Grid.localInputAndChecks(self,xmlNode)
    factsettings = xmlNode.find("ResponseSurfaceDesignSettings")
    if factsettings == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'ResponseSurfaceDesignSettings xml node not found!!!')
    facttype = factsettings.find("type")
    if facttype == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'node "type" not found in ResponseSurfaceDesignSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedOptions.keys():raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +' "type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedOptions.keys()))
    self.respOpt['type'] = facttype.text.lower()
    # set defaults
    if self.respOpt['type'] == 'boxbehnken': self.respOpt['options'] = {'ncenters':None}
    else                                   : self.respOpt['options'] = {'centers':(4,4),'alpha':'orthogonal','face':'circumscribed'}
    for child in factsettings:
      if child.tag not in 'type': self.respOpt['options'][child.tag] = child.text.lower()
    # start checking
    for key,value in self.respOpt['options'].items():
      if key not in self.acceptedOptions[facttype.text.lower()]:
        raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'node '+key+' unknown. Available are "'+' '.join(self.acceptedOptions[facttype.text.lower()])+'"!!')
      if self.respOpt['type'] == 'boxbehnken':
        if key == 'ncenters':
          try   : self.respOpt['options'][key] = int(value)
          except: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'"'+key+'" is not an integer!')
      else:
        if key == 'centers':
          if len(value.split(',')) != 2: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'"'+key+'" must be a comma separated string of 2 values only!')
          centers = value.split(',')
          try: self.respOpt['options'][key] = (int(centers[0]),int(centers[1]))
          except: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> ' +'"'+key+'" values must be integers!!')
        if key == 'alpha':
          if value not in ['orthogonal','rotatable']: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Not recognized options for node ' +'"'+key+'". Available are "orthogonal","rotatable"!')
        if key == 'face':
          if value not in ['circumscribed','faced','inscribed']: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> Not recognized options for node ' +'"'+key+'". Available are "circumscribed","faced","inscribed"!')
    # fill in the grid
    if 'limit' in xmlNode.attrib.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> limit is not used in' +self.type+' sampler!')
    if not self.axisName: self.axisName = []
    for child in xmlNode:
      if child.tag == "Distribution": varName = "<distribution>"+child.attrib['name']
      elif child.tag == "variable"  : varName = child.attrib['name']
      for childChild in child:
        if childChild.tag =='boundaries':
          self.axisName.append(varName)
          if 'type' not in childChild.attrib.keys(): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> in block '+ childChild.tag + ' attribute type not found!')
          self.gridInfo[varName] = [childChild.attrib['type'],'custom',[]]
          lower = childChild.find("lower")
          upper = childChild.find("upper")
          if lower == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> node "lower" not found in '+childChild.tag+' block!')
          if upper == None: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> node "upper" not found in '+childChild.tag+' block!')
          try: self.bounds[varName] = (float(lower.text),float(upper.text))
          except: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> node "upper" or "lower" must be float')
    if len(self.toBeSampled.keys()) != len(self.gridInfo.keys()): raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> inconsistency between number of variables and grid specification')
    self.gridCoordinate = [None]*len(self.axisName)
    if len(self.gridCoordinate) < self.minNumbVars[self.respOpt['type']]: raise IOError(self.printTag+': ' +returnPrintPostTag('ERROR') + '-> minimum number of variables for type "'+ self.respOpt['type'] +'" is '+str(self.minNumbVars[self.respOpt['type']])+'!!')
    self.externalgGridCoord = True

  def localAddInitParams(self,tempDict):
    Grid.localAddInitParams(self,tempDict)
    for key,value in self.respOpt.items():
      if key != 'options': tempDict['Response Design '+key] = value
      else:
        for kk,val in value.items(): tempDict['Response Design options '+kk] = val

  def localInitialize(self):
    '''
    This method initialize the response matrix. No actions are taken for full-factorial since it is equivalent to the Grid sampling this sampler is based on
    '''
    if   self.respOpt['type'] == 'boxbehnken'      : self.designMatrix = doe.bbdesign(len(self.gridInfo.keys()),center=self.respOpt['options']['ncenters'])
    elif self.respOpt['type'] == 'centralcomposite': self.designMatrix = doe.ccdesign(len(self.gridInfo.keys()), center=self.respOpt['options']['centers'], alpha=self.respOpt['options']['alpha'], face=self.respOpt['options']['face'])
    for cnt, varName in enumerate(self.axisName):
      column = np.unique(self.designMatrix[:,cnt])
      yi = np.array([self.bounds[varName][0], self.bounds[varName][1]])
      xi = np.array([min(column), max(column)])
      s = InterpolatedUnivariateSpline(xi, yi, k=1)
      self.gridInfo[varName][2] = s(column).tolist()
      self.mapping[varName] = column.tolist()
    Grid.localInitialize(self)
    self.limit = self.designMatrix.shape[0]

  def localGenerateInput(self,model,myInput):
    gridcoordinate = self.designMatrix[self.counter - 1][:].tolist()
    for cnt, varName in enumerate(self.axisName): self.gridCoordinate[cnt] = self.mapping[varName].index(gridcoordinate[cnt])
    Grid.localGenerateInput(self,model, myInput)
#
#
#
#
'''
 Interface Dictionary (factory) (private)
'''
__base = 'Sampler'
__interFaceDict = {}
__interFaceDict['MonteCarlo'              ] = MonteCarlo
__interFaceDict['DynamicEventTree'        ] = DynamicEventTree
__interFaceDict['LHS'                     ] = LHS
__interFaceDict['Grid'                    ] = Grid
__interFaceDict['Adaptive'                ] = AdaptiveSampler
__interFaceDict['AdaptiveDynamicEventTree'] = AdaptiveDET
__interFaceDict['FactorialDesign'         ] = FactorialDesign
__interFaceDict['ResponseSurfaceDesign'   ] = ResponseSurfaceDesign
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
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
