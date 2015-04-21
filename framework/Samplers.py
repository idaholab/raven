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
import utils
from BaseClasses import BaseType
from Assembler import Assembler
import Distributions
import TreeStructure as ETS
import SupervisedLearning
import pyDOE as doe
import Quadratures
import OrthoPolynomials
import SupervisedLearning
import IndexSets
import PostProcessors
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

#Internal Submodules---------------------------------------------------------------------------------
#Internal Submodules End--------------------------------------------------------------------------------

class Sampler(utils.metaclass_insert(abc.ABCMeta,BaseType),Assembler):
  """
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
  """

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
    self.reseedAtEachIteration         = False                     # Logical flag. True if every newer evaluation is performed after a new reseeding
    self.FIXME                         = False                     # FIXME flag
    self.printTag                      = self.type                 # prefix for all prints (sampler type)

    self._endJobRunnable               = sys.maxsize               # max number of inputs creatable by the sampler right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)

    ######
    self.variables2distributionsMapping = {}                       # for each variable 'varName'  , the following informations are included:  'varName': {'dim': 1, 'totDim': 2, 'name': 'distName'} ; dim = dimension of the variable; totDim = total dimensionality of its associated distribution
    self.distributions2variablesMapping = {}                       # for each variable 'distName' , the following informations are included: 'distName': [{'var1': 1}, {'var2': 2}]} where for each var it is indicated the var dimension
    self.ND_sampling_params             = {}                       # this dictionary contains a dictionary for each ND distribution (key). This latter dictionary contains the initialization parameters of the ND inverseCDF ('initial_grid_disc' and 'tolerance')
    ######

    self.assemblerObjects  = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    self.requiredAssObject = (False,([],[]))          # tuple. first entry boolean flag. True if the XML parser must look for objects;
                                                      # second entry tuple.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))
    self.assemblerDict     = {}  # {'class':[['subtype','name',instance]]}

  def _localGenerateAssembler(self,initDict):
    ''' see generateAssembler method '''
    availableDist = initDict['Distributions']
    self._generateDistributions(availableDist)


  def _localWhatDoINeed(self):
    """
    This method is a local mirror of the general whatDoINeed method.
    It is implemented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [] # Every sampler requires Distributions
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

    Assembler._readMoreXML(self,xmlNode)

    for child in xmlNode:
      prefix = ""
      if child.tag == 'Distribution':
        for childChild in child:
          if childChild.tag =='distribution':
            prefix = "<distribution>"
            tobesampled = childChild.text
        self.toBeSampled[prefix+child.attrib['name']] = tobesampled
      elif child.tag == 'variable':
        for childChild in child:
          if childChild.tag =='distribution':
            tobesampled = childChild.text
            varData={}
            varData['name']=childChild.text
            if childChild.get('dim') == None:
              dim=1
            else:
              dim=childChild.attrib['dim']
            varData['dim']=int(dim)
            self.variables2distributionsMapping[child.attrib['name']] = varData
        self.toBeSampled[prefix+child.attrib['name']] = tobesampled
      elif child.tag == "sampler_init":
        self.initSeed = Distributions.randomIntegers(0,2**31)
        for childChild in child:
          if childChild.tag == "limit":
            self.limit = childChild.text
          elif childChild.tag == "initial_seed":
            self.initSeed = int(childChild.text)
          elif childChild.tag == "reseed_at_each_iteration":
            if childChild.text.lower() in utils.stringsThatMeanTrue(): self.reseedAtEachIteration = True
          elif childChild.tag == "dist_init":
            for childChildChild in childChild:
              NDdistData = {}
              for childChildChildChild in childChildChild:
                if childChildChildChild.tag == 'initial_grid_disc':
                  NDdistData[childChildChildChild.tag] = int(childChildChildChild.text)
                elif childChildChildChild.tag == 'tolerance':
                  NDdistData[childChildChildChild.tag] = float(childChildChildChild.text)
                else:
                  self.raiseAnError(IOError,self,'Unknown tag '+childChildChildChild.tag+' .Available are: initial_grid_disc and tolerance!')
              self.ND_sampling_params[childChildChild.attrib['name']] = NDdistData
          else: self.raiseAnError(IOError,self,'Unknown tag '+child.tag+' .Available are: limit, initial_seed, reseed_at_each_iteration and dist_init!')

    if self.initSeed == None:
      self.initSeed = Distributions.randomIntegers(0,2**31)

    # Creation of the self.distributions2variablesMapping dictionary: {'dist_name': ({'variable_name1': dim1}, {'variable_name2': dim2})}
    for variable in self.variables2distributionsMapping.keys():
      distName = self.variables2distributionsMapping[variable]['name']
      dim      = self.variables2distributionsMapping[variable]['dim']
      list_element={}
      list_element[variable] = dim
      if (distName in self.distributions2variablesMapping.keys()):
        self.distributions2variablesMapping[distName].append(list_element)
      else:
        self.distributions2variablesMapping[distName]=[list_element]


    for key in self.variables2distributionsMapping.keys():
      dist = self.variables2distributionsMapping[key]['name']
      maxDim=1
      list = self.distributions2variablesMapping[dist]
      for var in list:
        if var.values()[0] > maxDim:
          maxDim = var.values()[0]
      self.variables2distributionsMapping[key]['totDim'] = maxDim #len(self.distributions2variablesMapping[self.variables2distributionsMapping[key]['name']])


    self.localInputAndChecks(xmlNode)

  def endJobRunnable(self): return self._endJobRunnable

  def localInputAndChecks(self,xmlNode):
    '''place here the additional reading, remember to add initial parameters in the method localAddInitParams'''
    pass

  def addInitParams(self,tempDict):
    """
    This function is called from the base class to print some of the information inside the class.
    Whatever is permanent in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary. No information about values that change during the simulation are allowed
    @ In/Out tempDict: {'attribute name':value}
    """
    for variable in self.toBeSampled.items():
      tempDict[variable[0]] = 'is sampled using the distribution ' +variable[1]
    tempDict['limit' ]        = self.limit
    tempDict['initial seed' ] = self.initSeed
    self.localAddInitParams(tempDict)

  def localAddInitParams(self,tempDict):
    """use this function to export to the printer in the base class the additional PERMANENT your local class have"""

  def addCurrentSetting(self,tempDict):
    """
    This function is called from the base class to print some of the information inside the class.
    Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
    The information is passed back in the dictionary
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict
    """
    tempDict['counter'       ] = self.counter
    tempDict['initial seed'  ] = self.initSeed
    for key in self.inputInfo:
      if key!='SampledVars': tempDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys(): tempDict['Variable: '+var+' has value'] = tempDict[key][var]
    self.localAddCurrentSetting(tempDict)

  def localAddCurrentSetting(self,tempDict):
    """use this function to export to the printer in the base class the additional PERMANENT your local class have"""
    pass

  def _generateDistributions(self,availableDist):
    """
    here the needed distribution are made available to the step as also the initialization
    of the seeding (the siding could be overriden by the step by calling the initialize method
    @in availableDist: {'distribution name':instance}
    """
    if self.initSeed != None:
      Distributions.randomSeed(self.initSeed)
    for key in self.toBeSampled.keys():
      if self.toBeSampled[key] not in availableDist.keys(): self.raiseAnError(IOError,self,'Distribution '+self.toBeSampled[key]+' not found among available distributions (check input)!')
      self.distDict[key] = availableDist[self.toBeSampled[key]]
      self.inputInfo['crowDist'][key] = json.dumps(self.distDict[key].getCrowDistDict())

  def initialize(self,externalSeeding=None,solutionExport=None):
    """
    This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
    @in solutionExport: in goal oriented sampling (a.k.a. adaptive sampling this is where the space/point satisfying the constrains)
    """
    self.counter = 0
    if   not externalSeeding          :
      Distributions.randomSeed(self.initSeed)       #use the sampler initialization seed
      self.auxcnt = self.initSeed
    elif externalSeeding=='continue'  : pass        #in this case the random sequence needs to be preserved
    else                              :
      Distributions.randomSeed(externalSeeding)     #the external seeding is used
      self.auxcnt = externalSeeding
    #for key in self.toBeSampled.keys():
    #    self.distDict[key].initializeDistribution()   #now we can initialize the distributions
    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport != None : self.localInitialize(solutionExport=solutionExport)
    else                      : self.localInitialize()

    for distrib in self.ND_sampling_params:
      if distrib in self.distributions2variablesMapping:
        params = self.ND_sampling_params[distrib]
        temp = self.distributions2variablesMapping[distrib][0].keys()[0]
        self.distDict[temp].updateRNGParam(params)

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
    model.getAdditionalInputEdits(self.inputInfo)
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
class AdaptiveSampler(Sampler):
  '''This is a general adaptive sampler'''
  def __init__(self):
    Sampler.__init__(self)
#    self.assemblerObjects = {}               #this dictionary contains information about the object needed by the adaptive sampler in order to work (ROM,targetEvaluation, etc)
    self.goalFunction     = None             #this is the pointer to the function defining the goal
    self.tolerance        = None             #this is norm of the error threshold
    self.subGridTol       = None             #This is the tolerance used to construct the testing sub grid
    self.toleranceWeight  = 'cdf'            #this is the a flag that controls if the convergence is checked on the hyper-volume or the probability
    self.persistence      = 5                #this is the number of times the error needs to fell below the tollerance before considering the sim converged
    self.repetition       = 0                #the actual number of time the error was below the requested threshold
    self.forceIteration   = False            #this flag control if at least a self.limit number of iteration should be done
    self.axisName         = None             #this is the ordered list of the variable names (ordering match self.gridStepSize anfd the ordering in the test matrixes)
    self.gridVectors      = {}               # {'name of the variable':numpy.ndarray['the coordinate']}
    self.testGridLenght   = 0                #this the total number of point in the testing grid
    self.oldTestMatrix    = None             #This is the test matrix to use to store the old evaluation of the function
    self.solutionExport   = None             #This is the data used to export the solution (it could also not be present)
    self.nVar             = 0                #this is the number of the variable sampled
    self.surfPoint        = None             #coordinate of the points considered on the limit surface
    self.hangingPoints    = []               #list of the points already submitted for evaluation for which the result is not yet available
    # postprocessor to compute the limit surface
    self.limitSurfacePP   = PostProcessors.returnInstance("LimitSurface",self)
    self.printTag         = 'SAMPLER ADAPTIVE'
    self.requiredAssObject = (True,(['TargetEvaluation','ROM','Function'],['n','n','-n']))       # tuple. first entry boolean flag. True if the XML parser must look for assembler objects;

  def localInputAndChecks(self,xmlNode):
    if 'limit' in xmlNode.attrib.keys():
      try: self.limit = int(xmlNode.attrib['limit'])
      except ValueError: self.raiseAnError(IOError,self,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    # convergence Node
    convergenceNode = xmlNode.find('Convergence')
    if convergenceNode==None:self.raiseAnError(IOError,self,'the node Convergence was missed in the definition of the adaptive sampler '+self.name)
    try   : self.tolerance=float(convergenceNode.text)
    except: self.raiseAnError(IOError,self,'Failed to convert '+convergenceNode.text+' to a meaningful number for the convergence')
    attribList = list(convergenceNode.attrib.keys())
    if 'limit'          in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('limit'))
      try   : self.limit = int (convergenceNode.attrib['limit'])
      except: self.raiseAnError(IOError,self,'Failed to convert the limit value '+convergenceNode.attrib['limit']+' to a meaningful number for the convergence')
    if 'persistence'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('persistence'))
      try   : self.persistence = int (convergenceNode.attrib['persistence'])
      except: self.raiseAnError(IOError,self,'Failed to convert the persistence value '+convergenceNode.attrib['persistence']+' to a meaningful number for the convergence')
    if 'weight'         in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('weight'))
      try   : self.toleranceWeight = str(convergenceNode.attrib['weight']).lower()
      except: self.raiseAnError(IOError,self,'Failed to convert the weight type '+convergenceNode.attrib['weight']+' to a meaningful string for the convergence')
    if 'subGridTol'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('subGridTol'))
      try   : self.subGridTol = float (convergenceNode.attrib['subGridTol'])
      except: self.raiseAnError(IOError,self,'Failed to convert the subGridTol '+convergenceNode.attrib['subGridTol']+' to a meaningful float for the convergence')
    if 'forceIteration' in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('forceIteration'))
      if   convergenceNode.attrib['forceIteration']=='True' : self.forceIteration   = True
      elif convergenceNode.attrib['forceIteration']=='False': self.forceIteration   = False
      else: self.raiseAnError(RuntimeError,self,'Reading the convergence setting for the adaptive sampler '+self.name+' the forceIteration keyword had an unknown value: '+str(convergenceNode.attrib['forceIteration']))
    #assembler node: Hidden from User
    # set subgrid
    if self.subGridTol == None: self.subGridTol = self.tolerance
    if self.subGridTol > self.tolerance: self.raiseAnError(IOError,self,'The sub grid tolerance '+str(self.subGridTol)+' must be smaller than the tolerance: '+str(self.tolerance))
    if len(attribList)>0: self.raiseAnError(IOError,self,'There are unknown keywords in the convergence specifications: '+str(attribList))

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
    if 'Function' in self.assemblerDict.keys(): self.goalFunction = self.assemblerDict['Function'][0][3]
    if 'TargetEvaluation' in self.assemblerDict.keys(): self.lastOutput = self.assemblerDict['TargetEvaluation'][0][3]
    self.memoryStep        = 5               # number of step for which the memory is kept
    self.solutionExport    = solutionExport
    # check if solutionExport is actually a "Datas" type "TimePointSet"
    if type(solutionExport).__name__ != "TimePointSet": self.raiseAnError(IOError,self,'solutionExport type is not a TimePointSet. Got '+ type(solutionExport).__name__+'!')
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.oldTestMatrix     = None             #This is the test matrix to use to store the old evaluation of the function
    self.persistenceMatrix = None             #this is a matrix that for each point of the testing grid tracks the persistence of the limit surface position
    if self.goalFunction.name not in self.solutionExport.getParaKeys('output'): self.raiseAnError(IOError,self,'Goal function name does not match solution export data output.')
    # set number of job requestable after a new evaluation
    self._endJobRunnable   = 1
    #check if convergence is not on probability if all variables are bounded in value otherwise the problem is unbounded
    if self.toleranceWeight=='value':
      for varName in self.distDict.keys():
        if not(self.distDict[varName].upperBoundUsed and self.distDict[varName].lowerBoundUsed):
          self.raiseAnError(TypeError,self,'It is impossible to converge on an unbounded domain (variable '+varName+' with distribution '+self.distDict[varName].name+') as requested to the sampler '+self.name)
    elif self.toleranceWeight=='cdf': pass
    else: self.raiseAnError(IOError,self,'Unknown weight string descriptor: '+self.toleranceWeight)
    #setup the grid. The grid is build such as each element has a volume equal to the sub grid tolerance
    #the grid is build in such a way that an unit change in each node within the grid correspond to a change equal to the tolerance
    self.nVar        = len(self.distDict.keys())               #Total number of variables
    stepLenght        = self.subGridTol**(1./float(self.nVar)) #build the step size in 0-1 range such as the differential volume is equal to the tolerance
    self.axisName     = []                                     #this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    #here we build lambda function to return the coordinate of the grid point depending if the tolerance is on probability or on volume
    if self.toleranceWeight!='cdf': stepParam = lambda x: [stepLenght*(self.distDict[x].upperBound-self.distDict[x].lowerBound), self.distDict[x].lowerBound, self.distDict[x].upperBound]
    else                          : stepParam = lambda _: [stepLenght, 0.0, 1.0]
    #moving forward building all the information set
    pointByVar = [None]*self.nVar                              #list storing the number of point by cooridnate
    #building the grid point coordinates
    gridVectorsForLS = {}
    for varId, varName in enumerate(self.distDict.keys()):
      self.axisName.append(varName)
      [myStepLength, start, end]  = stepParam(varName)
      start                      += 0.5*myStepLength
      if self.toleranceWeight=='cdf'     : self.gridVectors[varName] = np.asarray([self.distDict[varName].ppf(pbCoord) for pbCoord in  np.arange(start,end,myStepLength)])
      elif self.toleranceWeight=='value' : self.gridVectors[varName] = np.arange(start,end,myStepLength)
      pointByVar[varId]           = np.shape(self.gridVectors[varName])[0]
      gridVectorsForLS[varName.replace('<distribution>','')] = self.gridVectors[varName]
    self.oldTestMatrix            = np.zeros(tuple(pointByVar))
    # initialize LimitSurface PP
    self.limitSurfacePP._initFromDict({"parameters":[key.replace('<distribution>','') for key in self.distDict.keys()],"tolerance":self.subGridTol,"side":"both","gridVectors":gridVectorsForLS)
    self.limitSurfacePP.assemblerDict = self.assemblerDict
    self.limitSurfacePP._initializeLSpp({'WorkingDir':None},[self.lastOutput],{})
    self.persistenceMatrix        = np.zeros(tuple(pointByVar))      #matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.testGridLenght           = np.prod (pointByVar)          #total number of point on the grid
    self.oldTestMatrix            = np.zeros(tuple(pointByVar))      #swap matrix fro convergence test
    self.hangingPoints            = np.ndarray((0, self.nVar))
    self.raiseADebug(self,'Initialization done')

  def localStillReady(self,ready): #,lastOutput=None
    '''
    first perform some check to understand what it needs to be done possibly perform an early return
    ready is returned
    lastOutput should be present when the next point should be chosen on previous iteration and convergence checked
    lastOutput it is not considered to be present during the test performed for generating an input batch
    ROM if passed in it is used to construct the test matrix otherwise the nearest neightburn value is used
    '''
    self.raiseADebug(self,'From method localStillReady...')
    #test on what to do
    if ready      == False : return ready #if we exceeded the limit just return that we are done
    if type(self.lastOutput) == dict:
      if self.lastOutput == None and self.limitSurfacePP.ROM.amITrained==False: return ready
    else:
      #if the last output is not provided I am still generating an input batch, if the rom was not trained before we need to start clean
      if self.lastOutput.isItEmpty() and self.limitSurfacePP.ROM.amITrained==False: return ready
    #first evaluate the goal function on the newly sampled points and store them in mapping description self.functionValue RecontructEnding
    if type(self.lastOutput) == dict:
      if self.lastOutput != None: self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    else:
      if not self.lastOutput.isItEmpty(): self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    self.raiseADebug(self,'Training finished')
    np.copyto(self.oldTestMatrix,self.limitSurfacePP.getTestMatrix())    #copy the old solution (contained in the limit surface PP) for convergence check
    # evaluate the Limit Surface coordinates (return input space coordinates, evaluation vector and grid indexing)
    self.surfPoint, evaluations, listsurfPoint = self.limitSurfacePP.run(returnListSurfCoord = True)

    self.raiseADebug(self,'Prediction finished')
    # check hanging points
    if self.goalFunction.name in self.limitSurfacePP.getFunctionValue().keys(): indexLast = len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])-1
    else                                                                      : indexLast = -1
    #index of last set of point tested and ready to perform the function evaluation
    indexEnd  = len(self.limitSurfacePP.getFunctionValue()[self.axisName[0].replace('<distribution>','')])-1
    tempDict  = {}
    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.limitSurfacePP.getFunctionValue().items(): tempDict[key] = value[myIndex]
      if len(self.hangingPoints) > 0: self.hangingPoints = self.hangingPoints[~(self.hangingPoints==np.array([tempDict[varName] for varName in [key.replace('<distribution>','') for key in self.axisName]])).all(axis=1)][:]
    self.persistenceMatrix += self.limitSurfacePP.getTestMatrix()
    # test error
    testError = np.sum(np.abs(np.subtract(self.limitSurfacePP.getTestMatrix(),self.oldTestMatrix))) # compute the error
    if (testError > self.tolerance/self.subGridTol): ready, self.repetition = True, 0                         # we still have error
    else              : self.repetition +=1                                                                   # we are increasing persistence
    if self.persistence<self.repetition: ready =  False                                                       # we are done
    self.raiseADebug(self,'counter: '+str(self.counter)+'       Error: ' +str(testError)+' Repetition: '+str(self.repetition))
    #if the number of point on the limit surface is > than compute persistence
    if len(listsurfPoint)>0:
      self.invPointPersistence = np.ndarray(len(listsurfPoint))
      for pointID, coordinate in enumerate(listsurfPoint):
        self.invPointPersistence[pointID]=abs(self.persistenceMatrix[tuple(coordinate)])
      maxPers = np.max(self.invPointPersistence)
      self.invPointPersistence = (maxPers-self.invPointPersistence)/maxPers
      if self.solutionExport!=None:
        for varName in self.solutionExport.getParaKeys('inputs'):
          for varIndex in range(len(self.axisName)):
            if varName == [key.replace('<distribution>','') for key in self.axisName][varIndex]:
              self.solutionExport.removeInputValue(varName)
              for value in self.surfPoint[:,varIndex]: self.solutionExport.updateInputValue(varName,copy.copy(value))
        # to be fixed
        self.solutionExport.removeOutputValue(self.goalFunction.name)
        for index in range(len(evaluations)): self.solutionExport.updateOutputValue(self.goalFunction.name,copy.copy(evaluations[index]))
    return ready

  def localGenerateInput(self,model,oldInput):
    #self.adaptAlgo.nextPoint(self.dataContainer,self.goalFunction,self.values,self.distDict)
    # create values dictionary
    '''compute the direction normal to the surface, compute the derivative normal to the surface of the probability,
     check the points where the derivative probability is the lowest'''

    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    self.raiseADebug(self,'generating input')
    varSet=False
    if self.surfPoint!=None and len(self.surfPoint)>0:
      sampledMatrix = np.zeros((len(self.limitSurfacePP.getFunctionValue()[self.axisName[0].replace('<distribution>','')])+len(self.hangingPoints[:,0]),len(self.axisName)))
      for varIndex, name in enumerate([key.replace('<distribution>','') for key in self.axisName]): sampledMatrix [:,varIndex] = np.append(self.limitSurfacePP.getFunctionValue()[name],self.hangingPoints[:,varIndex])
      distanceTree = spatial.cKDTree(copy.copy(sampledMatrix),leafsize=12)
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
      else: self.raiseADebug(self,'np.max(distance)=0.0')

    if not varSet:
      #here we are still generating the batch
      for key in self.distDict.keys():
        if self.toleranceWeight=='cdf':
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
    self.raiseADebug(self,'At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
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
#      #getting the coordinate ready to be evaluated by the ROM
#      pbMapPointCoord.shape = (len(self.surfPoint)*(self.nVar*2+1),self.nVar)
#      tempDict = {}
#      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
#        tempDict[varName] = pbMapPointCoord.T[varIndex,:]
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
#
#      minIndex = np.argmin(np.abs(modGrad))
#      pdDist = self.sign*(pbPoint[minIndex,2*self.nVar][0]-0.5-10*self.tolerance)/modGrad[minIndex]
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
    self.printTag = 'SAMPLER MONTECARLO'

  def localInputAndChecks(self,xmlNode):
    if xmlNode.find('sampler_init')!= None:
      if xmlNode.find('sampler_init').find('limit')!= None:
        try: self.limit = int(xmlNode.find('sampler_init').find('limit').text)
        except ValueError:
          self.raiseAnError(IOError,self,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
      else:
        self.raiseAnError(IOError,self,'Monte Carlo sampling needs the limit block (number of samples) in the sampler_init block')
    else:
      self.raiseAnError(IOError,self,'Monte Carlo sampling needs the sampler_init block')


  def localGenerateInput(self,model,myInput):
    '''set up self.inputInfo before being sent to the model'''
    # create values dictionary

    for key in self.distDict:
      # check if the key is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
      #if key in self.ND_sampling_params.keys():

      dim    = self.variables2distributionsMapping[key]['dim']
      totDim = self.variables2distributionsMapping[key]['totDim']
      dist   = self.variables2distributionsMapping[key]['name']

      if dim == 1:
        rvsnum = self.distDict[key].rvs()
        for var in self.distributions2variablesMapping[dist]:
          varID  = var.keys()[0]
          varDim = var[varID]
          for kkey in varID.strip().split(','):
            self.values[kkey] = np.atleast_1d(rvsnum)[varDim-1]
            if totDim > 1 and dim == 1:
              coordinate=[];
              for i in range(totDim):
                coordinate.append(np.atleast_1d(rvsnum)[i])
              self.inputInfo['SampledVarsPb'][kkey] = self.distDict[key].pdf(coordinate)
            elif totDim == 1:
              self.inputInfo['SampledVarsPb'][kkey] = self.distDict[key].pdf(self.values[kkey])
            else:
              self.inputInfo['SampledVarsPb'][kkey] = 1.0

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
    self.printTag = 'SAMPLER GRID'
    self.gridCoordinate       = []    # the grid point to be used for each distribution (changes at each step)
    self.axisName             = []    # the name of each axis (variable)
    self.gridInfo             = {}    # {'name of the variable':('Type','Construction',[values])}  --> Type: Probability/Value; Construction:Custom/Equal
    self.externalgGridCoord   = False # boolean attribute. True if the coordinate list has been filled by external source (see factorial sampler)

    #gridInfo[var][0] is type, ...[1] is construction, ...[2] is values


  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    if 'limit' in xmlNode.attrib.keys(): self.raiseAnError(IOError,self,'limit is not used in Grid sampler')
    self.limit = 1
    if not self.axisName: self.axisName = []

    for child in xmlNode:
      if child.tag == "Distribution":
        #Add <distribution> to name so we know it is not a direct variable
        varName = "<distribution>"+child.attrib['name']
      elif child.tag == "variable":
        varName = child.attrib['name']
      for childChild in child:
        if (childChild.tag =='grid' and child.tag == "variable") or (childChild.tag =='grid' and child.tag == "Distribution"):
          self.axisName.append(varName)
          if childChild.attrib['type'] == 'global_grid':
            self.gridInfo[varName] = ('CDF','global_grid',childChild.text)
          else:
            constrType = childChild.attrib['construction']
            if constrType == 'custom':
              tempList = [float(i) for i in childChild.text.split()]
              tempList.sort()
              self.gridInfo[varName] = (childChild.attrib['type'],constrType,tempList)
              if self.gridInfo[varName][0]!='value' and self.gridInfo[varName][0]!='CDF': self.raiseAnError(IOError,self,'The type of grid is neither value nor CDF')
              self.limit = len(tempList)*self.limit
            elif constrType == 'equal':
              self.limit = self.limit*(int(childChild.attrib['steps'])+1)
              if   'lowerBound' in childChild.attrib.keys():
                self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['lowerBound']) + float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
                self.gridInfo[varName][2].sort()
              elif 'upperBound' in childChild.attrib.keys():
                self.gridInfo[varName] = (childChild.attrib['type'], constrType, [float(childChild.attrib['upperBound']) - float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
                self.gridInfo[varName][2].sort()
              else: self.raiseAnError(IOError,self,'no upper or lower bound has been declared for '+str(child.tag)+' in sampler '+str(self.name))
            else: self.raiseAnError(IOError,self,'not specified the grid construction type')

    if len(self.toBeSampled.keys()) != len(self.gridInfo.keys()):
      self.raiseAnError(IOError,self,'inconsistency between number of variables and grid specification')
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
            self.raiseAnError(TypeError,self,'the variable '+varName+'can not be sampled at '+str(valueMax)+' since outside the upper bound of the chosen distribution,Distripution Upper Bound = '+ str(self.distDict[varName].upperBound))
          if valueMax>self.distDict[varName].upperBound and valueMax-2.0*np.finfo(valueMax).eps<=self.distDict[varName].upperBound:
            valueMax = valueMax-2.0*np.finfo(valueMax).eps
        if self.distDict[varName].lowerBoundUsed:
          if valueMin<self.distDict[varName].lowerBound and valueMin+2.0*np.finfo(valueMin).eps<self.distDict[varName].lowerBound:
            self.raiseAnError(TypeError,self,'the variable '+varName+'can not be sampled at '+str(valueMin)+' since outside the lower bound of the chosen distribution,Distripution Lower Bound = '+str(self.distDict[varName].lowerBound))
          if valueMin<self.distDict[varName].lowerBound and valueMin+2.0*np.finfo(valueMax).eps>=self.distDict[varName].lowerBound:
            valueMin = valueMin-2.0*np.finfo(valueMin).eps
        self.gridInfo[varName][2][indexMax], self.gridInfo[varName][2][indexMin] = valueMax, valueMin

  def localGenerateInput(self,model,myInput):
    remainder = self.counter - 1 #used to keep track as we get to smaller strides
    stride = self.limit+1 #How far apart in the 1D array is the current gridCoordinate
    #self.inputInfo['distributionInfo'] = {}
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used

#     weight = 1.0
#     for i in range(len(self.gridCoordinate)):
#       varName = self.axisName[i]
#       if not self.externalgGridCoord:
#         stride = stride // len(self.gridInfo[varName][2])
#         #index is the index into the array self.gridInfo[varName][2]
#         index, remainder = divmod(remainder, stride )
#         self.gridCoordinate[i] = index
#
#       # check if the varName is a comma separated list of strings
#       # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
#       for kkey in varName.strip().split(','):
#         self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
#         self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
#         if self.gridInfo[varName][0]=='CDF':
#           self.values[kkey] = self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]])
#           self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
#         elif self.gridInfo[varName][0]=='value':
#           self.values[kkey] = self.gridInfo[varName][2][self.gridCoordinate[i]]
#           self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
#         else: raisea IOError (self.gridInfo[varName][0]+' is not know as value keyword for type. Sampler: '+self.name)
#
#       if self.gridInfo[varName][0]=='CDF':
#         if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
#         if self.gridCoordinate[i] == 0: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(0))/2.0)
#         if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(1))/2.0) - self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
#       else:
#         if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
#         if self.gridCoordinate[i] == 0: weight *= self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].lowerBound)/2.0)
#         if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1: weight *= self.distDict[varName].cdf((self.values[kkey]+self.distDict[varName].upperBound)/2.0) -self.distDict[varName].cdf((self.values[kkey]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
#     self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
#     self.inputInfo['ProbabilityWeight'] = weight
#     self.inputInfo['SamplerType'] = 'Grid'

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
      for key in varName.strip().split(','):
        self.inputInfo['distributionName'][key] = self.toBeSampled[varName]
        self.inputInfo['distributionType'][key] = self.distDict[varName].type

        if self.gridInfo[varName][0]=='CDF':
          if self.distDict[varName].getDimensionality()==1:
            self.values[key] = self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]])
          else:
            location = self.variables2distributionsMapping[varName]['dim']
            self.values[key] = self.distDict[varName].inverseMarginalDistribution(self.gridInfo[varName][2][self.gridCoordinate[i]],location-1)

        elif self.gridInfo[varName][0]=='value':
          self.values[key] = self.gridInfo[varName][2][self.gridCoordinate[i]]

        else: self.raiseAnError(IOError,self,gridInfo[varName][0]+' is not know as value keyword for type. Sampler: '+self.name)


    remainder = self.counter - 1 #used to keep track as we get to smaller strides
    stride = self.limit+1 #How far apart in the 1D array is the current gridCoordinate

    for i in range(len(self.gridCoordinate)):
      varName = self.axisName[i]
      if not self.externalgGridCoord:
        stride = stride // len(self.gridInfo[varName][2])
        index, remainder = divmod(remainder, stride )
        self.gridCoordinate[i] = index

      for key in varName.strip().split(','):
        self.inputInfo['distributionName'][key] = self.toBeSampled[varName]
        self.inputInfo['distributionType'][key] = self.distDict[varName].type

        if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim']==1):
          self.inputInfo['SampledVarsPb'][key] = self.distDict[varName].pdf(self.values[key])
        else:
          dist_name = self.variables2distributionsMapping[varName]['name']
          #NDcoordinate=np.zeros(len(self.distributions2variablesMapping[dist_name]))
          NDcoordinate=[]
          for i in range(len(self.distributions2variablesMapping[dist_name])):
            NDcoordinate.append(0)
          for var in self.distributions2variablesMapping[dist_name]:
            variable = var.keys()[0]
            position = var.values()[0]
            NDcoordinate[position-1] = self.values[variable.strip().split(',')[0]]
          self.inputInfo['SampledVarsPb'][key] = self.distDict[varName].pdf(NDcoordinate)

      # 1D variable
      if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim']==1):
        if self.gridInfo[varName][0]=='CDF':
          if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1:
            weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
          if self.gridCoordinate[i] == 0:
            weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]+1]))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(0))/2.0)
          if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1:
            weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(1))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(self.gridInfo[varName][2][self.gridCoordinate[i]-1]))/2.0)
        else:   # Value
          if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1:
            weight *= self.distDict[varName].cdf((self.values[key]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[key]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
          if self.gridCoordinate[i] == 0:
            weight *= self.distDict[varName].cdf((self.values[key]+self.gridInfo[varName][2][self.gridCoordinate[i]+1])/2.0) -self.distDict[varName].cdf((self.values[key]+self.distDict[varName].lowerBound)/2.0)
          if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1:
            weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].upperBound)/2.0) -self.distDict[varName].cdf((self.values[key]+self.gridInfo[varName][2][self.gridCoordinate[i]-1])/2.0)
      # ND variable
      else:
        if self.variables2distributionsMapping[varName]['dim']==1:    # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
          dist_name = self.variables2distributionsMapping[varName]['name']
          NDcoordinate=np.zeros(len(self.distributions2variablesMapping[dist_name]))
          dxs=np.zeros(len(self.distributions2variablesMapping[dist_name]))
          for var in self.distributions2variablesMapping[dist_name]:
            variable = var.keys()[0]
            position = var.values()[0]
            NDcoordinate[position-1] = self.values[variable.strip().split(',')[0]]
            if self.gridCoordinate[i] != 0 and self.gridCoordinate[i] < len(self.gridInfo[varName][2])-1:
              dxs[position-1] = (self.gridInfo[variable][2][self.gridCoordinate[i]+1] - self.gridInfo[variable][2][self.gridCoordinate[i]-1]) / 2.0
            if self.gridCoordinate[i] == 0:
              dxs[position-1] = self.gridInfo[variable][2][self.gridCoordinate[i]+1] - self.gridInfo[variable][2][self.gridCoordinate[i]]
            if self.gridCoordinate[i] == len(self.gridInfo[varName][2])-1:
              dxs[position-1] = self.gridInfo[variable][2][self.gridCoordinate[i]] - self.gridInfo[variable][2][self.gridCoordinate[i]-1]
          weight *= self.distDict[varName].cellIntegral(NDcoordinate,dxs)

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
    self.printTag = 'SAMPLER LHS'
    self.globalGrid          = {}    # Dictionary for the global_grid. These grids are used only for LHS for ND distributions.

  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)

    for child in xmlNode:
      if child.tag == "Distribution":
        #Add <distribution> to name so we know it is not a direct variable
        varName = "<distribution>"+child.attrib['name']

      elif child.tag == "global_grid":
         for childChild in child:
           if childChild.tag =='grid':
             globalGridName = childChild.attrib['name']
             constrType = childChild.attrib['construction']
             if constrType == 'custom':
              tempList = [float(i) for i in childChild.text.split()]
              tempList.sort()
              self.globalGrid[globalGridName] = (tempList)
              self.limit = len(tempList)*self.limit
             elif constrType == 'equal':
              self.limit = self.limit*(int(childChild.attrib['steps'])+1)
              if   'lowerBound' in childChild.attrib.keys():
                self.globalGrid[globalGridName] = ([float(childChild.attrib['lowerBound']) + float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
                self.globalGrid[globalGridName].sort()
              elif 'upperBound' in childChild.attrib.keys():
                self.globalGrid[globalGridName] = ([float(childChild.attrib['upperBound']) - float(childChild.text)*i for i in range(int(childChild.attrib['steps'])+1)])
                self.globalGrid[globalGridName].sort()
              else: self.raiseAnError(IOError,self,'no upper or lower bound has been declared for '+str(child.tag)+' in sampler '+str(self.name))
           else:
             self.raiseAnError(IOError,self,'The Tag ' + str(childChild.tag) + 'is not allowed in global_grid')

    for variable in self.gridInfo.keys():
      if self.gridInfo[variable][1] == 'global_grid':
        lst=list(self.gridInfo[variable])
        lst[2] = self.globalGrid[self.gridInfo[variable][2]]
        self.gridInfo[variable] = tuple(lst)

    pointByVar  = [len(self.gridInfo[variable][2]) for variable in self.gridInfo.keys()]
    if len(set(pointByVar))!=1: self.raiseAnError(IOError,self,'the latin Hyper Cube requires the same number of point in each dimension')
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
    '''
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
          self.values[kkey] = ppfvalue
          self.inputInfo['upper'][kkey] = ppfupper
          self.inputInfo['lower'][kkey] = ppflower
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
    self.inputInfo['ProbabilityWeight' ] = weight
    self.inputInfo['SamplerType'] = 'Stratified'
    '''

    j=0
    #self.inputInfo['distributionInfo'] = {}
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    weight = 1.0

    for varName in self.axisName:

      if not "<distribution>" in varName:
        if self.variables2distributionsMapping[varName]['totDim']>1 and self.variables2distributionsMapping[varName]['dim'] == 1:    # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
          upper = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]+1]
          lower = self.gridInfo[varName][2][self.sampledCoordinate[self.counter-2][j]  ]
          j += 1
          intervalFraction = Distributions.random()
          coordinate = lower + (upper-lower)*intervalFraction
          gridCoordinate =  self.distDict[varName].ppf(coordinate)
          distName = self.variables2distributionsMapping[varName]['name']
          for distVarName in self.distributions2variablesMapping[distName]:
            for kkey in distVarName.keys()[0].strip().split(','):
              self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
              self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
              self.values[kkey] = np.atleast_1d(gridCoordinate)[distVarName.values()[0]-1]
              #self.inputInfo['upper'][kkey] = ppfupper
              #self.inputInfo['lower'][kkey] = ppflower
              self.inputInfo['SampledVarsPb'][varName] = coordinate

          weight *= upper - lower

      if ("<distribution>" in varName) or self.variables2distributionsMapping[varName]['totDim']==1:   # 1D variable
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
            self.values[kkey] = ppfvalue
            self.inputInfo['upper'][kkey] = ppfupper
            self.inputInfo['lower'][kkey] = ppflower
            self.inputInfo['SampledVarsPb'][varName] = coordinate
            weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
          elif self.gridInfo[varName][0]=='value':
            self.values[varName] = coordinate
            self.inputInfo['upper'][kkey] = max(upper,lower)
            self.inputInfo['lower'][kkey] = min(upper,lower)
            self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
        if self.gridInfo[varName][0] =='CDF':
          weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
        else:
          weight *= self.distDict[varName].cdf(upper) - self.distDict[varName].cdf(lower)

    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = weight
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
    self.printTag = 'SAMPLER DYNAMIC ET'

  def _localWhatDoINeed(self):
    needDict = Sampler._localWhatDoINeed(self)
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
    parentNode.add('actual_end_ts',self.actual_end_ts)
#     # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
#     if(jobObject.identifier == self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().name): endInfo['parent_node'] = self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode()
#     else: endInfo['parent_node'] = list(self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().iter(jobObject.identifier))[0]
    endInfo['parent_node'] = parentNode
    # get the branchedLevel dictionary
    branchedLevel = {}
    for distk, distpb in zip(endInfo['parent_node'].get('initiator_distribution'),endInfo['parent_node'].get('PbThreshold')): branchedLevel[distk] = utils.index(self.branchProbabilities[distk],distpb)
    if not branchedLevel: self.raiseAnError(RuntimeError,self,'branchedLevel of node '+jobObject.identifier+'not found!')
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
      self.raiseADebug(self,'branch info file ' + os.path.basename(filename) +' has not been found. => No Branching.')
      return branch_present
    # Parse the file and create the xml element tree object
    #try:
    branch_info_tree = ET.parse(filename)
    self.raiseADebug(self,'Done parsing '+filename)
    #except? raisea IOError ('not able to parse ' + filename)
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
    self.inputInfo['prefix'                    ] = rname.encode()
    self.inputInfo['initiator_distribution'    ] = []
    self.inputInfo['PbThreshold'               ] = []
    self.inputInfo['ValueThreshold'            ] = []
    self.inputInfo['branch_changed_param'      ] = [b'None']
    self.inputInfo['branch_changed_param_value'] = [b'None']
    self.inputInfo['start_time'                ] = -sys.float_info.max
    self.inputInfo['end_ts'                    ] = 0
    self.inputInfo['parent_id'                 ] = 'root'
    self.inputInfo['conditional_prb'           ] = [1.0]
    self.inputInfo['conditional_pb'            ] = 1.0
    for key in self.branchProbabilities.keys():self.inputInfo['initiator_distribution'].append(key.encode())
    for key in self.branchProbabilities.keys():self.inputInfo['PbThreshold'].append(self.branchProbabilities[key][branchedLevel[key]])
    for key in self.branchProbabilities.keys():self.inputInfo['ValueThreshold'].append(self.branchValues[key][branchedLevel[key]])
    for varname in self.toBeSampled.keys():
      self.inputInfo['SampledVars'  ][varname] = self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
      self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]

    if(self.maxSimulTime): self.inputInfo['end_time'] = self.maxSimulTime
    # Call the model function "createNewInput" with the "values" dictionary just filled.
    # Add the new input path into the RunQueue system
    newInputs = model.createNewInput(myInput,self.type,**self.inputInfo)
    for key,value in self.inputInfo.items(): rootnode.add(key,value)
    self.RunQueue['queue'].append(newInputs)
    self.RunQueue['identifiers'].append(self.inputInfo['prefix'].encode())
    self.rootToJob[self.inputInfo['prefix']] = rname
    del newInputs
    self.counter += 1

  def _createRunningQueueBegin(self,model,myInput):
    # We construct the input for the first DET branch calculation'
    # Increase the counter
    # The root name of the xml element tree is the starting name for all the branches
    # (this root name = the user defined sampler name)
    # Get the initial branchedLevel dictionary (=> the list gets empty)
    branchedLevel = self.branchedLevel.pop(0)
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
      self.raiseADebug(self,'Branch ' + endInfo['parent_node'].get('name') + ' hit last Threshold for distribution ' + endInfo['branch_dist'])
      self.raiseADebug(self,'Branch ' + endInfo['parent_node'].get('name') + ' is dead end.')
      self.branchCountOnLevel = 1
      n_branches = endInfo['n_branches'] - 1

    # Loop over the branches for which the inputs must be created
    for _ in range(n_branches):
      del self.inputInfo
      self.counter += 1
      self.branchCountOnLevel += 1
      branchedLevel = branchedLevelParent
      # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
      rname = endInfo['parent_node'].get('name') + '-' + str(self.branchCountOnLevel)

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
          subGroup.add('branch_changed_param_value',endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2])
          subGroup.add('branch_changed_param_pb',endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2])
          cond_pb_c = cond_pb_c + endInfo['branch_changed_params'][key]['changed_cond_pb'][self.branchCountOnLevel-2]
        else:
          subGroup.add('branch_changed_param_value',endInfo['branch_changed_params'][key]['old_value'])
          subGroup.add('branch_changed_param_pb',endInfo['branch_changed_params'][key]['unchanged_pb'])
          cond_pb_un =  cond_pb_un + endInfo['branch_changed_params'][key]['unchanged_cond_pb']
      # add conditional probability
      if self.branchCountOnLevel != 1: subGroup.add('conditional_pb',cond_pb_c)
      else: subGroup.add('conditional_pb',cond_pb_un)
      # add initiator distribution info, start time, etc.
      subGroup.add('initiator_distribution',endInfo['branch_dist'])
      subGroup.add('start_time', endInfo['parent_node'].get('end_time'))
      # initialize the end_time to be equal to the start one... It will modified at the end of this branch
      subGroup.add('end_time', endInfo['parent_node'].get('end_time'))
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
      if str(endInfo['end_ts']) == 'None':
        pass
      self.inputInfo = {'prefix':rname.encode(),'end_ts':endInfo['end_ts'],
                'branch_changed_param':[subGroup.get('branch_changed_param')],
                'branch_changed_param_value':[subGroup.get('branch_changed_param_value')],
                'conditional_prb':[subGroup.get('conditional_pb')],
                'start_time':endInfo['parent_node'].get('end_time'),
                'parent_id':subGroup.get('parent')}
      # add the newer branch name to the map
      self.rootToJob[rname] = self.rootToJob[subGroup.get('parent')]
      # check if it is a preconditioned DET sampling, if so add the relative information
      precSampled = endInfo['parent_node'].get('preconditionerSampled')
      if precSampled:
        self.inputInfo['preconditionerCoordinate'] = copy.deepcopy(precSampled)
        subGroup.add('preconditionerSampled', precSampled)
      # Check if the distribution that just triggered hitted the last probability threshold .
      #  In this case there is not a probability threshold that needs to be added in the input
      #  for this particular distribution
      if not (branchedLevel[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']])):
        self.inputInfo['initiator_distribution'] = [endInfo['branch_dist']]
        self.inputInfo['PbThreshold'           ] = [self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]]
        self.inputInfo['ValueThreshold'        ] = [self.branchValues[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]]
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
        if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])): self.inputInfo['initiator_distribution'].append(key.encode())
      for key in self.branchProbabilities.keys():
        if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])):
          self.inputInfo['PbThreshold'   ].append(self.branchProbabilities[key][branchedLevel[key]])
          self.inputInfo['ValueThreshold'].append(self.branchValues[key][branchedLevel[key]])
      self.inputInfo['SampledVars']   = {}
      self.inputInfo['SampledVarsPb'] = {}
      for varname in self.toBeSampled.keys():
        self.inputInfo['SampledVars'][varname]   = self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
        self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
      self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())*subGroup.get('conditional_pb')
      self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
      # Call the model function  "createNewInput" with the "values" dictionary just filled.
      # Add the new input path into the RunQueue system
      self.RunQueue['queue'].append(model.createNewInput(myInput,self.type,**self.inputInfo))
      self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
      for key,value in self.inputInfo.items(): subGroup.add(key,value)
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
      self.raiseADebug(self,'A Branch ended!')
    return newerinput

  def _generateDistributions(self,availableDist):
    Grid._generateDistributions(self,availableDist)
    for preconditioner in self.preconditionerToApply.values(): preconditioner._generateDistributions(availableDist)

  def localInputAndChecks(self,xmlNode):
    Grid.localInputAndChecks(self,xmlNode)
    self.limit = sys.maxsize
    if 'print_end_xml' in xmlNode.attrib.keys():
      if xmlNode.attrib['print_end_xml'].lower() in utils.stringsThatMeanTrue(): self.print_end_xml = True
      else: self.print_end_xml = False
    if 'maxSimulationTime' in xmlNode.attrib.keys():
      try:    self.maxSimulTime = float(xmlNode.attrib['maxSimulationTime'])
      except (KeyError,NameError): self.raiseAnError(IOError,self,'Can not convert maxSimulationTime in float number!!!')
    for child in xmlNode:
      if child.tag == 'PreconditionerSampler':
        if not 'type' in child.attrib.keys()                          : self.raiseAnError(IOError,self,'Not found attribute type in PreconditionerSampler block!')
        if child.attrib['type'] in self.preconditionerToApply.keys()  : self.raiseAnError(IOError,self,'PreconditionerSampler type '+child.attrib['type'] + ' already inputted!')
        if child.attrib['type'] not in self.preconditionerAvail.keys(): self.raiseAnError(IOError,self,'PreconditionerSampler type' +child.attrib['type'] + 'unknown. Available are '+ str(self.preconditionerAvail.keys()).replace("[","").replace("]",""))
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
          self.raiseAWarning(self,"One of the Thresholds for distribution " + str(self.gridInfo[keyk][2]) + " is > 1")
          error_found = True
          for index in range(len(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float))):
            if sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float).count(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index]) > 1:
              self.raiseAWarning(self,"In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
              error_found = True
      else:
        self.branchValues[self.toBeSampled[keyk]] = self.gridInfo[keyk][2]
        self.branchValues[self.toBeSampled[keyk]].sort(key=float)
        for index in range(len(sorted(self.branchValues[self.toBeSampled[keyk]], key=float))):
          if sorted(self.branchValues[self.toBeSampled[keyk]], key=float).count(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index]) > 1:
            self.raiseAWarning(self,"In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
            error_found = True
    if error_found: self.raiseAnError(IOError,self,"In sampler named " + self.name+' Errors have been found!' )
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
        preconditioner.inputInfo['prefix'] = preconditioner.counter
        precondlistoflist[cnt].append(copy.deepcopy(preconditioner.inputInfo))
    if self.precNumberSamplers > 0:
      self.raiseADebug(self,'Number of Preconditioner Samples are ' + str(self.precNumberSamplers) + '!')
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
      self.TreeInfo[self.name + '_' + str(precSample+1)] = ETS.NodeTree(elm)

    for key in self.branchProbabilities.keys():
      #kk = self.toBeSampled.values().index(key)
      self.branchValues[key] = [self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].ppf(float(self.branchProbabilities[key][index])) for index in range(len(self.branchProbabilities[key]))]
    for key in self.branchValues.keys():
      #kk = self.toBeSampled.values().index(key)
      self.branchProbabilities[key] = [self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].cdf(float(self.branchValues[key][index])) for index in range(len(self.branchValues[key]))]
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
    self.printTag = 'SAMPLER ADAPTIVE DET'
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

  def _localWhatDoINeed(self):
    '''
    This method is a local mirrow of the general whatDoINeed method.
    It is implmented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    '''
    adaptNeed = AdaptiveSampler._localWhatDoINeed(self)
    DETNeed   = DynamicEventTree._localWhatDoINeed(self)
    return dict(adaptNeed.items()+ DETNeed.items())

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
      lowerCdfValues[key] = utils.find_le(self.branchProbabilities[self.toBeSampled[key]],cdfValues[key])[0]
      self.raiseADebug(self,str(self.toBeSampled[key]))
      self.raiseADebug(self,str(value))
      self.raiseADebug(self,str(cdfValues[key]))
      self.raiseADebug(self,str(lowerCdfValues[key]))
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
      neigh = neighbors.NearestNeighbors(n_neighbors=len(mapping.keys()))
      neigh.fit(nntrain)
      return self._checkValidityOfBranch(neigh.kneighbors(lowerCdfValues.values()),mapping),cdfValues
    else: return None,cdfValues

  def _checkValidityOfBranch(self,branchSet,mapping):
    """
    Function that checks if the nearest branches found by method _checkClosestBranch are valid
    @ In, tuple of branches
    @ In, dictionary of candidated branches
    @ Out, most valid branch (if noone found, return None)
    """
    validBranch   = None
    idOfBranches  = branchSet[1][-1]
    for closestBranch in idOfBranches:
      if not mapping[closestBranch+1].get('completedHistory'):
        validBranch = mapping[closestBranch+1]
        break
    return validBranch


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
    rname = info['parent_node'].get('name') + '-' + str(self.branchCountOnLevel)
    info['parent_node'].add('completedHistory', False)
    self.raiseADebug(self,str(rname))
    bcnt = self.branchCountOnLevel
    while info['parent_node'].isAnActualBranch(rname):
      bcnt += 1
      rname = info['parent_node'].get('name') + '-' + str(bcnt)
    # create a subgroup that will be appended to the parent element in the xml tree structure
    subGroup = ETS.Node(rname)
    subGroup.add('parent', info['parent_node'].get('name'))
    subGroup.add('name', rname)
    self.raiseADebug(self,'cond pb = '+str(info['parent_node'].get('conditional_pb')))
    cond_pb_c  = float(info['parent_node'].get('conditional_pb'))

    # Loop over  branch_changed_params (events) and start storing information,
    # such as conditional pb, variable values, into the xml tree object
    if endInfo:
      for key in endInfo['branch_changed_params'].keys():
        subGroup.add('branch_changed_param',key)
        subGroup.add('branch_changed_param_value',endInfo['branch_changed_params'][key]['old_value'][0])
        subGroup.add('branch_changed_param_pb',endInfo['branch_changed_params'][key]['associated_pb'][0])
    else:
      pass
    #cond_pb_c = cond_pb_c + copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_cond_pb'])
    # add conditional probability
    subGroup.add('conditional_pb',cond_pb_c)
    # add initiator distribution info, start time, etc.
    #subGroup.add('initiator_distribution',copy.deepcopy(endInfo['branch_dist']))
    subGroup.add('start_time', info['parent_node'].get('end_time'))
    # initialize the end_time to be equal to the start one... It will modified at the end of this branch
    subGroup.add('end_time', info['parent_node'].get('end_time'))
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
    self.inputInfo = {'prefix':rname,'end_ts':info['parent_node'].get('actual_end_ts'),
              'branch_changed_param':[subGroup.get('branch_changed_param')],
              'branch_changed_param_value':[subGroup.get('branch_changed_param_value')],
              'conditional_prb':[subGroup.get('conditional_pb')],
              'start_time':info['parent_node'].get('end_time'),
              'parent_id':subGroup.get('parent')}
    # add the newer branch name to the map
    self.rootToJob[rname] = self.rootToJob[subGroup.get('parent')]
#     check if it is a preconditioned DET sampling, if so add the relative information
#     precSampled = endInfo['parent_node'].get('preconditionerSampled')
#     if precSampled:
#       self.inputInfo['preconditionerCoordinate'] = copy.deepcopy(precSampled)
#       subGroup.add('preconditionerSampled', precSampled)
    # The probability Thresholds are stored here in the cdfValues dictionary... We are sure that they are whitin the ones defined in the grid
    # check is not needed
    self.inputInfo['initiator_distribution'] = [self.toBeSampled[key] for key in cdfValues.keys()]
    self.inputInfo['PbThreshold'           ] = cdfValues.values()
    self.inputInfo['ValueThreshold'        ] = [self.distDict[key].ppf(value) for key,value in cdfValues.items()]
    self.inputInfo['SampledVars'           ] = {}
    self.inputInfo['SampledVarsPb'         ] = {}
    for varname in self.toBeSampled.keys():
      self.inputInfo['SampledVars'][varname]   = self.distDict[varname].ppf(cdfValues[varname])
      self.inputInfo['SampledVarsPb'][varname] = cdfValues[varname]
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())*subGroup.get('conditional_pb')
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
    # Call the model function "createNewInput" with the "values" dictionary just filled.
    # Add the new input path into the RunQueue system
    self.RunQueue['queue'].append(model.createNewInput(myInput,self.type,**self.inputInfo))
    self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
    for key,value in self.inputInfo.items(): subGroup.add(key,value)
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
              if key not in lastOutDict['inputs'].keys(): lastOutDict['inputs'][key] = np.atleast_1d(histdict['inputs'][key])
              else                                      : lastOutDict['inputs'][key] = np.concatenate((np.atleast_1d(lastOutDict['inputs'][key]),np.atleast_1d(histdict['inputs'][key])))
            for key in histdict['outputs'].keys():
              if key not in lastOutDict['outputs'].keys(): lastOutDict['outputs'][key] = np.atleast_1d(histdict['outputs'][key])
              else                                       : lastOutDict['outputs'][key] = np.concatenate((np.atleast_1d(lastOutDict['outputs'][key]),np.atleast_1d(histdict['outputs'][key])))
        else: self.raiseAWarning(self,'No Completed histories! No possible to start an adaptive search! Something went wrongly!')
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
      if closestBranch == None: self.raiseADebug(self,'An usable branch for next candidate has not been found => create a parallel branch!')
      # add pbthresholds in the grid
      investigatedPoint = {}
      for key,value in cdfValues.items():
#         if self.insertAdaptBPb:
#           ind = utils.find_le_index(self.branchProbabilities[self.toBeSampled[key]],value)
#           if not ind: ind = 0
#           if value not in self.branchProbabilities[self.toBeSampled[key]]:
#             self.branchProbabilities[self.toBeSampled[key]].insert(ind,value)
#             self.branchValues[self.toBeSampled[key]].insert(ind,self.distDict[key].ppf(value))

        ind = utils.find_le_index(self.branchProbabilities[self.toBeSampled[key]],value)
        if not ind: ind = 0
        if value not in self.branchProbabilities[self.toBeSampled[key]]:
          self.branchProbabilities[self.toBeSampled[key]].insert(ind,value)
          self.branchValues[self.toBeSampled[key]].insert(ind,self.distDict[key].ppf(value))
        investigatedPoint[self.toBeSampled[key]] = value
      # collect investigated point
      self.investigatedPoints.append(investigatedPoint)

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
          branchedLevel[self.toBeSampled[key]] = utils.index(self.branchProbabilities[self.toBeSampled[key]],value)
        # The dictionary branchedLevel is stored in the xml tree too. That's because
        # the advancement of the thresholds must follow the tree structure
        elm.add('branchedLevel', branchedLevel)
        # Here it is stored all the info regarding the DET => we create the info for all the branchings and we store them
        #self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys())+1)] = ETS.NodeTree(copy.deepcopy(elm))
        self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys())+1)] = ETS.NodeTree(elm)
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
      else:  self.raiseAnError(IOError,self,'unknown mode '+xmlNode.attrib['mode']+'. Available are "online" and "post"!')
    if 'noTransitionStrategy' in xmlNode.attrib.keys():
      if xmlNode.attrib['noTransitionStrategy'].lower() == 'mc'    : self.noTransitionStrategy = 1
      elif xmlNode.attrib['noTransitionStrategy'].lower() == 'grid': self.noTransitionStrategy = 2
      else:  self.raiseAnError(IOError,self,'unknown noTransitionStrategy '+xmlNode.attrib['noTransitionStrategy']+'. Available are "mc" and "grid"!')
    if 'updateGrid' in xmlNode.attrib.keys():
      if xmlNode.attrib['updateGrid'].lower() in utils.stringsThatMeanTrue(): self.insertAdaptBPb = True

  def _generateDistributions(self,availableDist):
    DynamicEventTree._generateDistributions(self,availableDist)

  def localInitialize(self,solutionExport = None):
    if self.detAdaptMode == 2: self.startAdaptive = True
    DynamicEventTree.localInitialize(self)
    AdaptiveSampler.localInitialize(self,solutionExport=solutionExport)
    self._endJobRunnable    = sys.maxsize

  def generateInput(self,model,oldInput):
    return DynamicEventTree.generateInput(self, model, oldInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    returncode = DynamicEventTree.localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=False)
    if returncode:
      self._createRunningQueue(model,myInput)
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
    self.printTag = 'SAMPLER FACTORIAL DESIGN'
    # accepted types. full = full factorial, 2levelfract = 2-level fracional factorial, pb = Plackett-Burman design. NB. full factorial is equivalent to Grid sampling
    self.acceptedTypes = ['full','2levelfract','pb'] # accepted factorial types
    self.factOpt       = {}                          # factorial options (type,etc)
    self.designMatrix  = None                        # matrix container

  def localInputAndChecks(self,xmlNode):
    '''reading and construction of the grid'''
    Grid.localInputAndChecks(self,xmlNode)
    factsettings = xmlNode.find("FactorialSettings")
    if factsettings == None: self.raiseAnError(IOError,self,'FactorialSettings xml node not found!')
    facttype = factsettings.find("algorithm_type")
    if facttype == None: self.raiseAnError(IOError,self,'node "algorithm_type" not found in FactorialSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedTypes:self.raiseAnError(IOError,self,' "type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedTypes))
    self.factOpt['algorithm_type'] = facttype.text.lower()
    if self.factOpt['algorithm_type'] == '2levelfract':
      self.factOpt['options'] = {}
      self.factOpt['options']['gen'] = factsettings.find("gen")
      self.factOpt['options']['genMap'] = factsettings.find("genMap")
      if self.factOpt['options']['gen'] == None: self.raiseAnError(IOError,self,'node "gen" not found in FactorialSettings xml node!!!')
      if self.factOpt['options']['genMap'] == None: self.raiseAnError(IOError,self,'node "genMap" not found in FactorialSettings xml node!!!')
      self.factOpt['options']['gen'] = self.factOpt['options']['gen'].text.split(',')
      self.factOpt['options']['genMap'] = self.factOpt['options']['genMap'].text.split(',')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): self.raiseAnError(IOError,self,'number of variable in genMap != number of variables !!!')
      if len(self.factOpt['options']['gen']) != len(self.gridInfo.keys())   : self.raiseAnError(IOError,self,'number of variable in gen != number of variables !!!')
      rightOrder = [None]*len(self.gridInfo.keys())
      if len(self.factOpt['options']['genMap']) != len(self.factOpt['options']['gen']): self.raiseAnError(IOError,self,'gen and genMap different size!')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): self.raiseAnError(IOError,self,'number of gen attributes and variables different!')
      for ii,var in enumerate(self.factOpt['options']['genMap']):
        if var not in self.gridInfo.keys(): self.raiseAnError(IOError,self,' variable "'+var+'" defined in genMap block not among the inputted variables!')
        rightOrder[self.axisName.index(var)] = self.factOpt['options']['gen'][ii]
      self.factOpt['options']['orderedGen'] = rightOrder
    if self.factOpt['algorithm_type'] != 'full':
      self.externalgGridCoord = True
      for varname in self.gridInfo.keys():
        if len(self.gridInfo[varname][2]) != 2:
          self.raiseAnError(IOError,self,'The number of levels for type '+
                        self.factOpt['algorithm_type'] +' must be 2! In variable '+varname+ ' got number of levels = ' +
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
    if   self.factOpt['algorithm_type'] == '2levelfract': self.designMatrix = doe.fracfact(' '.join(self.factOpt['options']['orderedGen'])).astype(int)
    elif self.factOpt['algorithm_type'] == 'pb'         : self.designMatrix = doe.pbdesign(len(self.gridInfo.keys())).astype(int)
    if self.designMatrix != None:
      # convert all -1 in 0 => we can access to the grid info directly
      self.designMatrix[self.designMatrix == -1] = 0
      # the limit is the number of rows
      self.limit = self.designMatrix.shape[0]

  def localGenerateInput(self,model,myInput):
    if self.factOpt['algorithm_type'] == 'full':  Grid.localGenerateInput(self,model, myInput)
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
    self.printTag = 'SAMPLER RESPONSE SURF DESIGN'
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
    if factsettings == None: self.raiseAnError(IOError,self,'ResponseSurfaceDesignSettings xml node not found!')
    facttype = factsettings.find("algorithm_type")
    if facttype == None: self.raiseAnError(IOError,self,'node "algorithm_type" not found in ResponseSurfaceDesignSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedOptions.keys():self.raiseAnError(IOError,self,'"type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedOptions.keys()))
    self.respOpt['algorithm_type'] = facttype.text.lower()
    # set defaults
    if self.respOpt['algorithm_type'] == 'boxbehnken': self.respOpt['options'] = {'ncenters':None}
    else                                   : self.respOpt['options'] = {'centers':(4,4),'alpha':'orthogonal','face':'circumscribed'}
    for child in factsettings:
      if child.tag not in 'algorithm_type': self.respOpt['options'][child.tag] = child.text.lower()
    # start checking
    for key,value in self.respOpt['options'].items():
      if key not in self.acceptedOptions[facttype.text.lower()]:
        self.raiseAnError(IOError,self,'node '+key+' unknown. Available are "'+' '.join(self.acceptedOptions[facttype.text.lower()])+'"!!')
      if self.respOpt['algorithm_type'] == 'boxbehnken':
        if key == 'ncenters':
          try   : self.respOpt['options'][key] = int(value)
          except: self.raiseAnError(IOError,self,'"'+key+'" is not an integer!')
      else:
        if key == 'centers':
          if len(value.split(',')) != 2: self.raiseAnError(IOError,self,'"'+key+'" must be a comma separated string of 2 values only!')
          centers = value.split(',')
          try: self.respOpt['options'][key] = (int(centers[0]),int(centers[1]))
          except: self.raiseAnError(IOError,self,'"'+key+'" values must be integers!!')
        if key == 'alpha':
          if value not in ['orthogonal','rotatable']: self.raiseAnError(IOError,self,'Not recognized options for node ' +'"'+key+'". Available are "orthogonal","rotatable"!')
        if key == 'face':
          if value not in ['circumscribed','faced','inscribed']: self.raiseAnError(IOError,self,'Not recognized options for node ' +'"'+key+'". Available are "circumscribed","faced","inscribed"!')
    # fill in the grid
    if 'limit' in xmlNode.attrib.keys(): self.raiseAnError(IOError,self,'limit is not used in' +self.type+' sampler!')
    if not self.axisName: self.axisName = []
    for child in xmlNode:
      if child.tag == "Distribution": varName = "<distribution>"+child.attrib['name']
      elif child.tag == "variable"  : varName = child.attrib['name']
      for childChild in child:
        if childChild.tag =='boundaries':
          self.axisName.append(varName)
          if 'type' not in childChild.attrib.keys(): self.raiseAnError(IOError,self,'in block '+ childChild.tag + ' attribute type not found!')
          self.gridInfo[varName] = [childChild.attrib['type'],'custom',[]]
          lower = childChild.find("lower")
          upper = childChild.find("upper")
          if lower == None: self.raiseAnError(IOError,self,'node "lower" not found in '+childChild.tag+' block!')
          if upper == None: self.raiseAnError(IOError,self,'node "upper" not found in '+childChild.tag+' block!')
          try: self.bounds[varName] = (float(lower.text),float(upper.text))
          except: self.raiseAnError(IOError,self,'node "upper" or "lower" must be float')
    if len(self.toBeSampled.keys()) != len(self.gridInfo.keys()): self.raiseAnError(IOError,self,'inconsistency between number of variables and grid specification')
    self.gridCoordinate = [None]*len(self.axisName)
    if len(self.gridCoordinate) < self.minNumbVars[self.respOpt['algorithm_type']]: self.raiseAnError(IOError,self,'minimum number of variables for type "'+ self.respOpt['type'] +'" is '+str(self.minNumbVars[self.respOpt['type']])+'!!')
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
    if   self.respOpt['algorithm_type'] == 'boxbehnken'      : self.designMatrix = doe.bbdesign(len(self.gridInfo.keys()),center=self.respOpt['options']['ncenters'])
    elif self.respOpt['algorithm_type'] == 'centralcomposite': self.designMatrix = doe.ccdesign(len(self.gridInfo.keys()), center=self.respOpt['options']['centers'], alpha=self.respOpt['options']['alpha'], face=self.respOpt['options']['face'])
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

class SparseGridCollocation(Grid):
  def __init__(self):
    Grid.__init__(self)
    self.type           = 'SparseGridCollocationSampler'
    self.printTag       = 'SAMPLER '+self.type.upper()
    self.assemblerObjects={}    #dict of external objects required for assembly
    self.maxPolyOrder   = None  #L, the relative maximum polynomial order to use in any dimension
    self.indexSetType   = None  #TP, TD, or HC; the type of index set to use
    self.adaptive       = False #TODO
    self.polyDict       = {}    #varName-indexed dict of polynomial types
    self.quadDict       = {}    #varName-indexed dict of quadrature types
    self.importanceDict = {}    #varName-indexed dict of importance weights
    self.maxPolyOrder   = None  #integer, relative maximum polynomial order to be used in any one dimension
    self.lastOutput     = None  #pointer to output datas object
    self.ROM            = None  #pointer to ROM
    self.jobHandler     = None  #pointer to job handler for parallel runs
    self.doInParallel   = True  #compute sparse grid in parallel flag, recommended True

    self.requiredAssObject = (True,(['ROM'],['1']))                  # tuple. first entry boolean flag. True if the XML parser must look for assembler objects;

  def _localWhatDoINeed(self):
    gridDict = Grid._localWhatDoINeed(self)
    gridDict['internal'] = [(None,'jobHandler')]
    return gridDict

  def _localGenerateAssembler(self,initDict):
    Grid._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']

  def localInputAndChecks(self,xmlNode):
    self.doInParallel = xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes'] if 'parallel' in xmlNode.attrib.keys() else True
    self.writeOut = xmlNode.attrib['outfile'] if 'outfile' in xmlNode.attrib.keys() else None
    for child in xmlNode:
      if child.tag == 'Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
        self.axisName.append(varName)

  def localInitialize(self):
    for key in self.assemblerDict.keys():
      if 'ROM' in key:
        indice = 0
        for value in self.assemblerDict[key]:
          self.ROM = self.assemblerDict[key][indice][3]
          indice += 1
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0] #often need only one
    self._generateQuadsAndPolys(SVL)
    #print out the setup for each variable.
    msg=self.printTag,'INTERPOLATION INFO:\n'
    msg+='    Variable | Distribution | Quadrature | Polynomials\n'
    for v in self.quadDict.keys():
      msg+='   ',' | '.join([v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type])+'\n'
    msg+='    Polynomial Set Degree: '+str(self.maxPolyOrder)+'\n'
    msg+='    Polynomial Set Type  : '+str(SVL.indexSetType)+'\n'
    self.raiseADebug(self,msg)

    self.raiseADebug(self,'Starting index set generation...')
    self.indexSet = IndexSets.returnInstance(SVL.indexSetType,self)
    self.indexSet.initialize(self.distDict,self.importanceDict,self.maxPolyOrder,self.messageHandler)

    self.raiseADebug(self,'Starting sparse grid generation...')
    self.sparseGrid = Quadratures.SparseQuad()
    # NOTE this is the most expensive step thus far; try to do checks before here
    self.sparseGrid.initialize(self.indexSet,self.distDict,self.quadDict,self.jobHandler,self.messageHandler)

    if self.writeOut != None:
      msg=self.sparseGrid.__csv__()
      outFile=file(self.writeOut,'w')
      outFile.writelines(msg)
      outFile.close()

    self.limit=len(self.sparseGrid)
    self.raiseADebug(self,'Size of Sparse Grid  :'+str(self.limit))
    self.raiseADebug(self,'Finished sampler generation.')
    for SVL in self.ROM.SupervisedEngine.values():
      SVL.initialize({'SG':self.sparseGrid,
                      'dists':self.distDict,
                      'quads':self.quadDict,
                      'polys':self.polyDict,
                      'iSet':self.indexSet})

  def _generateQuadsAndPolys(self,SVL):
    '''
      Builds the quadrature objects, polynomial objects, and importance weights for all
      the distributed variables.  Also sets maxPolyOrder.
      @ In, SVL, one of the SupervisedEngine objects from the ROM
      @ Out, None
    '''
    ROMdata = SVL.interpolationInfo() #they are all the same? -> yes, I think so
    self.maxPolyOrder = SVL.maxPolyOrder
    #check input space consistency
    samVars=self.axisName[:]
    romVars=SVL.features[:]
    try:
      for v in self.axisName:
        samVars.remove(v)
        romVars.remove(v)
    except ValueError:
      self.raiseAnError(IOError,self,'variable '+v+' used in sampler but not ROM features! Collocation requires all vars in both.')
    if len(romVars)>0:
      self.raiseAnError(IOError,self,'variables '+str(romVars)+' specified in ROM but not sampler! Collocation requires all vars in both.')
    for v in ROMdata.keys():
      if v not in self.axisName:
        self.raiseAnError(IOError,self,'variable '+v+' given interpolation rules but '+v+' not in sampler!')
      else:
        self.gridInfo[v] = ROMdata[v] #quad, poly, weight
    #set defaults, then replace them if they're asked for
    for v in self.axisName:
      if v not in self.gridInfo.keys():
        self.gridInfo[v]={'poly':'DEFAULT','quad':'DEFAULT','weight':'1'}
    #establish all the right names for the desired types
    for varName,dat in self.gridInfo.items():
      if dat['poly'] == 'DEFAULT': dat['poly'] = self.distDict[varName].preferredPolynomials
      if dat['quad'] == 'DEFAULT': dat['quad'] = self.distDict[varName].preferredQuadrature
      polyType=dat['poly']
      subType = None
      distr = self.distDict[varName]
      if polyType == 'Legendre':
        if distr.type == 'Uniform':
          quadType=dat['quad']
        else:
          quadType='CDF'
          subType=dat['quad']
          if subType not in ['Legendre','ClenshawCurtis']:
            self.raiseAnError(IOError,self,'Quadrature '+subType+' not compatible with Legendre polys for '+distr.type+' for variable '+varName+'!')
      else:
        quadType=dat['quad']
      if quadType not in distr.compatibleQuadrature:
        self.raiseAnError(IOError,self,' Quadrature type "'+quadType+'" is not compatible with variable "'+varName+'" distribution "'+distr.type+'"')

      quad = Quadratures.returnInstance(quadType,self,Subtype=subType)
      quad.initialize(distr,self.messageHanlder)
      self.quadDict[varName]=quad

      poly = OrthoPolynomials.returnInstance(polyType,self)
      poly.initialize(quad,self.messageHandler)
      self.polyDict[varName] = poly

      self.importanceDict[varName] = float(dat['weight'])

  def localGenerateInput(self,model,myInput):
    '''Provide the next point in the sparse grid.'''
    pt,weight = self.sparseGrid[self.counter-1]
    for v,varName in enumerate(self.distDict.keys()):
      self.values[varName] = pt[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointsProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = weight
    self.inputInfo['SamplerType'] = 'Sparse Grid Collocation'

class Sobol(SparseGridCollocation):
  def __init__(self):
    '''
      Initializes members to be used in the sampler.
      @ In, None
      @ Out, None
    '''
    Grid.__init__(self)
    self.type           = 'SobolSampler'
    self.printTag       = 'SAMPLER SOBOL'
    self.assemblerObjects={}    #dict of external objects required for assembly
    self.maxPolyOrder   = None  #L, the relative maximum polynomial order to use in any dimension
    self.sobolOrder     = None  #S, the order of the HDMR expansion (1,2,3), queried from the sobol ROM
    self.indexSetType   = None  #TP, TD, or HC; the type of index set to use, queried from the sobol ROM
    self.polyDict       = {}    #varName-indexed dict of polynomial types
    self.quadDict       = {}    #varName-indexed dict of quadrature types
    self.importanceDict = {}    #varName-indexed dict of importance weights
    self.references     = {}    #reference (mean) values for distributions, by var
    self.solns          = None  #pointer to output datas object
    self.ROM            = None  #pointer to sobol ROM
    self.jobHandler     = None  #pointer to job handler for parallel runs
    self.doInParallel   = True  #compute sparse grid in parallel flag, recommended True

    self.requiredAssObject = (True,(['ROM'],['1']))                  # tuple. first entry boolean flag. True if the XML parser must look for assembler objects;

  def _localWhatDoINeed(self):
    '''
      Used to obtain necessary objects.  See base class.
      @ In, None
      @ Out, None
    '''
    gridDict = Grid._localWhatDoINeed(self)
    gridDict['internal'] = [(None,'jobHandler')]
    return gridDict

  def _localGenerateAssembler(self,initDict):
    '''
      Used to obtain necessary objects.  See base class.
      @ In, initDict, dictionary of objects required to initialize
      @ Out, None
    '''
    Grid._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']

  def localInputAndChecks(self,xmlNode):
    '''
      Extended readMoreXML after other objects are instantiated
      @ In, xmlNode, xmlNode object whose head should be Sobol under Sampler.
      @ Out, None
    '''
    self.doInParallel = xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes'] if 'parallel' in xmlNode.attrib.keys() else True
    self.writeOut = xmlNode.attrib['outfile'] if 'outfile' in xmlNode.attrib.keys() else None
    for child in xmlNode:
      if child.tag == 'Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
        self.axisName.append(varName)

  def localInitialize(self):
    '''
      Initializes Sampler, including building sub-ROMs for Sobol decomposition.  Note that re-using this
      sampler will destroy any ROM trained and attached to this sampler, and can be retrained after sampling.
      @ In, None
      @ Out, None
    '''
    for key in self.assemblerDict.keys():
      if 'ROM' in key:
        indice = 0
        for value in self.assemblerDict[key]:
          self.ROM = self.assemblerDict[key][indice][3]
          indice += 1
    #make combination of ROMs that we need
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0]
    self.sobolOrder = SVL.sobolOrder
    self._generateQuadsAndPolys(SVL)
    varis = SVL.features
    needCombos = itertools.chain.from_iterable(itertools.combinations(varis,r) for r in range(self.sobolOrder+1))
    self.SQs={}
    self.ROMs={}
    for combo in needCombos:
      if len(combo)==0:
        continue
      distDict={}
      quadDict={}
      polyDict={}
      imptDict={}
      limit=0
      for c in combo:
        distDict[c]=self.distDict[c]
        quadDict[c]=self.quadDict[c]
        polyDict[c]=self.polyDict[c]
        imptDict[c]=self.importanceDict[c]
      iset=IndexSets.returnInstance(SVL.indexSetType,self)
      iset.initialize(distDict,imptDict,SVL.maxPolyOrder)
      self.SQs[combo] = Quadratures.SparseQuad()
      self.SQs[combo].initialize(iset,SVL.maxPolyOrder,distDict,quadDict,polyDict,self.jobHandler)
      initDict={'IndexSet':iset, 'PolynomialOrder':SVL.maxPolyOrder, 'Interpolation':SVL.itpDict}
      initDict['Features']=','.join(combo)
      initDict['Target']=SVL.target #TODO make it work for multitarget
      self.ROMs[combo] = SupervisedLearning.returnInstance('GaussPolynomialRom',self,**initDict)
      initDict={'SG':self.SQs[combo], 'dists':distDict, 'quads':quadDict, 'polys':polyDict, 'iSet':iset}
      self.ROMs[combo].initialize(initDict)
    #make combined sparse grids
    self.references={}
    for var,dist in self.distDict.items():
      self.references[var]=dist.untruncatedMean()
    std = self.distDict.keys()
    self.pointsToRun=[]
    #make sure reference case gets in there
    newpt = np.zeros(len(self.distDict))
    for v,var in enumerate(self.distDict.keys()):
      newpt[v] = self.references[var]
    self.pointsToRun.append(tuple(newpt))
    #now do the rest
    for combo,rom in self.ROMs.items():
      SG = rom.sparseGrid
      SG._remap(combo)
      for l in range(len(SG)):
        pt,wt = SG[l]
        newpt = np.zeros(len(std))
        for v,var in enumerate(std):
          if var in combo: newpt[v] = pt[combo.index(var)]
          else: newpt[v] = self.references[var]
        newpt=tuple(newpt)
        if newpt not in self.pointsToRun: self.pointsToRun.append(newpt)
    self.limit = len(self.pointsToRun)
    initdict={'ROMs':self.ROMs,
              'SG':self.SQs,
              'dists':self.distDict,
              'quads':self.quadDict,
              'polys':self.polyDict,
              'refs':self.references}
    self.ROM.SupervisedEngine.values()[0].initialize(initdict)

  def localGenerateInput(self,model,myInput):
    '''Provide the next point in the sparse grid.  Note that this sampler cannot assign probabilty
       weights to individual points, as several sub-ROMs will use them with different weights.
       See base class.'''
    pt = self.pointsToRun[self.counter-1]
    for v,varName in enumerate(self.distDict.keys()):
      self.values[varName] = pt[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointsProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    #self.inputInfo['ProbabilityWeight'] =  N/A
    self.inputInfo['SamplerType'] = 'Sparse Grids for Sobol'

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
__interFaceDict['SparseGridCollocation'   ] = SparseGridCollocation
__interFaceDict['Sobol'                   ] = Sobol
__knownTypes = list(__interFaceDict.keys())

def addKnownTypes(newDict):
  for name, value in newDict.items():
    __interFaceDict[name]=value

def knownTypes():
  return __knownTypes

def addKnownTypes(newDict):
  for name, value in newDict.items():
    __interFaceDict[name]=value
    __knownTypes.append(name)

def returnInstance(Type,caller):
  '''
  function used to generate a Sampler class
  @ In, Type : Sampler type
  @ Out,Instance of the Specialized Sampler class
  '''
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'SAMPLERS','not known '+__base+' type '+Type)

def optionalInputs(Type):
  pass

def mandatoryInputs(Type):
  pass
