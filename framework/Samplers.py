"""
Module where the base class and the specialization of different type of sampler are
"""
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
from operator import mul,itemgetter
from collections import OrderedDict
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
import MessageHandler
import GridEntities
import DataObjects
import Models
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

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
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    BaseType.__init__(self)
    self.counter                       = 0                         # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.auxcnt                        = 0                         # Aux counter of samples performed (for its usage check initialize method)
    self.limit                         = sys.maxsize               # maximum number of Samples (for example, Monte Carlo = Number of HistorySet to run, DET = Unlimited)
    self.toBeSampled                   = {}                        # Sampling mapping dictionary {'Variable Name':'name of the distribution'}
    self.dependentSample               = {}                        # Sampling mapping dictionary for dependent variables {'Variable Name':'name of the external function'}
    self.distDict                      = {}                        # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.funcDict                      = {}                        # Contains the instance of the function     to be used, it is created every time the sampler is initialized. keys are the variable names
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
    self.restartData                   = None                      # presampled points to restart from

    self._endJobRunnable               = sys.maxsize               # max number of inputs creatable by the sampler right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)

    ######
    self.variables2distributionsMapping = {}                       # for each variable 'varName'  , the following informations are included:  'varName': {'dim': 1, 'totDim': 2, 'name': 'distName'} ; dim = dimension of the variable; totDim = total dimensionality of its associated distribution
    self.distributions2variablesMapping = {}                       # for each variable 'distName' , the following informations are included: 'distName': [{'var1': 1}, {'var2': 2}]} where for each var it is indicated the var dimension
    self.ND_sampling_params             = {}                       # this dictionary contains a dictionary for each ND distribution (key). This latter dictionary contains the initialization parameters of the ND inverseCDF ('initial_grid_disc' and 'tolerance')
    ######

    self.assemblerObjects               = {}                       # {MainClassName(e.g.Distributions):[class(e.g.Models),type(e.g.ROM),objectName]}
    #self.requiredAssObject             = (False,([],[]))          # tuple. first entry boolean flag. True if the XML parser must look for objects;
                                                                   # second entry tuple.first entry list of object can be retrieved, second entry multiplicity (-1,-2,-n means optional (max 1 object,2 object, no number limit))
    self.requiredAssObject              = (True,(['Restart','function'],['-n','-n']))
    self.assemblerDict                  = {}                       # {'class':[['subtype','name',instance]]}

  def _localGenerateAssembler(self,initDict):
    """ see generateAssembler method """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    self._generateDistributions(availableDist,availableFunc)

  def _addAssObject(self,name,flag):
    """
      Method to add required assembler objects to the requiredAssObject dictionary.
      @ In, name, the node name to search for
      @ In, flag, the number of nodes to look for (- means optional, n means any number)
      @ Out, None
    """
    self.requiredAssObject[1][0].append(name)
    self.requiredAssObject[1][1].append(flag)

  def _localWhatDoINeed(self):
    """
    This method is a local mirror of the general whatDoINeed method.
    It is implemented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    """
    needDict = {}
    needDict['Distributions'] = [] # Every sampler requires Distributions OR a Function
    needDict['Functions']     = [] # Every sampler requires Distributions OR a Function
    for dist in self.toBeSampled.values():     needDict['Distributions'].append((None,dist))
    for func in self.dependentSample.values(): needDict['Functions'].append((None,func))
    return needDict

  def _readMoreXML(self,xmlNode):
    """
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    The text i supposed to contain the info where and which variable to change.
    In case of a code the syntax is specified by the code interface itself
    """

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
        foundDistOrFunc = False
        for childChild in child:
          if childChild.tag =='distribution':
            if not foundDistOrFunc: foundDistOrFunc = True
            else: self.raiseAnError(IOError,'A sampled variable cannot have both a distribution and a function!')
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
          elif childChild.tag == 'function':
            if not foundDistOrFunc: foundDistOrFunc = True
            else: self.raiseAnError(IOError,'A sampled variable cannot have both a distribution and a function!')
            tobesampled = childChild.text
            varData['name']=childChild.text
            self.dependentSample[prefix+child.attrib['name']] = tobesampled
        if not foundDistOrFunc: self.raiseAnError(IOError,'Sampled variable',child.attrib['name'],'has neither a <distribution> nor <function> node specified!')
      elif child.tag == "sampler_init":
        self.initSeed = Distributions.randomIntegers(0,2**31,self)
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
                  self.raiseAnError(IOError,'Unknown tag '+childChildChildChild.tag+' .Available are: initial_grid_disc and tolerance!')
              self.ND_sampling_params[childChildChild.attrib['name']] = NDdistData
          else: self.raiseAnError(IOError,'Unknown tag '+child.tag+' .Available are: limit, initial_seed, reseed_at_each_iteration and dist_init!')

    if self.initSeed == None:
      self.initSeed = Distributions.randomIntegers(0,2**31,self)

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
      listvar = self.distributions2variablesMapping[dist]
      for var in listvar:
        if var.values()[0] > maxDim:
          maxDim = var.values()[0]
      self.variables2distributionsMapping[key]['totDim'] = maxDim #len(self.distributions2variablesMapping[self.variables2distributionsMapping[key]['name']])
    self.localInputAndChecks(xmlNode)


  def read_sampler_init(self,xmlNode):
    """
    This method is responsible to read only the sampler_init block in the .xml file.
    This method has been moved from the base sampler class since the sampler_init block is needed only for the MC and stratified (LHS) samplers
    @ In xmlNode
    """
    for child in xmlNode:
      if child.tag == "sampler_init":
        self.initSeed = Distributions.randomIntegers(0,2**31,self)
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
                  self.raiseAnError(IOError,'Unknown tag '+childChildChildChild.tag+' .Available are: initial_grid_disc and tolerance!')
              self.ND_sampling_params[childChildChild.attrib['name']] = NDdistData
          else: self.raiseAnError(IOError,'Unknown tag '+child.tag+' .Available are: limit, initial_seed, reseed_at_each_iteration and dist_init!')

  def endJobRunnable(self):
    """
    Returns the maximum number of inputs allowed to be created by the sampler
    right after a job ends (e.g., infinite for MC, 1 for Adaptive, etc)
    """
    return self._endJobRunnable

  def localInputAndChecks(self,xmlNode):
    """place here the additional reading, remember to add initial parameters in the method localAddInitParams"""
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

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distrbutions and functions.
      @ In, availDist, dict of distributions
      @ In, availDist, dict of functions
      @Out, None
    """
    if self.initSeed != None:
      Distributions.randomSeed(self.initSeed)
    for key in self.toBeSampled.keys():
      if self.toBeSampled[key] not in availableDist.keys(): self.raiseAnError(IOError,'Distribution '+self.toBeSampled[key]+' not found among available distributions (check input)!')
      self.distDict[key] = availableDist[self.toBeSampled[key]]
      self.inputInfo['crowDist'][key] = json.dumps(self.distDict[key].getCrowDistDict())
    for key,val in self.dependentSample.items():
      if val not in availableFunc.keys(): self.raiseAnError('Function',val,'was not found amoung the available functions:',availableFunc.keys())
      self.funcDict[key] = availableFunc[val]

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

    #grab restart dataobject if it's available, then in localInitialize the sampler can deal with it.
    if 'Restart' in self.assemblerDict.keys():
      self.raiseADebug('Restart object: '+str(self.assemblerDict['Restart']))
      self.restartData = self.assemblerDict['Restart'][0][3]
      self.raiseAMessage('Restarting from '+self.restartData.name)
      #check consistency of data
      try:
        rdata = self.restartData.getAllMetadata()['crowDist']
        sdata = self.inputInfo['crowDist']
        self.raiseAMessage('sampler inputs:')
        for sk,sv in sdata.items():
          self.raiseAMessage('|   '+str(sk)+': '+str(sv))
        for i,r in enumerate(rdata):
          if type(r) != dict: continue
          if not r==sdata:
            self.raiseAMessage('restart inputs %i:' %i)
            for rk,rv in r.items():
              self.raiseAMessage('|   '+str(rk)+': '+str(rv))
            self.raiseAnError(IOError,'Restart "%s" data[%i] does not have same inputs as sampler!' %(self.restartData.name,i))
      except KeyError as e:
        self.raiseAWarning("No CROW distribution available in restart -",e)
    else:
      self.raiseAMessage('No restart for '+self.printTag)

    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport != None : self.localInitialize(solutionExport=solutionExport)
    else                      : self.localInitialize()

    for distrib in self.ND_sampling_params:
      if distrib in self.distributions2variablesMapping:
        params = self.ND_sampling_params[distrib]
        temp = self.distributions2variablesMapping[distrib][0].keys()[0]
        self.distDict[temp].updateRNGParam(params)
      else:
        self.raiseAnError(IOError,'Distribution "%s" specified in dist_init block of sampler "%s" does not exist!' %(distrib,self.name))

  def localInitialize(self):
    """
    use this function to add initialization features to the derived class
    it is call at the beginning of each step
    """
    pass

  def amIreadyToProvideAnInput(self): #inLastOutput=None):
    """
    This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
    This method act as a "traffic light" for generating a new input.
    Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
    @ In, None, None
    @ Out, ready, Boolean
    """
    if(self.counter < self.limit): ready = True
    else                         : ready = False
    ready = self.localStillReady(ready)
    return ready

  def localStillReady(self,ready): #,lastOutput=None
    """Use this function to change the ready status"""
    return ready

  def generateInput(self,model,oldInput):
    """
    This method have to be overwrote to provide the specialization for the specific sampler
    The model instance in might be needed since, especially for external codes,
    only the code interface possesses the dictionary for reading the variable definition syntax
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    @return     : [] containing the new inputs -in reality it is the model that return this the Sampler generate the value to be placed in the intput the model
    """
    self.counter +=1                              #since we are creating the input for the next run we increase the counter and global counter
    self.auxcnt  +=1
    if self.reseedAtEachIteration: Distributions.randomSeed(self.auxcnt-1)
    self.inputInfo['prefix'] = str(self.counter)
    model.getAdditionalInputEdits(self.inputInfo)
    self.localGenerateInput(model,oldInput)
    # generate the function variable values
    for var,funcName in self.dependentSample.items():
      test=self.funcDict[var].evaluate(var,self.values)
      self.values[var] = test
    return model.createNewInput(oldInput,self.type,**self.inputInfo)

  @abc.abstractmethod
  def localGenerateInput(self,model,oldInput):
    """
    This class need to be overwritten since it is here that the magic of the sampler happens.
    After this method call the self.inputInfo should be ready to be sent to the model
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    """
    pass

  def generateInputBatch(self,myInput,model,batchSize,projector=None): #,lastOutput=None
    """
    this function provide a mask to create several inputs at the same time
    It call the generateInput function as many time as needed
    @in myInput: [] list containing one input set
    @in model: instance of a model
    @in batchSize: integer the number of input sets required
    @in projector used for adaptive sampling to provide the projection of the solution on the success metric
    @return newInputs: [[]] list of the list of input sets"""
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
    """
    This function is used by samplers that need to collect information from a
    finished run.
    Provides a generic interface that all samplers will use, for specifically
    handling any sub-class, the localFinalizeActualSampling should be overridden
    instead, as finalizeActualSampling provides only generic functionality
    shared by all Samplers and will in turn call the localFinalizeActualSampling
    before returning.
    @in jobObject: an instance of a JobHandler
    @in model    : an instance of a model
    @in myInput  : the generating input
    """
    self.localFinalizeActualSampling(jobObject,model,myInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
    Overwrite only if you need something special at the end of each run....
    This function is used by samplers that need to collect information from the just ended run
    For example, for a Dynamic Event Tree case, this function can be used to retrieve
    the information from the just finished run of a branch in order to retrieve, for example,
    the distribution name that caused the trigger, etc.
    It is a essentially a place-holder for most of the sampler to remain compatible with the StepsCR structure
    @in jobObject: an instance of a JobHandler
    @in model    : an instance of a model
    @in myInput  : the generating input
    """
    pass

  def handleFailedRuns(self,failedRuns):
    """Collects the failed runs from the Step and allows samples to handle them individually if need be.
    @ In, failedRuns, list of JobHandler.ExternalRunner objects
    @Out, None
    """
    self.raiseADebug('===============')
    self.raiseADebug('| RUN SUMMARY |')
    self.raiseADebug('===============')
    if len(failedRuns)>0:
      self.raiseAWarning('There were %i failed runs!  Run with verbosity = debug for more details.' %(len(failedRuns)))
      for run in failedRuns:
        metadata = run.returnMetadata()
        self.raiseADebug('  Run number %s FAILED:' %run.identifier,run.command)
        self.raiseADebug('      return code :',run.getReturnCode())
        self.raiseADebug('      sampled vars:')
        for v,k in metadata['SampledVars'].items():
          self.raiseADebug('         ',v,':',k)
    else:
      self.raiseADebug('All runs completed without returning errors.')
    self._localHandleFailedRuns(failedRuns)
    self.raiseADebug('===============')
    self.raiseADebug('  END SUMMARY  ')
    self.raiseADebug('===============')

  def _localHandleFailedRuns(self,failedRuns):
    """Specialized method for samplers to handle failed runs.  Defaults to failing runs.
    @ In, failedRuns, list of JobHandler.ExternalRunner objects
    @Out, None
    """
    if len(failedRuns)>0:
      self.raiseAnError(IOError,'There were failed runs; aborting RAVEN.')
#
#
#
#

class StaticSampler(Sampler):
  """This is a general static, blind, once-through sampler"""
  pass
#
#
#
#
class AdaptiveSampler(Sampler):
  """This is a general adaptive sampler"""
  pass



class LimitSurfaceSearch(AdaptiveSampler):
  """
  A sampler that will adaptively locate the limit surface of a given problem
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Sampler.__init__(self)
    self.goalFunction     = None             #this is the pointer to the function defining the goal
    self.tolerance        = None             #this is norm of the error threshold
    self.subGridTol       = None             #This is the tolerance used to construct the testing sub grid
    self.toleranceWeight  = 'cdf'            #this is the a flag that controls if the convergence is checked on the hyper-volume or the probability
    self.persistence      = 5                #this is the number of times the error needs to fell below the tollerance before considering the sim converged
    self.repetition       = 0                #the actual number of time the error was below the requested threshold
    self.forceIteration   = False            #this flag control if at least a self.limit number of iteration should be done
    self.axisName         = None             #this is the ordered list of the variable names (ordering match self.gridStepSize anfd the ordering in the test matrixes)
    self.oldTestMatrix    = None             #This is the test matrix to use to store the old evaluation of the function
    self.solutionExport   = None             #This is the data used to export the solution (it could also not be present)
    self.nVar             = 0                #this is the number of the variable sampled
    self.surfPoint        = None             #coordinate of the points considered on the limit surface
    self.hangingPoints    = []               #list of the points already submitted for evaluation for which the result is not yet available
    # postprocessor to compute the limit surface
    self.limitSurfacePP   = None
    self.printTag         = 'SAMPLER ADAPTIVE'

    self._addAssObject('TargetEvaluation','n')
    self._addAssObject('ROM','n')
    self._addAssObject('Function','-n')

  def localInputAndChecks(self,xmlNode):
    """
    Class specific xml inputs will be read here and checked for validity.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    @ Out, None
    """
    if 'limit' in xmlNode.attrib.keys():
      try: self.limit = int(xmlNode.attrib['limit'])
      except ValueError: self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    # convergence Node
    convergenceNode = xmlNode.find('Convergence')
    if convergenceNode==None:self.raiseAnError(IOError,'the node Convergence was missed in the definition of the adaptive sampler '+self.name)
    try   : self.tolerance=float(convergenceNode.text)
    except: self.raiseAnError(IOError,'Failed to convert '+convergenceNode.text+' to a meaningful number for the convergence')
    attribList = list(convergenceNode.attrib.keys())
    if 'limit'          in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('limit'))
      try   : self.limit = int (convergenceNode.attrib['limit'])
      except: self.raiseAnError(IOError,'Failed to convert the limit value '+convergenceNode.attrib['limit']+' to a meaningful number for the convergence')
    if 'persistence'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('persistence'))
      try   : self.persistence = int (convergenceNode.attrib['persistence'])
      except: self.raiseAnError(IOError,'Failed to convert the persistence value '+convergenceNode.attrib['persistence']+' to a meaningful number for the convergence')
    if 'weight'         in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('weight'))
      try   : self.toleranceWeight = str(convergenceNode.attrib['weight']).lower()
      except: self.raiseAnError(IOError,'Failed to convert the weight type '+convergenceNode.attrib['weight']+' to a meaningful string for the convergence')
    if 'subGridTol'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('subGridTol'))
      try   : self.subGridTol = float (convergenceNode.attrib['subGridTol'])
      except: self.raiseAnError(IOError,'Failed to convert the subGridTol '+convergenceNode.attrib['subGridTol']+' to a meaningful float for the convergence')
    if 'forceIteration' in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('forceIteration'))
      if   convergenceNode.attrib['forceIteration']=='True' : self.forceIteration   = True
      elif convergenceNode.attrib['forceIteration']=='False': self.forceIteration   = False
      else: self.raiseAnError(RuntimeError,'Reading the convergence setting for the adaptive sampler '+self.name+' the forceIteration keyword had an unknown value: '+str(convergenceNode.attrib['forceIteration']))
    #assembler node: Hidden from User
    # set subgrid
    if self.subGridTol == None: self.subGridTol = self.tolerance
    if self.subGridTol > self.tolerance: self.raiseAnError(IOError,'The sub grid tolerance '+str(self.subGridTol)+' must be smaller than the tolerance: '+str(self.tolerance))
    if len(attribList)>0: self.raiseAnError(IOError,'There are unknown keywords in the convergence specifications: '+str(attribList))

  def localAddInitParams(self,tempDict):
    """
    Appends a given dictionary with class specific member variables and their
    associated initialized values.
    @ InOut, tempDict: The dictionary where we will add the initialization
                       parameters specific to this Sampler.
    """
    tempDict['Iter. forced'    ] = str(self.forceIteration)
    tempDict['Norm tolerance'  ] = str(self.tolerance)
    tempDict['Sub grid size'   ] = str(self.subGridTol)
    tempDict['Error Weight'    ] = str(self.toleranceWeight)
    tempDict['Persistence'     ] = str(self.repetition)

  def localAddCurrentSetting(self,tempDict):
    """
    Appends a given dictionary with class specific information regarding the
    current status of the object.
    @ InOut, tempDict: The dictionary where we will add the parameters specific
                       to this Sampler and their associated values.
    """
    if self.solutionExport!=None:
      tempDict['The solution is exported in '    ] = 'Name: ' + self.solutionExport.name + 'Type: ' + self.solutionExport.type
    if self.goalFunction!=None:
      tempDict['The function used is '] = self.goalFunction.name

  def localInitialize(self,solutionExport=None):
    """
    Will perform all initialization specific to this Sampler. For instance,
    creating an empty container to hold the identified surface points, error
    checking the optionally provided solution export and other preset values,
    and initializing the limit surface Post-Processor used by this sampler.

    @ InOut, solutionExport: a PointSet to hold the solution (a list of limit
                             surface points)
    """
    self.limitSurfacePP   = PostProcessors.returnInstance("LimitSurface",self)
    if 'Function' in self.assemblerDict.keys(): self.goalFunction = self.assemblerDict['Function'][0][3]
    if 'TargetEvaluation' in self.assemblerDict.keys(): self.lastOutput = self.assemblerDict['TargetEvaluation'][0][3]
    #self.memoryStep        = 5               # number of step for which the memory is kept
    self.solutionExport    = solutionExport
    # check if solutionExport is actually a "DataObjects" type "PointSet"
    if type(solutionExport).__name__ != "PointSet": self.raiseAnError(IOError,'solutionExport type is not a PointSet. Got '+ type(solutionExport).__name__+'!')
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.oldTestMatrix     = None             #This is the test matrix to use to store the old evaluation of the function
    self.persistenceMatrix = None             #this is a matrix that for each point of the testing grid tracks the persistence of the limit surface position
    if self.goalFunction.name not in self.solutionExport.getParaKeys('output'): self.raiseAnError(IOError,'Goal function name does not match solution export data output.')
    # set number of job requestable after a new evaluation
    self._endJobRunnable   = 1
    #check if convergence is not on probability if all variables are bounded in value otherwise the problem is unbounded
    if self.toleranceWeight=='value':
      for varName in self.distDict.keys():
        if not(self.distDict[varName].upperBoundUsed and self.distDict[varName].lowerBoundUsed):
          self.raiseAnError(TypeError,'It is impossible to converge on an unbounded domain (variable '+varName+' with distribution '+self.distDict[varName].name+') as requested to the sampler '+self.name)
    elif self.toleranceWeight=='cdf': pass
    else: self.raiseAnError(IOError,'Unknown weight string descriptor: '+self.toleranceWeight)
    #setup the grid. The grid is build such as each element has a volume equal to the sub grid tolerance
    #the grid is build in such a way that an unit change in each node within the grid correspond to a change equal to the tolerance
    self.nVar         = len(self.distDict.keys())              # Total number of variables
    bounds          = {"lowerBounds":{},"upperBounds":{}}
    transformMethod = {}
    for varName in self.distDict.keys():
      if self.toleranceWeight!='cdf': bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = self.distDict[varName].lowerBound, self.distDict[varName].upperBound
      else:
        bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = 0.0, 1.0
        transformMethod[varName.replace('<distribution>','')] = self.distDict[varName].ppf
    #moving forward building all the information set
    self.axisName = self.distDict.keys()
    self.axisName.sort()
    # initialize LimitSurface PP
    self.limitSurfacePP._initFromDict({"parameters":[key.replace('<distribution>','') for key in self.axisName],"tolerance":self.subGridTol,"side":"both","transformationMethods":transformMethod,"bounds":bounds})
    self.limitSurfacePP.assemblerDict = self.assemblerDict
    self.limitSurfacePP._initializeLSpp({'WorkingDir':None},[self.lastOutput],{})
    self.persistenceMatrix        = np.zeros(self.limitSurfacePP.getTestMatrix().shape) #matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.oldTestMatrix            = np.zeros(self.limitSurfacePP.getTestMatrix().shape) #swap matrix fro convergence test
    self.hangingPoints            = np.ndarray((0, self.nVar))
    self.raiseADebug('Initialization done')

  def localStillReady(self,ready): #,lastOutput=None
    """
    first perform some check to understand what it needs to be done possibly perform an early return
    ready is returned
    lastOutput should be present when the next point should be chosen on previous iteration and convergence checked
    lastOutput it is not considered to be present during the test performed for generating an input batch
    ROM if passed in it is used to construct the test matrix otherwise the nearest neightburn value is used
    """
    self.raiseADebug('From method localStillReady...')
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
    self.raiseADebug('Training finished')
    np.copyto(self.oldTestMatrix,self.limitSurfacePP.getTestMatrix())    #copy the old solution (contained in the limit surface PP) for convergence check
    # evaluate the Limit Surface coordinates (return input space coordinates, evaluation vector and grid indexing)
    self.surfPoint, evaluations, listsurfPoint = self.limitSurfacePP.run(returnListSurfCoord = True)

    self.raiseADebug('Prediction finished')
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
    self.raiseADebug('counter: '+str(self.counter)+'       Error: ' +str(testError)+' Repetition: '+str(self.repetition))
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
    # create values dictionary
    """compute the direction normal to the surface, compute the derivative normal to the surface of the probability,
     check the points where the derivative probability is the lowest"""

    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    self.raiseADebug('generating input')
    varSet=False
    if self.surfPoint!=None and len(self.surfPoint)>0:
      sampledMatrix = np.zeros((len(self.limitSurfacePP.getFunctionValue()[self.axisName[0].replace('<distribution>','')])+len(self.hangingPoints[:,0]),len(self.axisName)))
      for varIndex, name in enumerate([key.replace('<distribution>','') for key in self.axisName]): sampledMatrix [:,varIndex] = np.append(self.limitSurfacePP.getFunctionValue()[name],self.hangingPoints[:,varIndex])
      distanceTree = spatial.cKDTree(copy.copy(sampledMatrix),leafsize=12)
      #the hanging point are added to the list of the already explored points so not to pick the same when in //
      #      lastPoint = [self.functionValue[name][-1] for name in [key.replace('<distribution>','') for key in self.axisName]]
      #      for varIndex, name in enumerate([key.replace('<distribution>','') for key in self.axisName]): tempDict[name] = np.append(self.functionValue[name],self.hangingPoints[:,varIndex])
      tempDict = {}
      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
        tempDict[varName]     = self.surfPoint[:,varIndex]
        self.inputInfo['distributionName'][self.axisName[varIndex]] = self.toBeSampled[self.axisName[varIndex]]
        self.inputInfo['distributionType'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].type
      #distLast = np.sqrt(distLast)
      distance, _ = distanceTree.query(self.surfPoint)
      distance = np.multiply(distance,self.invPointPersistence)
      if np.max(distance)>0.0:
        for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
          self.values[self.axisName[varIndex]] = copy.copy(float(self.surfPoint[np.argmax(distance),varIndex]))
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True
      else: self.raiseADebug('np.max(distance)=0.0')
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
    self.raiseADebug('At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
    self.inputInfo['SamplerType'] = 'Adaptive'
    self.inputInfo['subGridTol' ] = self.subGridTol

    #      This is the normal derivation to be used later on
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
    """generate representation of goal function"""
    pass

#
#
#
#
class MonteCarlo(Sampler):
  """MONTE CARLO Sampler"""
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Sampler.__init__(self)
    self.printTag = 'SAMPLER MONTECARLO'

  def localInputAndChecks(self,xmlNode):
    """
    Class specific xml inputs will be read here and checked for validity.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    @ Out, None
    """
    Sampler.read_sampler_init(self,xmlNode)

    if xmlNode.find('sampler_init')!= None:
      if xmlNode.find('sampler_init').find('limit')!= None:
        try: self.limit = int(xmlNode.find('sampler_init').find('limit').text)
        except ValueError:
          self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
      else:
        utils.raiseAnError(IOError,self,'Monte Carlo sampler '+self.name+' needs the limit block (number of samples) in the sampler_init block')
    else:
      utils.raiseAnError(IOError,self,'Monte Carlo sampler '+self.name+' needs the sampler_init block')

  def localInitialize(self):
    """
    Will perform all initialization specific to this Sampler. This will be
    called at the beginning of each Step where this object is used. See base
    class for more details.
    """
    if self.restartData:
      self.counter+=len(self.restartData)
      self.raiseAMessage('Number of points from restart: %i' %self.counter)
      self.raiseAMessage('Number of points needed:       %i' %(self.limit-self.counter))
    #pass #TODO fix the limit based on restartData

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
    (set up self.inputInfo before being sent to the model)
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs (unused)
      @ Out, None
    """
    # create values dictionary
    for key in self.distDict:
      # check if the key is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables

      dim    = self.variables2distributionsMapping[key]['dim']
      totDim = self.variables2distributionsMapping[key]['totDim']
      dist   = self.variables2distributionsMapping[key]['name']

      for var in self.distributions2variablesMapping[dist]:
        if dim == 1:
          rvsnum = self.distDict[key].rvs()
          varID  = var.keys()[0]
          varDim = var[varID]
          for kkey in varID.strip().split(','):
            self.values[kkey] = np.atleast_1d(rvsnum)[varDim-1]
            if totDim > 1:
              coordinate=[];
              for i in range(totDim):
                coordinate.append(np.atleast_1d(rvsnum)[i])
              self.inputInfo['SampledVarsPb'][kkey] = self.distDict[key].pdf(coordinate)
            elif totDim == 1:
              self.inputInfo['SampledVarsPb'][kkey] = self.distDict[key].pdf(self.values[kkey])
            else:
              self.inputInfo['SampledVarsPb'][kkey] = 1.0
      #else? #FIXME

    if len(self.inputInfo['SampledVarsPb'].keys()) > 0:
      self.inputInfo['PointProbability'  ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
      #self.inputInfo['ProbabilityWeight' ] = 1.0 #MC weight is 1/N => weight is one
    self.inputInfo['SamplerType'] = 'MC'

  def _localHandleFailedRuns(self,failedRuns):
    """Specialized method for samplers to handle failed runs.  Defaults to failing runs.
    @ In, failedRuns, list of JobHandler.ExternalRunner objects
    @Out, None
    """
    if len(failedRuns)>0: self.raiseADebug('  Continuing with reduced-size Monte Carlo sampling.')

#
#
#
#
class Grid(Sampler):
  """
  Samples the model on a given (by input) set of points
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Sampler.__init__(self)
    self.printTag = 'SAMPLER GRID'
    self.axisName             = []    # the name of each axis (variable)
    self.gridInfo             = {}    # {'name of the variable':Type}  --> Type: CDF/Value
    self.externalgGridCoord   = False # boolean attribute. True if the coordinate list has been filled by external source (see factorial sampler)
    self.gridCoordinate       = []    # current grid coordinates
    self.existing             = []    # restart points
    self.gridEntity           = GridEntities.returnInstance('GridEntity',self)

  def localInputAndChecks(self,xmlNode):
    """
    Class specific xml inputs will be read here and checked for validity.
    Specifically, reading and construction of the grid for this Sampler.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    @ Out, None
    """
    if 'limit' in xmlNode.attrib.keys(): self.raiseAnError(IOError,'limit is not used in Grid sampler')
    self.limit = 1
    self.gridEntity._readMoreXml(xmlNode,dimensionTags=["variable","Distribution"],messageHandler=self.messageHandler, dimTagsPrefix={"Distribution":"<distribution>"})
    grdInfo = self.gridEntity.returnParameter("gridInfo")
    for axis, value in grdInfo.items(): self.gridInfo[axis] = value[0]
    if len(self.toBeSampled.keys()) != len(grdInfo.keys()): self.raiseAnError(IOError,'inconsistency between number of variables and grid specification')
    self.axisName = grdInfo.keys()
    self.axisName.sort()

  def localAddInitParams(self,tempDict):
    """
    Appends a given dictionary with class specific member variables and their
    associated initialized values.
    @ InOut, tempDict: The dictionary where we will add the initialization
                       parameters specific to this Sampler.
    """
    for variable,value in self.gridInfo.items():
      tempDict[variable+' is sampled using a grid in '] = value

  def localAddCurrentSetting(self,tempDict):
    """
    Appends a given dictionary with class specific information regarding the
    current status of the object.
    @ InOut, tempDict: The dictionary where we will add the parameters specific
                       to this Sampler and their associated values.
    """
    for var, value in self.values.items():
      tempDict['coordinate '+var+' has value'] = value

  def localInitialize(self):
    """
    This is used to check if the points and bounds are compatible with the distribution provided.
    It could not have been done earlier since the distribution might not have been initialized first
    """
    self.gridEntity.initialize()
    self.limit = len(self.gridEntity)
    if self.restartData is not None:
      inps = self.restartData.getInpParametersValues()
      self.existing = zip(*list(v for v in inps.values()))

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs (unused)
      @ Out, None
    """
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    weight = 1.0
    found=False
    while not found:
      recastDict = {}
      for i in range(len(self.axisName)):
        varName = self.axisName[i]
        if self.gridInfo[varName]=='CDF':
          if self.distDict[varName].getDimensionality()==1: recastDict[varName] = [self.distDict[varName].ppf]
          else: recastDict[varName] = [self.distDict[varName].inverseMarginalDistribution,[self.variables2distributionsMapping[varName]['dim']-1]]
        elif self.gridInfo[varName]!='value': self.raiseAnError(IOError,self.gridInfo[varName]+' is not know as value keyword for type. Sampler: '+self.name)
      if self.externalgGridCoord: currentIndexes, coordinates = self.gridEntity.returnIteratorIndexesFromIndex(self.gridCoordinate), self.gridEntity.returnCoordinateFromIndex(self.gridCoordinate, True, recastDict)
      else                      : currentIndexes, coordinates = self.gridEntity.returnIteratorIndexes(), self.gridEntity.returnPointAndAdvanceIterator(True,recastDict)
      if coordinates == None: raise utils.NoMoreSamplesNeeded
      coordinatesPlusOne  = self.gridEntity.returnShiftedCoordinate(currentIndexes,dict.fromkeys(self.axisName,1))
      coordinatesMinusOne = self.gridEntity.returnShiftedCoordinate(currentIndexes,dict.fromkeys(self.axisName,-1))
      for i in range(len(self.axisName)):
        varName = self.axisName[i]
        for key in varName.strip().split(','):
          self.inputInfo['distributionName'][key] = self.toBeSampled[varName]
          self.inputInfo['distributionType'][key] = self.distDict[varName].type
          self.values[key] = coordinates[varName]
          if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim']==1): self.inputInfo['SampledVarsPb'][key] = self.distDict[varName].pdf(self.values[key])
          else:
            # N-Dimensional pdf
            dist_name = self.variables2distributionsMapping[varName]['name']
            NDcoordinate=[0]*len(self.distributions2variablesMapping[dist_name])
            for var in self.distributions2variablesMapping[dist_name]:
              variable = var.keys()[0]
              position = var.values()[0]
              NDcoordinate[position-1] = float(coordinates[variable.strip()])
            self.inputInfo['SampledVarsPb'][key] = self.distDict[varName].pdf(NDcoordinate)
        if ("<distribution>" in varName) or (self.variables2distributionsMapping[varName]['totDim']==1):
          if self.distDict[varName].getDisttype() == 'Discrete':
            weight *= self.distDict[varName].pdf(coordinates[varName])
          else:
            if self.gridInfo[varName]=='CDF':
              if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(coordinatesPlusOne[varName]))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(coordinatesMinusOne[varName]))/2.0)
              if coordinatesMinusOne[varName] == -sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(coordinatesPlusOne[varName]))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(0))/2.0)
              if coordinatesPlusOne[varName] == sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(1))/2.0) - self.distDict[varName].cdf((self.values[key]+self.distDict[varName].ppf(coordinatesMinusOne[varName]))/2.0)
            else:   # Value
              if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+coordinatesPlusOne[varName])/2.0) -self.distDict[varName].cdf((self.values[key]+coordinatesMinusOne[varName])/2.0)
              if coordinatesMinusOne[varName] == -sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+coordinatesPlusOne[varName])/2.0) -self.distDict[varName].cdf((self.values[key]+self.distDict[varName].lowerBound)/2.0)
              if coordinatesPlusOne[varName] == sys.maxsize:
                weight *= self.distDict[varName].cdf((self.values[key]+self.distDict[varName].upperBound)/2.0) -self.distDict[varName].cdf((self.values[key]+coordinatesMinusOne[varName])/2.0)
        # ND variable
        else:
          if self.variables2distributionsMapping[varName]['dim']==1:    # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
            dist_name = self.variables2distributionsMapping[varName]['name']
            NDcoordinate=np.zeros(len(self.distributions2variablesMapping[dist_name]))
            dxs=np.zeros(len(self.distributions2variablesMapping[dist_name]))
            for var in self.distributions2variablesMapping[dist_name]:
              variable = var.keys()[0]
              position = var.values()[0]
              NDcoordinate[position-1] = coordinates[variable.strip()]
              if self.gridInfo[varName]=='CDF':
                if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
                  dxs[position-1] = self.distDict[varName].inverseMarginalDistribution((coordinatesPlusOne[varName] - coordinatesMinusOne[varName]) / 2.0,self.variables2distributionsMapping[varName]['dim']-1)
                if coordinatesMinusOne[varName] == -sys.maxsize:
                  dxs[position-1] = self.distDict[varName].inverseMarginalDistribution(coordinatesPlusOne[varName],self.variables2distributionsMapping[varName]['dim']-1) - coordinates[variable.strip()]
                if coordinatesPlusOne[varName] == sys.maxsize:
                  dxs[position-1] = coordinates[variable.strip()] - self.distDict[varName].inverseMarginalDistribution(coordinatesMinusOne[varName],self.variables2distributionsMapping[varName]['dim']-1)
              else:
                if coordinatesPlusOne[varName] != sys.maxsize and coordinatesMinusOne[varName] != -sys.maxsize:
                  dxs[position-1] = (coordinatesPlusOne[varName] - coordinatesMinusOne[varName]) / 2.0
                if coordinatesMinusOne[varName] == -sys.maxsize:
                  dxs[position-1] = coordinatesPlusOne[varName] - coordinates[variable.strip()]
                if coordinatesPlusOne[varName] == sys.maxsize:
                  dxs[position-1] = coordinates[variable.strip()] - coordinatesMinusOne[varName]
            weight *= self.distDict[varName].cellIntegral(NDcoordinate,dxs)
      newpoint = tuple(self.values[key] for key in self.values.keys())
      if newpoint not in self.existing:
        found=True
        self.raiseADebug('New point found: '+str(newpoint))
      else:
        self.counter+=1
        if self.counter>=self.limit: raise utils.NoMoreSamplesNeeded
        self.raiseADebug('Existing point: '+str(newpoint))
      self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
      self.inputInfo['ProbabilityWeight'] = copy.deepcopy(weight)
      self.inputInfo['SamplerType'] = 'Grid'
#
#
#
#
class Stratified(Grid):
  """
  Stratified sampler, also known as Latin Hypercube Sampling (LHS). Currently no
  special filling methods are implemented
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Grid.__init__(self)
    self.sampledCoordinate    = [] # a list of list for i=0,..,limit a list of the coordinate to be used this is needed for the LHS
    self.printTag = 'SAMPLER Stratified'
    self.globalGrid          = {}    # Dictionary for the global_grid. These grids are used only for Stratified for ND distributions.

  def localInputAndChecks(self,xmlNode):
    """
    Class specific xml inputs will be read here and checked for validity.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    @ Out, None
    """
    Sampler.read_sampler_init(self,xmlNode)
    Grid.localInputAndChecks(self,xmlNode)
    pointByVar  = [len(self.gridEntity.returnParameter("gridInfo")[variable][2]) for variable in self.gridInfo.keys()]
    if len(set(pointByVar))!=1: self.raiseAnError(IOError,'the latin Hyper Cube requires the same number of point in each dimension')
    self.pointByVar         = pointByVar[0]
    self.inputInfo['upper'] = {}
    self.inputInfo['lower'] = {}

  def localInitialize(self):
    """
    the local initialize is used to generate test the box being within the
    distribution upper/lower bound and filling mapping of the hyper cube.
    @ In, None
    @ Out, None
    """
    Grid.localInitialize(self)
    self.limit = (self.pointByVar-1)
    globGridsCount = {}
    dimInfo = self.gridEntity.returnParameter("dimInfo")
    for val in dimInfo.values():
      if val[-1] != None and val[-1] not in globGridsCount.keys(): globGridsCount[val[-1]] = 0
      globGridsCount[val[-1]] += 1
    diff = -sum(globGridsCount.values())+len(globGridsCount.keys())
    tempFillingCheck = [[None]*(self.pointByVar-1)]*(len(self.gridEntity.returnParameter("dimensionNames"))+diff) #for all variables

    self.sampledCoordinate = [[None]*len(self.axisName)]*(self.pointByVar-1)
    for i in range(len(tempFillingCheck)): tempFillingCheck[i]  = Distributions.randomPermutation(list(range(self.pointByVar-1)),self) #pick a random interval sequence
    cnt = 0
    mappingIdVarName = {}
    for varName in self.axisName:
      if varName not in dimInfo.keys(): mappingIdVarName[varName] = cnt
      else:
        for addKey,value in dimInfo.items():
          if value[1] == dimInfo[varName][1] and addKey not in mappingIdVarName.keys(): mappingIdVarName[addKey] = cnt
      if len(mappingIdVarName.keys()) == len(self.axisName): break
      cnt +=1
    for nPoint in range(self.pointByVar-1): self.sampledCoordinate[nPoint]= [tempFillingCheck[mappingIdVarName[varName]][nPoint] for varName in self.axisName]
    if self.restartData:
      self.counter+=len(self.restartData)
      self.raiseAMessage('Number of points from restart: %i' %self.counter)
      self.raiseAMessage('Number of points needed:       %i' %(self.limit-self.counter))

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs (unused)
      @ Out, None
    """
    varCount = 0
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    weight = 1.0
    for varName in self.axisName:
      upper = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]+1})[varName]
      lower = self.gridEntity.returnShiftedCoordinate(self.gridEntity.returnIteratorIndexes(),{varName:self.sampledCoordinate[self.counter-1][varCount]})[varName]
      coordinate = lower + (upper-lower)*Distributions.random()
      varCount += 1
      if not "<distribution>" in varName:
        if self.variables2distributionsMapping[varName]['totDim']>1 and self.variables2distributionsMapping[varName]['dim'] == 1:    # to avoid double count of weight for ND distribution; I need to count only one variable instaed of N
          gridCoordinate, distName =  self.distDict[varName].ppf(coordinate), self.variables2distributionsMapping[varName]['name']
          for distVarName in self.distributions2variablesMapping[distName]:
            for kkey in distVarName.keys()[0].strip().split(','):
              self.inputInfo['distributionName'][kkey], self.inputInfo['distributionType'][kkey], self.values[kkey] = self.toBeSampled[varName], self.distDict[varName].type, np.atleast_1d(gridCoordinate)[distVarName.values()[0]-1]
              self.inputInfo['SampledVarsPb'][varName] = coordinate
          weight *= upper - lower
      if ("<distribution>" in varName) or self.variables2distributionsMapping[varName]['totDim']==1:   # 1D variable
        # if the varName is a comma separated list of strings the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables
        if self.gridInfo[varName] =='CDF':
          ppfvalue, ppflower, ppfupper = self.distDict[varName].ppf(coordinate), self.distDict[varName].ppf(min(upper,lower)), self.distDict[varName].ppf(max(upper,lower))
        for kkey in varName.strip().split(','):
          self.inputInfo['distributionName'][kkey] = self.toBeSampled[varName]
          self.inputInfo['distributionType'][kkey] = self.distDict[varName].type
          if self.gridInfo[varName] =='CDF':
            self.values[kkey], self.inputInfo['upper'][kkey], self.inputInfo['lower'][kkey], self.inputInfo['SampledVarsPb'][varName]  = ppfvalue, ppfupper, ppflower, coordinate
            weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
          elif self.gridInfo[varName] =='value':
            self.values[varName], self.inputInfo['upper'][kkey], self.inputInfo['lower'][kkey] = coordinate, max(upper,lower), min(upper,lower)
            self.inputInfo['SampledVarsPb'][kkey] = self.distDict[varName].pdf(self.values[kkey])
        if self.gridInfo[varName] =='CDF': weight *= self.distDict[varName].cdf(ppfupper) - self.distDict[varName].cdf(ppflower)
        else                             : weight *= self.distDict[varName].cdf(upper) - self.distDict[varName].cdf(lower)
    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = weight
    self.inputInfo['SamplerType'] = 'Stratified'
#
#
#
#
class DynamicEventTree(Grid):
  """
  DYNAMIC EVENT TREE Sampler (DET)
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
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
    self.preconditionerAvail['Stratified'] = Stratified      # Stratified
    self.preconditionerAvail['Grid'      ] = Grid            # Grid
    # dictionary of inputted preconditioners need to be applied
    self.preconditionerToApply             = {}
    # total number of preconditioner samples (combination of all different preconditioner strategy)
    self.precNumberSamplers                = 0
    self.printTag = 'SAMPLER DYNAMIC ET'

  def _localWhatDoINeed(self):
    """
    This method is a local mirror of the general whatDoINeed method.
    It is implmented here because this Sampler requests special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    """
    needDict = Sampler._localWhatDoINeed(self)
    for preconditioner in self.preconditionerToApply.values():
      preneedDict = preconditioner.whatDoINeed()
      for key,value in preneedDict.items():
        if key not in needDict.keys(): needDict[key] = []
        needDict[key] = needDict[key] + value
    return needDict

  def localStillReady(self, ready): #,lastOutput=None
    """
    Function that inquires if there is at least an input the in the queue that
    needs to be run
    @ InOut, ready, boolean specifying whether the sampler is ready
    """
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0): ready = True
    else:
      if self.print_end_xml:
        myFile = open(os.path.join(self.workingDir,self.name + "_output_summary.xml"),'w')
        for treeNode in self.TreeInfo.values(): treeNode.writeNodeTree(myFile)
        myFile.close()
      ready = False
    return ready

  def _retrieveParentNode(self,idj):
    """
    Grants access to the parent node of a particular job
    @ In, idj, the identifier of a job object
    @ Out, the parent node of the job linked to idj
    """
    if(idj == self.TreeInfo[self.rootToJob[idj]].getrootnode().name): parentNode = self.TreeInfo[self.rootToJob[idj]].getrootnode()
    else: parentNode = list(self.TreeInfo[self.rootToJob[idj]].getrootnode().iter(idj))[0]
    return parentNode

  def localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=True):
    """
    General function (available to all samplers) that finalize the sampling
    calculation just ended. In this case (DET), The function reads the
    information from the ended calculation, updates the working variables, and
    creates the new inputs for the next branches
    @ In, jobObject: JobHandler Instance of the job (run) just finished
    @ In, model        : Model Instance... It may be a Code Instance, ROM, etc.
    @ In, myInput      : List of the original input files
    @ In, genRunQueue  : bool, generated Running queue at the end of the
                         finalization?
    @ Out, None
    """
    self.workingDir = model.workingDir

    # returnBranchInfo = self.__readBranchInfo(jobObject.output)
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
    # # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    # if(jobObject.identifier == self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().name): endInfo['parent_node'] = self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode()
    # else: endInfo['parent_node'] = list(self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().iter(jobObject.identifier))[0]
    endInfo['parent_node'] = parentNode
    # get the branchedLevel dictionary
    branchedLevel = {}
    for distk, distpb in zip(endInfo['parent_node'].get('initiator_distribution'),endInfo['parent_node'].get('PbThreshold')): branchedLevel[distk] = utils.index(self.branchProbabilities[distk],distpb)
    if not branchedLevel: self.raiseAnError(RuntimeError,'branchedLevel of node '+jobObject.identifier+'not found!')
    # Loop of the parameters that have been changed after a trigger gets activated
    for key in endInfo['branch_changed_params']:
      endInfo['n_branches'] = 1 + int(len(endInfo['branch_changed_params'][key]['actual_value']))
      if(len(endInfo['branch_changed_params'][key]['actual_value']) > 1):
        #  Multi-Branch mode => the resulting branches from this parent calculation (just ended)
        # will be more then 2
        # unchanged_pb = probability (not conditional probability yet) that the event does not occur
        unchanged_pb = 0.0
        try:
          # changed_pb = probability (not conditional probability yet) that the event A occurs and the final state is 'alpha' """
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
    # # set runEnded and running to true and false respectively
    # endInfo['parent_node'].add('runEnded',True)
    # endInfo['parent_node'].add('running',False)
    # endInfo['parent_node'].add('end_time',self.actual_end_time)
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
    """
    Function to compute Conditional probability of the branches that are going to be run.
    The conditional probabilities are stored in the self.endInfo object
    @ In, index: position in the self.endInfo list (optional). Default = 0
    @ Out, None
    """
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
    """
    Function to read the Branching info that comes from a Model
    The branching info (for example, distribution that triggered, parameters must be changed, etc)
    are supposed to be in a xml format
    @ In, out_base: is the output root that, if present, is used to construct the file name the function is going
                    to try reading.
    @ Out, boolean: true if the info are present (a set of new branches need to be run), false if the actual parent calculation reached an end point
    """
    # Remove all the elements from the info container
    del self.actualBranchInfo
    branch_present = False
    self.actualBranchInfo = {}
    # Construct the file name adding the out_base root if present
    if out_base: filename = out_base + "_actual_branch_info.xml"
    else: filename = "actual_branch_info.xml"
    if not os.path.isabs(filename): filename = os.path.join(self.workingDir,filename)
    if not os.path.exists(filename):
      self.raiseADebug('branch info file ' + os.path.basename(filename) +' has not been found. => No Branching.')
      return branch_present
    # Parse the file and create the xml element tree object
    #try:
    branch_info_tree = ET.parse(filename)
    self.raiseADebug('Done parsing '+filename)
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
    """
    Method to generate the running internal queue for one point in the epistemic
    space. It generates the initial information to instantiate the root of a
    Deterministic Dynamic Event Tree.
    @ In, rootTree, TreeStructure object, the rootTree of the single coordinate in
          the epistemic space.
    @ In, branchedLevel, dict, dictionary of the levels reached by the rootTree
          mapped in the internal grid dictionary (self.branchProbabilities)
    @ In, model, Models object, the model that is used to explore the input space
          (e.g. a code, like RELAP-7)
    @ In, myInput, list, list of inputs for the Models object (passed through the
          Steps XML block)
    @ Out, None
    """
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
    """
    Method to generate the running internal queue for all the points in
    the epistemic space. It generates the initial information to
    instantiate the roots of all the N-D coordinates to construct multiple
    Deterministic Dynamic Event Trees.
    @ In, model, Models object, the model that is used to explore the input
          space (e.g. a code, like RELAP-7)
    @ In, myInput, list, list of inputs for the Models object (passed through
          the Steps XML block)
    @ Out, None
    """
    # We construct the input for the first DET branch calculation'
    # Increase the counter
    # The root name of the xml element tree is the starting name for all the branches
    # (this root name = the user defined sampler name)
    # Get the initial branchedLevel dictionary (=> the list gets empty)
    branchedLevel = self.branchedLevel.pop(0)
    for rootTree in self.TreeInfo.values(): self._createRunningQueueBeginOne(rootTree,branchedLevel, model,myInput)
    return

  def _createRunningQueueBranch(self,model,myInput):
    """ Method to generate the running internal queue right after a branch occurred
    It generates the the information to insatiate the branches' continuation of the Deterministic Dynamic Event Tree
    @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
    @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
    """
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
      self.raiseADebug('Branch ' + endInfo['parent_node'].get('name') + ' hit last Threshold for distribution ' + endInfo['branch_dist'])
      self.raiseADebug('Branch ' + endInfo['parent_node'].get('name') + ' is dead end.')
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
      #  subGroup.set('restartFileRoot',endInfo['restartRoot'])
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
    """
    Function to create and append new inputs to the queue. It uses all the containers have been updated by the previous functions
    @ In, model  : Model instance. It can be a Code type, ROM, etc.
    @ In, myInput: List of the original inputs
    @ Out, None
    """
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
    """
    Function to get an input from the internal queue system
    @ In, None
    @ Out, jobInput: First input in the queue
    """
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
    """
    This method needs to be overwritten by the Dynamic Event Tree Sampler, since the input creation strategy is completely different with the respect the other samplers
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    @return     : [] containing the new inputs -in reality it is the model that returns this, the Sampler generates the values to be placed in the model input
    """
    return self.localGenerateInput(model, oldInput)

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs (unused)
      @ Out, None
    """
    if self.counter <= 1:
      # If first branch input, create the queue
      self._createRunningQueue(model, myInput)
    # retrieve the input from the queue
    newerinput = self.__getQueueElement()
    if not newerinput:
      # If no inputs are present in the queue => a branch is finished
      self.raiseADebug('A Branch ended!')
    return newerinput

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distrbutions and functions.
      @ In, availDist, dict of distributions
      @ In, availDist, dict of functions
      @Out, None
    """
    Grid._generateDistributions(self,availableDist,availableFunc)
    for preconditioner in self.preconditionerToApply.values(): preconditioner._generateDistributions(availableDist,availableFunc)

  def localInputAndChecks(self,xmlNode):
    """
    Class specific inputs will be read here and checked for validity.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    """
    Grid.localInputAndChecks(self,xmlNode)
    if 'print_end_xml' in xmlNode.attrib.keys():
      if xmlNode.attrib['print_end_xml'].lower() in utils.stringsThatMeanTrue(): self.print_end_xml = True
      else: self.print_end_xml = False
    if 'maxSimulationTime' in xmlNode.attrib.keys():
      try:    self.maxSimulTime = float(xmlNode.attrib['maxSimulationTime'])
      except (KeyError,NameError): self.raiseAnError(IOError,'Can not convert maxSimulationTime in float number!!!')
    for child in xmlNode:
      if child.tag == 'PreconditionerSampler':
        if not 'type' in child.attrib.keys()                          : self.raiseAnError(IOError,'Not found attribute type in PreconditionerSampler block!')
        if child.attrib['type'] in self.preconditionerToApply.keys()  : self.raiseAnError(IOError,'PreconditionerSampler type '+child.attrib['type'] + ' already inputted!')
        if child.attrib['type'] not in self.preconditionerAvail.keys(): self.raiseAnError(IOError,'PreconditionerSampler type' +child.attrib['type'] + 'unknown. Available are '+ str(self.preconditionerAvail.keys()).replace("[","").replace("]",""))
        self.precNumberSamplers = 1
        # the user can decided how to preconditionate
        self.preconditionerToApply[child.attrib['type']] = self.preconditionerAvail[child.attrib['type']]()
        # give the preconditioner sampler the message handler
        self.preconditionerToApply[child.attrib['type']].setMessageHandler(self.messageHandler)
        # make the preconditioner sampler read  its own xml block
        self.preconditionerToApply[child.attrib['type']]._readMoreXML(child)
    branchedLevel = {}
    error_found = False
    gridInfo = self.gridEntity.returnParameter("gridInfo")
    for keyk in self.axisName:
      branchedLevel[self.toBeSampled[keyk]] = 0
      if self.gridInfo[keyk] == 'CDF':
        self.branchProbabilities[self.toBeSampled[keyk]] = gridInfo[keyk][2]
        self.branchProbabilities[self.toBeSampled[keyk]].sort(key=float)
        if max(self.branchProbabilities[self.toBeSampled[keyk]]) > 1:
          self.raiseAWarning("One of the Thresholds for distribution " + str(gridInfo[keyk][2]) + " is > 1")
          error_found = True
          for index in range(len(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float))):
            if sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float).count(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index]) > 1:
              self.raiseAWarning("In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
              error_found = True
      else:
        self.branchValues[self.toBeSampled[keyk]] = gridInfo[keyk][2]
        self.branchValues[self.toBeSampled[keyk]].sort(key=float)
        for index in range(len(sorted(self.branchValues[self.toBeSampled[keyk]], key=float))):
          if sorted(self.branchValues[self.toBeSampled[keyk]], key=float).count(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index]) > 1:
            self.raiseAWarning("In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
            error_found = True
    if error_found: self.raiseAnError(IOError,"In sampler named " + self.name+' Errors have been found!' )
    # Append the branchedLevel dictionary in the proper list
    self.branchedLevel.append(branchedLevel)

  def localAddInitParams(self,tempDict):
    """
    Appends a given dictionary with class specific member variables and their
    associated initialized values.
    @ InOut, tempDict: The dictionary where we will add the initialization
                       parameters specific to this Sampler.
    """
    for key in self.branchProbabilities.keys(): tempDict['Probability Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchProbabilities[key]]
    for key in self.branchValues.keys()       : tempDict['Values Thresholds for dist ' + str(key) + ' are: '] = [str(x) for x in self.branchValues[key]]

  def localAddCurrentSetting(self,tempDict):
    """
    Appends a given dictionary with class specific information regarding the
    current status of the object.
    @ InOut, tempDict: The dictionary where we will add the parameters specific
                       to this Sampler and their associated values.
    """
    tempDict['actual threshold levels are '] = self.branchedLevel[0]

  def localInitialize(self):
    """
    Will perform all initialization specific to this Sampler. This will be
    called at the beginning of each Step where this object is used. See base
    class for more details.
    @ In None
    @ Out None
    """
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
      self.raiseADebug('Number of Preconditioner Samples are ' + str(self.precNumberSamplers) + '!')
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
    self.limit = sys.maxsize
#
#
#
#
class AdaptiveDET(DynamicEventTree, LimitSurfaceSearch):
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    DynamicEventTree.__init__(self)  # init DET
    LimitSurfaceSearch.__init__(self)   # init Adaptive
    self.detAdaptMode         = 1    # Adaptive Dynamic Event Tree method (=1 -> DynamicEventTree as preconditioner and subsequent LimitSurfaceSearch,=2 -> DynamicEventTree online adaptive)
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
    """
    This method is a local mirror of the general whatDoINeed method.
    It is implmented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
<<<<<<< Temporary merge branch 1
    '''
    adaptNeed = LimitSurfaceSearch._localWhatDoINeed(self)
=======
    """
    adaptNeed = LimitSurfaceSearch._localWhatDoINeed(self)
    DETNeed   = DynamicEventTree._localWhatDoINeed(self)
    return dict(adaptNeed.items()+ DETNeed.items())

  def _checkIfStartAdaptive(self):
    """
    Function that checks if the adaptive needs to be started (mode 1)
    @ In, None
    @ Out, None
    """
    if not self.startAdaptive:
      self.startAdaptive = True
      for treer in self.TreeInfo.values():
        for _ in treer.iterProvidedFunction(self._checkIfRunnint):
          self.startAdaptive = False
          break
        if not self.startAdaptive: break

  def _checkClosestBranch(self):
    """
    Function that checks the closest branch already evaluated
    @ In, None
    @ Out, dict, key:gridPosition
    """
    # compute cdf of sampled vars
    lowerCdfValues = {}
    cdfValues         = {}
    for key,value in self.values.items():
      cdfValues[key] = self.distDict[key].cdf(value)
      lowerCdfValues[key] = utils.find_le(self.branchProbabilities[self.toBeSampled[key]],cdfValues[key])[0]
      self.raiseADebug(str(self.toBeSampled[key]))
      self.raiseADebug(str(value))
      self.raiseADebug(str(cdfValues[key]))
      self.raiseADebug(str(lowerCdfValues[key]))
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
    """
     Function that retrieves the key information from a branch to start a newer calculation
     @ In, branch
     @ Out, dictionary with those information
    """
    info = branch.getValues()
    info['actualBranchOnLevel'] = branch.numberBranches()
    info['parent_node']         = branch
    return info

  def _constructEndInfoFromBranch(self,model, myInput, info, cdfValues):
    """
    @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
    @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
    @ In, info, dict, dictionary of information at the end of a branch (information collected by the method _retrieveBranchInfo)
    @ In, cdfValues, dict, dictionary of CDF thresholds reached by the branch that just ended.
    """
    endInfo = info['parent_node'].get('endInfo')
    #branchedLevel = {}
    #for distk, distpb in zip(info['initiator_distribution'],info['PbThreshold']): branchedLevel[distk] = index(self.branchProbabilities[distk],distpb)
    del self.inputInfo
    self.counter           += 1
    self.branchCountOnLevel = info['actualBranchOnLevel']+1
    # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
    rname = info['parent_node'].get('name') + '-' + str(self.branchCountOnLevel)
    info['parent_node'].add('completedHistory', False)
    self.raiseADebug(str(rname))
    bcnt = self.branchCountOnLevel
    while info['parent_node'].isAnActualBranch(rname):
      bcnt += 1
      rname = info['parent_node'].get('name') + '-' + str(bcnt)
    # create a subgroup that will be appended to the parent element in the xml tree structure
    subGroup = ETS.Node(rname)
    subGroup.add('parent', info['parent_node'].get('name'))
    subGroup.add('name', rname)
    self.raiseADebug('cond pb = '+str(info['parent_node'].get('conditional_pb')))
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
    # check if it is a preconditioned DET sampling, if so add the relative information
    # precSampled = endInfo['parent_node'].get('preconditionerSampled')
    # if precSampled:
    #   self.inputInfo['preconditionerCoordinate'] = copy.deepcopy(precSampled)
    #   subGroup.add('preconditionerSampled', precSampled)
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
    """
    Function that inquires if there is at least an input the in the queue that needs to be run
    @ InOut, ready, boolean
    @ Out, boolean
    """
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
        else: self.raiseAWarning('No Completed HistorySet! Not possible to start an adaptive search! Something went wrong!')
      if len(completedHistNames) > self.completedHistCnt:
        self.actualLastOutput = self.lastOutput
        self.lastOutput       = self.actualLastOutput
        ready = LimitSurfaceSearch.localStillReady(self,ready)
        self.completedHistCnt = len(completedHistNames)
      else: ready = False
      self.adaptiveReady = ready
      if ready or detReady and self.persistence > self.repetition : return True
      else: return False
    return detReady

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs
      @ Out, None
    """
    if self.startAdaptive:
      LimitSurfaceSearch.localGenerateInput(self,model,myInput)
      #the adaptive sampler created the next point sampled vars
      #find the closest branch
      closestBranch, cdfValues = self._checkClosestBranch()
      if closestBranch == None: self.raiseADebug('An usable branch for next candidate has not been found => create a parallel branch!')
      # add pbthresholds in the grid
      investigatedPoint = {}
      for key,value in cdfValues.items():
        # if self.insertAdaptBPb:
        #   ind = utils.find_le_index(self.branchProbabilities[self.toBeSampled[key]],value)
        #   if not ind: ind = 0
        #   if value not in self.branchProbabilities[self.toBeSampled[key]]:
        #     self.branchProbabilities[self.toBeSampled[key]].insert(ind,value)
        #     self.branchValues[self.toBeSampled[key]].insert(ind,self.distDict[key].ppf(value))

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
    """
    Class specific inputs will be read here and checked for validity.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    """
    DynamicEventTree.localInputAndChecks(self,xmlNode)
    LimitSurfaceSearch.localInputAndChecks(self,xmlNode)
    if 'mode' in xmlNode.attrib.keys():
      if xmlNode.attrib['mode'].lower() == 'online': self.detAdaptMode = 2
      elif xmlNode.attrib['mode'].lower() == 'post': self.detAdaptMode = 1
      else:  self.raiseAnError(IOError,'unknown mode '+xmlNode.attrib['mode']+'. Available are "online" and "post"!')
    if 'noTransitionStrategy' in xmlNode.attrib.keys():
      if xmlNode.attrib['noTransitionStrategy'].lower() == 'mc'    : self.noTransitionStrategy = 1
      elif xmlNode.attrib['noTransitionStrategy'].lower() == 'grid': self.noTransitionStrategy = 2
      else:  self.raiseAnError(IOError,'unknown noTransitionStrategy '+xmlNode.attrib['noTransitionStrategy']+'. Available are "mc" and "grid"!')
    if 'updateGrid' in xmlNode.attrib.keys():
      if xmlNode.attrib['updateGrid'].lower() in utils.stringsThatMeanTrue(): self.insertAdaptBPb = True

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distrbutions and functions.
      @ In, availDist, dict of distributions
      @ In, availDist, dict of functions
      @Out, None
    """
    DynamicEventTree._generateDistributions(self,availableDist,availableFunc)

  def localInitialize(self,solutionExport = None):
    """
    Will perform all initialization specific to this Sampler. This will be
    called at the beginning of each Step where this object is used. See base
    class for more details.
    @ InOut, solutionExport: a PointSet to hold the solution
    @ Out None
    """
    if self.detAdaptMode == 2: self.startAdaptive = True
    DynamicEventTree.localInitialize(self)
    LimitSurfaceSearch.localInitialize(self,solutionExport=solutionExport)
    self._endJobRunnable    = sys.maxsize

  def generateInput(self,model,oldInput):
    """
    Will generate an input
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g.
                     list of files, etc. etc)
    @return     : [] containing the new inputs -in reality it is the model that
                     returns this, the Sampler generates the values to be placed
                     in the model input
    """
    return DynamicEventTree.generateInput(self, model, oldInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
    General function (available to all samplers) that finalizes the sampling
    calculation just ended. See base class for more information.
    @ In, jobObject    : JobHandler Instance of the job (run) just finished
    @ In, model        : Model Instance... It may be a Code Instance, ROM, etc.
    @ In, myInput      : List of the original input files
    @ Out, None
    """
    returncode = DynamicEventTree.localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=False)
    if returncode:
      self._createRunningQueue(model,myInput)
#
#
#
#
class FactorialDesign(Grid):
  """
  Samples the model on a given (by input) set of points
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Grid.__init__(self)
    self.printTag = 'SAMPLER FACTORIAL DESIGN'
    # accepted types. full = full factorial, 2levelfract = 2-level fracional factorial, pb = Plackett-Burman design. NB. full factorial is equivalent to Grid sampling
    self.acceptedTypes = ['full','2levelfract','pb'] # accepted factorial types
    self.factOpt       = {}                          # factorial options (type,etc)
    self.designMatrix  = None                        # matrix container

  def localInputAndChecks(self,xmlNode):
    """
    Class specific xml inputs will be read here and checked for validity.
    Specifically, reading and construction of the grid.
    @ In, xmlNode: The xml element node that will be checked against the
                   available options specific to this Sampler.
    @ Out, None
    """
    Grid.localInputAndChecks(self,xmlNode)
    factsettings = xmlNode.find("FactorialSettings")
    if factsettings == None: self.raiseAnError(IOError,'FactorialSettings xml node not found!')
    facttype = factsettings.find("algorithm_type")
    if facttype == None: self.raiseAnError(IOError,'node "algorithm_type" not found in FactorialSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedTypes:self.raiseAnError(IOError,' "type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedTypes))
    self.factOpt['algorithm_type'] = facttype.text.lower()
    if self.factOpt['algorithm_type'] == '2levelfract':
      self.factOpt['options'] = {}
      self.factOpt['options']['gen'] = factsettings.find("gen")
      self.factOpt['options']['genMap'] = factsettings.find("genMap")
      if self.factOpt['options']['gen'] == None: self.raiseAnError(IOError,'node "gen" not found in FactorialSettings xml node!!!')
      if self.factOpt['options']['genMap'] == None: self.raiseAnError(IOError,'node "genMap" not found in FactorialSettings xml node!!!')
      self.factOpt['options']['gen'] = self.factOpt['options']['gen'].text.split(',')
      self.factOpt['options']['genMap'] = self.factOpt['options']['genMap'].text.split(',')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): self.raiseAnError(IOError,'number of variable in genMap != number of variables !!!')
      if len(self.factOpt['options']['gen']) != len(self.gridInfo.keys())   : self.raiseAnError(IOError,'number of variable in gen != number of variables !!!')
      rightOrder = [None]*len(self.gridInfo.keys())
      if len(self.factOpt['options']['genMap']) != len(self.factOpt['options']['gen']): self.raiseAnError(IOError,'gen and genMap different size!')
      if len(self.factOpt['options']['genMap']) != len(self.gridInfo.keys()): self.raiseAnError(IOError,'number of gen attributes and variables different!')
      for ii,var in enumerate(self.factOpt['options']['genMap']):
        if var not in self.gridInfo.keys(): self.raiseAnError(IOError,' variable "'+var+'" defined in genMap block not among the inputted variables!')
        rightOrder[self.axisName.index(var)] = self.factOpt['options']['gen'][ii]
      self.factOpt['options']['orderedGen'] = rightOrder
    if self.factOpt['algorithm_type'] != 'full':
      self.externalgGridCoord = True
      for varname in self.gridInfo.keys():
        if len(self.gridEntity.returnParameter("gridInfo")[varname][2]) != 2:
          self.raiseAnError(IOError,'The number of levels for type '+
                        self.factOpt['algorithm_type'] +' must be 2! In variable '+varname+ ' got number of levels = ' +
                        str(len(self.gridEntity.returnParameter("gridInfo")[varname][2])))
    else: self.externalgGridCoord = False

  def localAddInitParams(self,tempDict):
    """
    Appends a given dictionary with class specific member variables and their
    associated initialized values.
    @ InOut, tempDict: The dictionary where we will add the initialization
                       parameters specific to this Sampler.
    """
    Grid.localAddInitParams(self,tempDict)
    for key,value in self.factOpt.items():
      if key != 'options': tempDict['Factorial '+key] = value
      else:
        for kk,val in value.items(): tempDict['Factorial options '+kk] = val

  def localInitialize(self):
    """
    This method initialize the factorial matrix. No actions are taken for full-factorial since it is equivalent to the Grid sampling this sampler is based on
    """
    Grid.localInitialize(self)
    if   self.factOpt['algorithm_type'] == '2levelfract': self.designMatrix = doe.fracfact(' '.join(self.factOpt['options']['orderedGen'])).astype(int)
    elif self.factOpt['algorithm_type'] == 'pb'         : self.designMatrix = doe.pbdesign(len(self.gridInfo.keys())).astype(int)
    if self.designMatrix != None:
      self.designMatrix[self.designMatrix == -1] = 0 # convert all -1 in 0 => we can access to the grid info directly
      self.limit = self.designMatrix.shape[0]        # the limit is the number of rows

  def localGenerateInput(self,model,myInput):
    """
    Will generate an input and associate it with a probability
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs (unused)
      @ Out, None
    """
    if self.factOpt['algorithm_type'] == 'full':  Grid.localGenerateInput(self,model, myInput)
    else:
      self.gridCoordinate = self.designMatrix[self.counter - 1][:].tolist()
      Grid.localGenerateInput(self,model, myInput)
#
#
#
#
class ResponseSurfaceDesign(Grid):
  """
  Samples the model on a given (by input) set of points
  """
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Grid.__init__(self)
    self.limit    = 1
    self.printTag = 'SAMPLER RESPONSE SURF DESIGN'
    self.respOpt         = {}                                    # response surface design options (type,etc)
    self.designMatrix    = None                                  # matrix container
    self.bounds          = {}                                    # dictionary of lower and upper
    self.mapping         = {}                                    # mapping between designmatrix coordinates and position in grid
    self.minNumbVars     = {'boxbehnken':3,'centralcomposite':2} # minimum number of variables
    # dictionary of accepted types and options (required True, optional False)
    self.acceptedOptions = {'boxbehnken':['ncenters'], 'centralcomposite':['centers','alpha','face']}

  def localInputAndChecks(self,xmlNode):
    """reading and construction of the grid"""
    Grid.localInputAndChecks(self,xmlNode)
    factsettings = xmlNode.find("ResponseSurfaceDesignSettings")
    if factsettings == None: self.raiseAnError(IOError,'ResponseSurfaceDesignSettings xml node not found!')
    facttype = factsettings.find("algorithm_type")
    if facttype == None: self.raiseAnError(IOError,'node "algorithm_type" not found in ResponseSurfaceDesignSettings xml node!!!')
    elif not facttype.text.lower() in self.acceptedOptions.keys():self.raiseAnError(IOError,'"type" '+facttype.text+' unknown! Available are ' + ' '.join(self.acceptedOptions.keys()))
    self.respOpt['algorithm_type'] = facttype.text.lower()
    # set defaults
    if self.respOpt['algorithm_type'] == 'boxbehnken': self.respOpt['options'] = {'ncenters':None}
    else                                             : self.respOpt['options'] = {'centers':(4,4),'alpha':'orthogonal','face':'circumscribed'}
    for child in factsettings:
      if child.tag not in 'algorithm_type': self.respOpt['options'][child.tag] = child.text.lower()
    # start checking
    for key,value in self.respOpt['options'].items():
      if key not in self.acceptedOptions[facttype.text.lower()]: self.raiseAnError(IOError,'node '+key+' unknown. Available are "'+' '.join(self.acceptedOptions[facttype.text.lower()])+'"!!')
      if self.respOpt['algorithm_type'] == 'boxbehnken':
        if key == 'ncenters':
          if self.respOpt['options'][key] != None:
            try   : self.respOpt['options'][key] = int(value)
            except: self.raiseAnError(IOError,'"'+key+'" is not an integer!')
      else:
        if key == 'centers':
          if len(value.split(',')) != 2: self.raiseAnError(IOError,'"'+key+'" must be a comma separated string of 2 values only!')
          try: self.respOpt['options'][key] = (int(value.split(',')[0]),int(value.split(',')[1]))
          except: self.raiseAnError(IOError,'"'+key+'" values must be integers!!')
        if key == 'alpha':
          if value not in ['orthogonal','rotatable']: self.raiseAnError(IOError,'Not recognized options for node ' +'"'+key+'". Available are "orthogonal","rotatable"!')
        if key == 'face':
          if value not in ['circumscribed','faced','inscribed']: self.raiseAnError(IOError,'Not recognized options for node ' +'"'+key+'". Available are "circumscribed","faced","inscribed"!')
    gridInfo = self.gridEntity.returnParameter('gridInfo')
    if len(self.toBeSampled.keys()) != len(gridInfo.keys()): self.raiseAnError(IOError,'inconsistency between number of variables and grid specification')
    for varName, values in gridInfo.items():
      if values[1] != "custom" : self.raiseAnError(IOError,"The grid construct needs to be custom for variable "+varName)
      if len(values[2]) != 2   : self.raiseAnError(IOError,"The number of values can be accepted are only 2 (lower and upper bound) for variable "+varName)
    self.gridCoordinate = [None]*len(self.axisName)
    if len(self.gridCoordinate) < self.minNumbVars[self.respOpt['algorithm_type']]: self.raiseAnError(IOError,'minimum number of variables for type "'+ self.respOpt['type'] +'" is '+str(self.minNumbVars[self.respOpt['type']])+'!!')
    self.externalgGridCoord = True

  def localAddInitParams(self,tempDict):
    """
    Appends a given dictionary with class specific member variables and their
    associated initialized values.
    @ InOut, tempDict: The dictionary where we will add the initialization
                       parameters specific to this Sampler.
    """
    Grid.localAddInitParams(self,tempDict)
    for key,value in self.respOpt.items():
      if key != 'options': tempDict['Response Design '+key] = value
      else:
        for kk,val in value.items(): tempDict['Response Design options '+kk] = val

  def localInitialize(self):
    """
    This method initialize the response matrix. No actions are taken for full-factorial since it is equivalent to the Grid sampling this sampler is based on
    """
    if   self.respOpt['algorithm_type'] == 'boxbehnken'      : self.designMatrix = doe.bbdesign(len(self.gridInfo.keys()),center=self.respOpt['options']['ncenters'])
    elif self.respOpt['algorithm_type'] == 'centralcomposite': self.designMatrix = doe.ccdesign(len(self.gridInfo.keys()), center=self.respOpt['options']['centers'], alpha=self.respOpt['options']['alpha'], face=self.respOpt['options']['face'])
    gridInfo   = self.gridEntity.returnParameter('gridInfo')
    stepLenght = {}
    for cnt, varName in enumerate(self.axisName):
      self.mapping[varName] = np.unique(self.designMatrix[:,cnt]).tolist()
      gridInfo[varName] = (gridInfo[varName][0],gridInfo[varName][1],InterpolatedUnivariateSpline(np.array([min(self.mapping[varName]), max(self.mapping[varName])]),
                           np.array([min(gridInfo[varName][2]), max(gridInfo[varName][2])]), k=1)(self.mapping[varName]).tolist())
      stepLenght[varName] = [round(gridInfo[varName][-1][k+1] - gridInfo[varName][-1][k],14) for k in range(len(gridInfo[varName][-1])-1)]
    self.gridEntity.updateParameter("stepLenght", stepLenght, False)
    self.gridEntity.updateParameter("gridInfo", gridInfo)
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
class SparseGridCollocation(Grid):
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
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
    self.lastOutput     = None  #pointer to output dataObjects object
    self.ROM            = None  #pointer to ROM
    self.jobHandler     = None  #pointer to job handler for parallel runs
    self.doInParallel   = True  #compute sparse grid in parallel flag, recommended True
    self.existing       = []    #restart data points

    self._addAssObject('ROM','1')

  def _localWhatDoINeed(self):
    """
    This method is a local mirror of the general whatDoINeed method.
    It is implemented by the samplers that need to request special objects
    @ In , None, None
    @ Out, needDict, list of objects needed
    """
    gridDict = Grid._localWhatDoINeed(self)
    gridDict['internal'] = [(None,'jobHandler')]
    return gridDict

  def _localGenerateAssembler(self,initDict):
    """Generates the assembler.
    @ In, initDict, dict of init objects
    @ Out, None
    """
    Grid._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']
    #do a distributions check for ND
    for dist in self.distDict.values():
      if isinstance(dist,Distributions.NDimensionalDistributions): self.raiseAnError(IOError,'ND Dists not supported for this sampler (yet)!')

  def localInputAndChecks(self,xmlNode):
    """
    Reads in XML node
    @ In, xmlNode, XML node, input xml
    @ Out, None
    """
    self.doInParallel = xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes'] if 'parallel' in xmlNode.attrib.keys() else True
    self.writeOut = xmlNode.attrib['outfile'] if 'outfile' in xmlNode.attrib.keys() else None
    for child in xmlNode:
      if child.tag == 'Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
        self.axisName.append(varName)

  def localInitialize(self):
    """Performs local initialization
    @ In, None
    @ Out, None
    """
    for key in self.assemblerDict.keys():
      if 'ROM' in key:
        for value in self.assemblerDict[key]: self.ROM = value[3]
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0] #often need only one
    self.features = SVL.features
    self._generateQuadsAndPolys(SVL)
    #print out the setup for each variable.
    msg=self.printTag+' INTERPOLATION INFO:\n'
    msg+='    Variable | Distribution | Quadrature | Polynomials\n'
    for v in self.quadDict.keys():
      msg+='   '+' | '.join([v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type])+'\n'
    msg+='    Polynomial Set Degree: '+str(self.maxPolyOrder)+'\n'
    msg+='    Polynomial Set Type  : '+str(SVL.indexSetType)+'\n'
    self.raiseADebug(msg)

    self.raiseADebug('Starting index set generation...')
    self.indexSet = IndexSets.returnInstance(SVL.indexSetType,self)
    self.indexSet.initialize(self.distDict,self.importanceDict,self.maxPolyOrder)
    if self.indexSet.type=='Custom':
      self.indexSet.setPoints(SVL.indexSetVals)

    self.raiseADebug('Starting sparse grid generation...')
    self.sparseGrid = Quadratures.SparseQuad()
    # NOTE this is the most expensive step thus far; try to do checks before here
    self.sparseGrid.initialize(self.features,self.indexSet,self.distDict,self.quadDict,self.jobHandler,self.messageHandler)

    if self.writeOut != None:
      msg=self.sparseGrid.__csv__()
      outFile=file(self.writeOut,'w')
      outFile.writelines(msg)
      outFile.close()

    #if restart, figure out what runs we need; else, all of them
    if self.restartData != None:
      inps = self.restartData.getInpParametersValues()
      #make reorder map
      reordmap=list(inps.keys().index(i) for i in self.features)
      solns = list(v for v in inps.values())
      ordsolns = [solns[i] for i in reordmap]
      self.existing = zip(*ordsolns)

    self.limit=len(self.sparseGrid)
    self.raiseADebug('Size of Sparse Grid  :'+str(self.limit))
    self.raiseADebug('Number from Restart :'+str(len(self.existing)))
    self.raiseADebug('Number of Runs Needed :'+str(self.limit-len(self.existing)))
    self.raiseADebug('Finished sampler generation.')

    self.raiseADebug('indexset:',self.indexSet)
    for SVL in self.ROM.SupervisedEngine.values():
      SVL.initialize({'SG':self.sparseGrid,
                      'dists':self.distDict,
                      'quads':self.quadDict,
                      'polys':self.polyDict,
                      'iSet':self.indexSet})

  def _generateQuadsAndPolys(self,SVL):
    """
      Builds the quadrature objects, polynomial objects, and importance weights for all
      the distributed variables.  Also sets maxPolyOrder.
      @ In, SVL, one of the SupervisedEngine objects from the ROM
      @ Out, None
    """
    ROMdata = SVL.interpolationInfo()
    self.maxPolyOrder = SVL.maxPolyOrder
    #check input space consistency
    samVars=self.axisName[:]
    romVars=SVL.features[:]
    try:
      for v in self.axisName:
        samVars.remove(v)
        romVars.remove(v)
    except ValueError:
      self.raiseAnError(IOError,'variable '+v+' used in sampler but not ROM features! Collocation requires all vars in both.')
    if len(romVars)>0:
      self.raiseAnError(IOError,'variables '+str(romVars)+' specified in ROM but not sampler! Collocation requires all vars in both.')
    for v in ROMdata.keys():
      if v not in self.axisName:
        self.raiseAnError(IOError,'variable '+v+' given interpolation rules but '+v+' not in sampler!')
      else:
        self.gridInfo[v] = ROMdata[v] #quad, poly, weight
    #set defaults, then replace them if they're asked for
    for v in self.axisName:
      if v not in self.gridInfo.keys():
        self.gridInfo[v]={'poly':'DEFAULT','quad':'DEFAULT','weight':'1'}
    #establish all the right names for the desired types
    for varName,dat in self.gridInfo.items():
      self.raiseADebug('checking dat',dat.keys())
      self.raiseADebug('checking distDict',self.distDict.keys())
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
            self.raiseAnError(IOError,'Quadrature '+subType+' not compatible with Legendre polys for '+distr.type+' for variable '+varName+'!')
      else:
        quadType=dat['quad']
      if quadType not in distr.compatibleQuadrature:
        self.raiseAnError(IOError,' Quadrature type "'+quadType+'" is not compatible with variable "'+varName+'" distribution "'+distr.type+'"')

      quad = Quadratures.returnInstance(quadType,self,Subtype=subType)
      quad.initialize(distr,self.messageHandler)
      self.quadDict[varName]=quad

      poly = OrthoPolynomials.returnInstance(polyType,self)
      poly.initialize(quad,self.messageHandler)
      self.polyDict[varName] = poly

      self.importanceDict[varName] = float(dat['weight'])

  def localGenerateInput(self,model,myInput):
    """
      Provide the next point in the sparse grid.
      @ In, model, the model to evaluate
      @ In, myInput, list of original inputs
      @ Out, None
    """
    found=False
    while not found:
      try: pt,weight = self.sparseGrid[self.counter-1]
      except IndexError: raise utils.NoMoreSamplesNeeded
      if pt in self.existing:
        self.counter+=1
        if self.counter==self.limit: raise utils.NoMoreSamplesNeeded
        continue
      else:
        found=True
        for v,varName in enumerate(self.sparseGrid.varNames):
          self.values[varName] = pt[v]
          self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
        self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
        self.inputInfo['ProbabilityWeight'] = weight
        self.inputInfo['SamplerType'] = 'Sparse Grid Collocation'
#
#
#
#
class AdaptiveSparseGrid(AdaptiveSampler,SparseGridCollocation):
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    SparseGridCollocation.__init__(self)
    self.type             = 'AdaptiveSparseGridSampler'
    self.printTag         = self.type
    self.solns            = None #TimePointSet of solutions -> assembled
    self.ROM              = None #eventual final ROM object
    self.activeSGs        = OrderedDict() #list of potential SGs
    self.maxPolyOrder     = 0
    self.persistence      = 0    #number of forced iterations, default 2
    self.error            = 0    #estimate of percent of moment calculated so far
    self.moment           = 0
    self.oldSG            = None #previously-accepted sparse grid
    self.convType         = None #convergence criterion to use
    self.existing         = {} #rolling list of sampled points
    self.batchDone        = True #flag for whether jobHandler has complete batch or not
    self.unfinished       = 0 #number of runs still running when convergence complete

    self._addAssObject('TargetEvaluation','1')

  def localInputAndChecks(self,xmlNode):
    """
    Reads in XML node
    @ In, xmlNode, XML node, input xml
    @ Out, None
    """
    SparseGridCollocation.localInputAndChecks(self,xmlNode)
    if 'Convergence' not in list(c.tag for c in xmlNode): self.raiseAnError(IOError,'Convergence node not found in input!')
    convnode = xmlNode.find('Convergence')
    self.convType     = convnode.attrib['target']
    self.maxPolyOrder = int(convnode.attrib.get('maxPolyOrder',10))
    self.persistence  = int(convnode.attrib.get('persistence',2))
    self.maxRuns     = convnode.attrib.get('maxRuns',None)
    self.convValue    = float(convnode.text)

    if self.maxRuns is not None: self.maxRuns = int(self.maxRuns)

  def  localInitialize(self):
    """Performs local initialization
    @ In, None
    @ Out, None
    """
    #set a pointer to the end-product ROM
    self.ROM = self.assemblerDict['ROM'][0][3]
    #obtain the DataObject that contains evaluations of the model
    self.solns = self.assemblerDict['TargetEvaluation'][0][3]
    #set a pointer to the GaussPolynomialROM object
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0] #sampler doesn't always care about which target
    self.features=SVL.features #the input space variables
    mpo = self.maxPolyOrder #save it to re-set it after calling generateQuadsAndPolys
    self._generateQuadsAndPolys(SVL) #lives in GaussPolynomialRom object
    self.maxPolyOrder = mpo #re-set it

    #print out the setup for each variable.
    self.raiseADebug(' INTERPOLATION INFO:')
    self.raiseADebug(' {:^15} | {:^15} | {:^15} | {:^15}'.format('Variable','Distribution','Quadrature','Polynomials'))
    self.raiseADebug(' {0:*^16}|{0:*^17}|{0:*^17}|{0:*^17}'.format(''))
    #self.raiseADebug(' {:*^69}'.format(''))
    for v in self.quadDict.keys():
      self.raiseADebug(' {:^15} | {:^15} | {:^15} | {:^15}'.format(v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type))
        #self.raiseADebug('   '+' | '.join([v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type]))
    self.raiseADebug('    Polynomial Set Type  : adaptive')

    #create the index set
    self.raiseADebug('Starting index set generation...')
    self.indexSet = IndexSets.returnInstance('AdaptiveSet',self)
    self.indexSet.initialize(self.distDict,self.importanceDict,self.maxPolyOrder)

    #set up the already-existing solutions (and re-order the inputs appropriately)
    self._updateExisting()

    #make the first sparse grid ('dummy' is an unneeded index set)
    self.sparseGrid,dummy = self._makeSparseQuad(self.indexSet.active.keys())

    #set up the points we need RAVEN to run before we can continue
    self.neededPoints = []
    self.pointsNeededToMakeROM = []
    self.newSolutionSizeShouldBe = len(self.existing)
    for pt in self.sparseGrid.points()[:]:
      if pt not in self.pointsNeededToMakeROM:
        self.pointsNeededToMakeROM.append(pt)
      if pt not in self.neededPoints and pt not in self.existing.keys():
        self.neededPoints.append(pt)
        self.newSolutionSizeShouldBe+=1

  def _makeSparseQuad(self,points=[]):
    """
      Generates a sparseGrid object using the self.indexSet adaptively established points
      as well as and additional points passed in (often the indexSet's adaptive points).
      Also returns the index set used to generate the sparse grid.
      @ In, points, list of tuples
      @ Out, (sparseGrid, indexSet) object tuple
    """
    sparseGrid = Quadratures.SparseQuad()
    iset = IndexSets.returnInstance('Custom',self)
    iset.initialize(self.distDict,self.importanceDict,self.maxPolyOrder)
    iset.setPoints(self.indexSet.points)
    iset.addPoints(points)
    sparseGrid.initialize(self.features,iset,self.distDict,self.quadDict,self.jobHandler,self.messageHandler)
    return sparseGrid,iset

  def _makeARom(self,grid,inset):
    """
      Generates a GaussPolynomialRom object using the passed in sparseGrid and indexSet,
      otherwise fundamentally a copy of the end-target ROM.
      @ In, grid, a sparseGrid object
      @ In, inset, a indexSet object
      @ Out, a GaussPolynomialROM object
    """
    #deepcopy prevents overwriting
    rom  = copy.deepcopy(self.ROM) #preserves interpolation requests via deepcopy
    sg   = copy.deepcopy(grid)
    iset = copy.deepcopy(inset)
    sg.messageHandler   = self.messageHandler
    iset.messageHandler = self.messageHandler
    rom.messageHandler  = self.messageHandler
    for svl in rom.SupervisedEngine.values():
      svl.initialize({'SG'   :sg,
                      'dists':self.distDict,
                      'quads':self.quadDict,
                      'polys':self.polyDict,
                      'iSet' :iset
                      })
    #while the training won't always need all of solns, it is smart enough to take what it needs
    rom.train(self.solns)
    return rom

  def _impactParameter(self,new,old):
    """
      Calculates the impact factor g_k based on the Ayres-Eaton 2015 paper model.
      @ In, new, the new metric
      @ In, old, the old metric
      @ Out, the impact parameter
    """
    impact=0
    if abs(old)>1e-14: return((new-old)/old)
    else: return new

  def _updateExisting(self):
    """
      Goes through the stores solutions PointSet and pulls out solutions, ordering them
      by the order the features we're evaluating.
      @ In, None
      @ Out, None
    """
    if not self.solns.isItEmpty():
      inps = self.solns.getInpParametersValues()
      outs = self.solns.getOutParametersValues()
      #make reorder map
      reordmap=list(inps.keys().index(i) for i in self.features)
      solns = list(v for v in inps.values())
      ordsolns = [tuple(solns[i] for i in reordmap)]
      self.raiseADebug('      solns:',ordsolns)
      existinginps = zip(*ordsolns)
      outvals = zip(*list(v for v in outs.values()))
      self.existing = dict(zip(existinginps,outvals))
    else:
      self.raiseADebug('solns is empty!')

  def _integrateFunction(self,sg,r,i):
    """
      Uses the sparse grid sg to effectively integrate the r-th moment of the model.
      @ In, sg, sparseGrid object
      @ In, r, integer moment
      @ In, i, index of target to evaluate
      @ Out, float, approximate integral
    """
    tot=0
    for n in range(len(sg)):
      pt,wt = sg[n]
      if pt not in self.existing.keys(): self.raiseAnError(RuntimeError,'Trying to integrate with point',pt,'but it is not in the solutions!')
      tot+=self.existing[pt][i]**r*wt
    return tot

  def _convergence(self,sparseGrid,iset,i):
    """
      Checks the convergence of the adaptive index set via one of several ways, currently "mean", "variance", or "coeffs",
      meaning the moment coefficients of the stochastic polynomial expansion.
      @ In, sparseGrid, sparseGrid object
      @ In, iset, indexSet object
      @ In, i, index of target to check convergence with respect to
      @ Out, estimated impact factor for this index set and sparse grid
    """
    if self.convType.lower()=='mean':
      new = self._integrateFunction(sparseGrid,1,i)
      if self.oldSG!=None: old = self._integrateFunction(self.oldSG,1,i)
      else: old = 0
      impact = self._impactParameter(new,old)
    elif self.convType.lower()=='variance':
      new = self._integrateFunction(sparseGrid,2,i)
      if self.oldSG!=None:
        old = self._integrateFunction(self.oldSG,2,i)
      else: old = 0
      impact = self._impactParameter(new,old)
    elif self.convType.lower()=='coeffs':
      new = self._makeARom(sparseGrid,iset).SupervisedEngine.values()[i]
      tot = 0 #for L2 norm of coeffs
      if self.oldSG != None:
        oSG,oSet = self._makeSparseQuad()
        old = self._makeARom(oSG,oSet).SupervisedEngine.values()[i]
      else: old=None
      for coeff in new.polyCoeffDict.keys():
        if old!=None and coeff in old.polyCoeffDict.keys():
          n = new.polyCoeffDict[coeff]
          o = old.polyCoeffDict[coeff]
          tot+= (n - o)**2
        else:
          tot+= new.polyCoeffDict[coeff]**2
      impact = np.sqrt(tot)
    else: self.raiseAnError(KeyError,'Unexpected convergence criteria:',self.convType)
    return impact

  def localStillReady(self,ready):
    """
      Determines what additional points are necessary for RAVEN to run.
      @ In, ready, bool, true if ready
      @ Out, ready, bool, true if ready
    """
    #update existing solutions
    self.raiseAWarning('...checking local still ready: start')
    self._updateExisting()
    self.raiseAWarning('...checking local still ready: neededPoints')
    self.raiseAWarning('     ',self.neededPoints)
    #if we're not ready elsewhere, just be not ready
    if ready==False: return ready
    #if we still have a list of points to sample, just keep on trucking.
    if len(self.neededPoints)>0: return True
    #if points all submitted but not all done, not ready for now.
    if (not self.batchDone) or (not self.jobHandler.isFinished()):
      return False
    if len(self.existing) < self.newSolutionSizeShouldBe:
      #self.raiseADebug('Still collecting; existing has less points (%i) than it should (%i)!' %(len(self.existing),self.newSolutionSizeShouldBe))
      return False
    #if no points to check right now, search for points to sample
    while len(self.neededPoints)<1:
      self.raiseAWarning('checking local still ready 3')
      self.raiseADebug('')
      self.raiseADebug('Evaluating new points...')
      #update QoIs and impact parameters
      done=False
      self.error=0
      #re-evaluate impact of active set, since it could have changed
      for active in self.indexSet.active.keys():
        #create new SG using active point
        sparseGrid,iset = self._makeSparseQuad(active)
        #store it
        self.activeSGs[active]=sparseGrid
        #get impact from  convergence
        av_impact = 0
        for i,target in enumerate(self.ROM.SupervisedEngine.keys()):
          av_impact += self._convergence(sparseGrid,iset,i)
        impact = av_impact/float(len(self.ROM.SupervisedEngine.keys()))
        #stash the sparse grid, impact factor for future reference
        self.indexSet.setSG(active,sparseGrid)
        self.indexSet.setImpact(active,impact)
        #the estimated error is the sum of all the impacts
        self.error+=impact
      self.raiseAMessage('  estimated remaining error: %1.4e target error: %1.4e, runs: %i' %(self.error,self.convValue,len(self.pointsNeededToMakeROM)))
      if abs(self.error)<self.convValue and len(self.indexSet.points)>self.persistence:
        done=True #we've converged!
        self.raiseADebug('converged estimated error:',self.error)
        #clear the active index set
        for key in self.indexSet.active.keys():
          if self.indexSet.active[key]==None: del self.indexSet.active[key]
        #clear needed points
        self.neededPoints=[]
        break
      elif self.maxRuns is not None:
        self.unfinished = self.jobHandler.numRunning()
        if len(self.pointsNeededToMakeROM)-self.unfinished >=self.maxRuns:
          done=True #not converged, but reached max number of polys to use
          self.raiseAMessage('Not converged, but max runs (%i) reached!' %self.maxRuns)
          self.raiseADebug('end estimated error:',self.error)
          for key in self.indexSet.active.keys():
            if self.indexSet.active[key]==None: del self.indexSet.active[key]
          #clear needed points
          self.neededPoints=[]
          break
      #if we're not converged...
      self.raiseADebug('new iset:')
      self.indexSet.printOut()
      #store the old rom, if we have it
      if len(self.indexSet.points)>1:
        self.oldSG = self.activeSGs[self.indexSet.newestPoint]
      #get the active point with the biggest impact and make him permanent
      point,impact = self.indexSet.expand()
      # find the forward points of the most effective point
      self.indexSet.forward(point,self.maxPolyOrder)
      #find the new points needed to evaluate, if any (there should be usually)
      for point in self.indexSet.active.keys():
        sparseGrid,dummy=self._makeSparseQuad(point)
        for pt in sparseGrid.points()[:]:
          if pt not in self.pointsNeededToMakeROM:
            self.pointsNeededToMakeROM.append(pt)
          if pt not in self.neededPoints and pt not in self.existing.keys():
            self.newSolutionSizeShouldBe+=1
            self.neededPoints.append(pt)
    #if we exited the while-loop searching for new points and there aren't any, we're done!
    if len(self.neededPoints)==0:
      self.indexSet.printOut()
      self.finalizeROM()
      self.unfinished = self.jobHandler.numRunning()
      self.jobHandler.terminateAll()
      return False
    #otherwise, we have work to do.
    return True

  def finalizeROM(self):
    """
      Initializes final target ROM with necessary objects for training.
      @ In, None
      @ Out, None
    """
    self.raiseADebug('No more samples to try! Declaring sampling complete.')
    #initialize final rom with final sparse grid and index set
    self.sparseGrid = Quadratures.SparseQuad()
    self.sparseGrid.initialize(self.features,self.indexSet,self.distDict,self.quadDict,self.jobHandler,self.messageHandler)
    for SVL in self.ROM.SupervisedEngine.values():
      SVL.initialize({'SG':self.sparseGrid,
                      'dists':self.distDict,
                      'quads':self.quadDict,
                      'polys':self.polyDict,
                      'iSet':self.indexSet,
                      'numRuns':len(self.pointsNeededToMakeROM)-self.unfinished})
    self.indexSet.printHistory()
    self.indexSet.writeHistory()

  def localGenerateInput(self,model,myInput):
    """
      Generates an input. Parameters inherited.
      @ In, model, unused
      @ In, myInput, unused
    """
    pt = self.neededPoints.pop() # [self.counter-1]
    for v,varName in enumerate(self.sparseGrid.varNames):
      self.values[varName] = pt[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['SamplerType'] = self.type

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """Performs actions after samples have been collected.
    @ In, jobObject, the job that finished
    @ In, model, the model that was run
    @ In, myInput, the input used for the run
    @Out, None
    """
    #check if all sampling is done
    if self.jobHandler.isFinished(): self.batchDone = True
    else: self.batchDone = False
#
#
#
#
class Sobol(SparseGridCollocation):
  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Grid.__init__(self)
    self.type           = 'SobolSampler'
    self.printTag       = 'SAMPLER SOBOL'
    self.assemblerObjects={}    #dict of external objects required for assembly
    self.maxPolyOrder   = None  #L, the relative maximum polynomial order to use in any dimension
    self.sobolOrder     = None  #S, the order of the HDMR expansion (1,2,3), queried from the sobol ROM
    self.indexSetType   = None  #the type of index set to use, queried from the sobol ROM
    self.polyDict       = {}    #varName-indexed dict of polynomial types
    self.quadDict       = {}    #varName-indexed dict of quadrature types
    self.importanceDict = {}    #varName-indexed dict of importance weights
    self.references     = {}    #reference (mean) values for distributions, by var
    self.solns          = None  #pointer to output dataObjects object
    self.ROM            = None  #pointer to sobol ROM
    self.jobHandler     = None  #pointer to job handler for parallel runs
    self.doInParallel   = True  #compute sparse grid in parallel flag, recommended True
    self.existing       = []

    self._addAssObject('ROM','1')

  def _localWhatDoINeed(self):
    """
      Used to obtain necessary objects.
      @ In, None
      @ Out, None
    """
    gridDict = Grid._localWhatDoINeed(self)
    gridDict['internal'] = [(None,'jobHandler')]
    return gridDict

  def _localGenerateAssembler(self,initDict):
    """
      Used to obtain necessary objects.
      @ In, initDict, dictionary of objects required to initialize
      @ Out, None
    """
    Grid._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']

  def localInputAndChecks(self,xmlNode):
    """
      Extended readMoreXML after other objects are instantiated
      @ In, xmlNode, xmlNode object whose head should be Sobol under Sampler.
      @ Out, None
    """
    self.doInParallel = xmlNode.attrib['parallel'].lower() in ['1','t','true','y','yes'] if 'parallel' in xmlNode.attrib.keys() else True
    self.writeOut = xmlNode.attrib['outfile'] if 'outfile' in xmlNode.attrib.keys() else None
    for child in xmlNode:
      if child.tag == 'Distribution':
        varName = '<distribution>'+child.attrib['name']
      elif child.tag == 'variable':
        varName = child.attrib['name']
        self.axisName.append(varName)

  def localInitialize(self):
    """
      Initializes Sampler, including building sub-ROMs for Sobol decomposition.  Note that re-using this
      sampler will destroy any ROM trained and attached to this sampler, and can be retrained after sampling.
      @ In, None
      @ Out, None
    """
    for key in self.assemblerDict.keys():
      if 'ROM' in key:
        indice = 0
        for value in self.assemblerDict[key]:
          self.ROM = self.assemblerDict[key][indice][3]
          indice += 1
    #make combination of ROMs that we need
    self.targets  = self.ROM.SupervisedEngine.keys()
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0]
    self.sobolOrder = SVL.sobolOrder
    self._generateQuadsAndPolys(SVL)
    features = SVL.features
    needCombos = itertools.chain.from_iterable(itertools.combinations(features,r) for r in range(self.sobolOrder+1))
    self.SQs={}
    self.ROMs={} #keys are [target][combo]
    for t in self.targets: self.ROMs[t]={}
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
      self.SQs[combo].initialize(combo,iset,distDict,quadDict,self.jobHandler,self.messageHandler)
      # initDict is for SVL.__init__()
      initDict={'IndexSet'       :iset.type,        # type of index set
                'PolynomialOrder':SVL.maxPolyOrder, # largest polynomial
                'Interpolation'  :SVL.itpDict,      # polys, quads per input
                'Features'       :','.join(combo),  # input variables
                'Target'         :None}             # set below, per-case basis
      #initializeDict is for SVL.initialize()
      initializeDict={'SG'   :self.SQs[combo],      # sparse grid
                      'dists':distDict,             # distributions
                      'quads':quadDict,             # quadratures
                      'polys':polyDict,             # polynomials
                      'iSet' :iset}                 # index set
      for name,SVL in self.ROM.SupervisedEngine.items():
        initDict['Target']     = SVL.target
        self.ROMs[name][combo] = SupervisedLearning.returnInstance('GaussPolynomialRom',self,**initDict)
        self.ROMs[name][combo].initialize(initializeDict)
    #if restart, figure out what runs we need; else, all of them
    if self.restartData != None:
      inps = self.restartData.getInpParametersValues()
      self.existing = zip(*list(v for v in inps.values()))
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
    #if tuple(newpt) not in existing:
    self.pointsToRun.append(tuple(newpt))
    #now do the rest
    for combo,rom in self.ROMs.values()[0].items(): #each target is the same, so just for each combo
      SG = rom.sparseGrid #they all should have the same sparseGrid
      SG._remap(combo)
      for l in range(len(SG)):
        pt,wt = SG[l]
        newpt = np.zeros(len(std))
        for v,var in enumerate(std):
          if var in combo: newpt[v] = pt[combo.index(var)]
          else: newpt[v] = self.references[var]
        newpt=tuple(newpt)
        if newpt not in self.pointsToRun:# and newpt not in existing: #the second half used to be commented...
          self.pointsToRun.append(newpt)
    self.limit = len(self.pointsToRun)
    self.raiseADebug('Needed points: %i' %self.limit)
    self.raiseADebug('From Restart : %i' %len(self.existing))
    self.raiseADebug('Still Needed : %i' %(self.limit-len(self.existing)))
    initdict={'ROMs':None, #self.ROMs,
              'SG':self.SQs,
              'dists':self.distDict,
              'quads':self.quadDict,
              'polys':self.polyDict,
              'refs':self.references}
    for target in self.targets:
      initdict['ROMs'] = self.ROMs[target]
      self.ROM.SupervisedEngine[target].initialize(initdict)

  def localGenerateInput(self,model,myInput):
    """
      Generates an input. Parameters inherited.
      @ In, model, unused
      @ In, myInput, unused
    """
    found=False
    while not found:
      try: pt = self.pointsToRun[self.counter-1]
      except IndexError: raise utils.NoMoreSamplesNeeded
      if pt in self.existing:
        self.counter+=1
        if self.counter==self.limit: raise utils.NoMoreSamplesNeeded
        continue
      else: found=True
      for v,varName in enumerate(self.distDict.keys()):
        self.values[varName] = pt[v]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
      self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
      #self.inputInfo['ProbabilityWeight'] =  N/A
      self.inputInfo['SamplerType'] = 'Sparse Grids for Sobol'
#
#
#
#
class AdaptiveSobol(Sobol,AdaptiveSparseGrid):
  def __init__(self):
    '''
      Initializes members to be used in the sampler.
      @ In, None
      @ Out, None
    '''
    Sobol.__init__(self)
    self.type = 'AdaptiveSobolSampler'
    self.printTag = 'SAMPLER ADAPTIVE SOBOL'
    self.maxComboCard = None

    self.SQs      = {} #stores sparse grid quadrature objects
    self.samplers = {} #stores adaptive sparse grid sampling objects
    self.romshell = {} #stores Model.ROM objects for each combo

    self._addAssObject('TargetEvaluation','1')

  def localInputAndChecks(self,xmlNode):
    '''
      Extended readMoreXML after other objects are instantiated
      @ In, xmlNode, xmlNode object whose head should be Sobol under Sampler.
      @ Out, None
    '''
    Sobol.localInputAndChecks(self,xmlNode)
    foundConv = False
    for child in xmlNode:
      if child.tag == 'Convergence':
        foundConv = True
        self.convType      = child.attrib['target'] #TODO
        self.maxSobolOrder = int(child.attrib.get('maxSobolOrder',2)) #TODO
        self.maxPolyOrder = int(child.attrib.get('maxPolyOrder',10)) #TODO
        self.convValue     = float(child.text)
    if not foundConv:
      self.raiseAnError(IOError,'Convergence node not found in input!')

  def localInitialize(self):
    '''
      Initializes Sampler, including building sub-ROMs for Sobol decomposition.  Note that re-using this
      sampler will destroy any ROM trained and attached to this sampler, and can be retrained after sampling.
      @ In, None
      @ Out, None
    '''
    self.solns = self.assemblerDict['TargetEvaluation'][0][3]
    self.ROM = self.assemblerDict['ROM'][0][3]
    SVLs = self.ROM.SupervisedEngine.values()
    SVL = SVLs[0]
    self._generateQuadsAndPolys(SVL)
    self.features     = SVL.features
    self.targets      = self.ROM.initializationOptionDict['Target'].split(',')
    self.iSets        = {} #dict of adaptive index sets
    self.ROMs         = {} #dict of adaptive ROMs
    self.samplers     = {}
    self.inTraining   = {} #dict of samplers still in sampling
    self.doneTraining = {} #dict of samplers finished sampling
    self.useSet       = {} #dict of accepted combos organized by level
    self.actImpact    = {} #dict of actual impacts by combo
    self.expImpact    = {} #dict of predicted impacts by combo
    #calculate first order combos
    self.raiseADebug('features:',self.features)
    self.first_combos = itertools.chain.from_iterable(itertools.combinations(self.features,r) for r in [1,1])
    #return itertools.chain.from_iterable(itertools.combinations(self.features,r) for r in range(low,high+1))
    self.raiseADebug('first set:')
    for c in self.first_combos:
      self.raiseADebug('  ',c)
      self._makeComboParts(c)
      self.inTraining[c]=self.samplers[c]
    #establish reference cut
    self.references={}
    for var,dist in self.distDict.items():
      self.references[var] = dist.untruncatedMean() #TODO run this case!
    #load the solutions up
    self.existing=[]
    self._updateExisting()
    #collect first set of runs needed
    self.neededPoints=[tuple(self.references[var] for var in self.features)]
    self._collectNeededPoints(self.first_combos)

  def _convergence(self):
    pass #TODO

  def _fillCutPoint(self,combo,pt):
    '''
      Fills a cut point with reference values
      @ In, combo, tuple(str) of the features provided
      @ In, pt, tuple(float) of len(combo), feature values
      @ Out, tuple(float), full evaluation point
    '''
    newpt = np.zeros(len(self.features))
    for v,var in enumerate(self.features):
      if var in combo: newpt[v] = pt[combo.index(var)]
      else:            newpt[v] = self.references[var]
    return tuple(newpt)

  def _checkCutPoint(self,combo,pt):
    '''
      Establishes whether a point is part of the cut set for features in combo
      @ In, combo, tuple(str), the desired features
      @ In, pt, tuple(float), the full point
      @ Out, bool, true if pt only varies in combo dimensions from the reference point
    '''
    isCut = True
    for v,var in enumerate(self.features):
      if var in combo:continue
      if pt[v] != self.references[var]:
        isCut = False
        break
    return isCut

  def _makeCombos(self):#,low,high):
    '''
      Returns a list of the possible subset combinations with a cardinality from low to high.
      @ In,  low,         the smallest subset size (number of variables included in each subset)
      @ In,  high,        the maximum subset size
      @ Out, list(tuple), list of combinations
    '''
    l = max(len(c) for c in self.useSet.keys())
    #TODO this is inefficient; we really want a cross product of useSet with firstSet, but I haven't found a way
    potential = itertools.combinations(self.features,l+1)
    for p in potential:
      if all(set(c).issubset(p) for c in self.useSet.keys()):
        self._makeComboParts(p)
        self.inTraining[p] = self.samplers[c]
        self.expImpact[p]  = self._calcExpImpact[p]
    #order expImpact by low impact
    tosort = zip(self.expImpact.keys(),self.expImpact.values())
    tosort.sort(key=itemgetter(1)) #not reversed, so pop() does last-in-first-out
    self.expImpact = OrderedDict()
    for key,impact in tosort:
        self.expImpact[key]=impact

    #return itertools.chain.from_iterable(itertools.combinations(self.features,r) for r in range(low,high+1))
    #do cross product of existing with originals

  def _collectNeededPoints(self,combos):
    '''
      Goes through subset samplers and adds neededPoints to queue
      @In, combos, list of tuples to collect needed points from (probably usually all the untrained combos)
    '''
    for c in combos:
      self.raiseADebug('checking needs for combo',c)
      if self.expImpact.get(c,self.convValue+1)>self.convValue:
        while len(self.samplers[c].neededPoints) > 0:
          pt = self.samplers[c].neededPoints.pop()
          self.raiseADebug('...checking on point',pt)
          fullpt = self._fillCutPoint(c,pt)
          if fullpt not in self.neededPoints and fullpt not in self.existing:
            self.raiseADebug('...adding point',pt)
            self.neededPoints.append(fullpt)

  def _makeComboParts(self,combo):
    '''
      Constructs a single ROM for the given subset (combo).
      @ In, combo, tuple(string) subset description, i.e. ('x','y')
      @ Out, GaussPolynomialROM object
    '''
    self.raiseADebug('Generating sub-combos for combo',combo)
    #node = ET.Element('sampler')
    #node.append(ET.Element('Convergence',text=self.convValue))
    #for c in combo:
    #    var = str(c)
    #    vnode = ET.Element('variable',text=var)
    SVL = self.ROM.SupervisedEngine.values()[0] #FIXME multitarget
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
    iset = IndexSets.returnInstance('AdaptiveSet',self)
    iset.initialize(distDict,imptDict,self.maxPolyOrder)
    self.raiseADebug('iset pts:',iset)
    #TODO intialize sparse quad! FIXME does this work?  We need adaptive sparse grid
    # I'm pretty sure this is a dummy quadrature that will get replaced.
    #  ...on the other hand, it appears we're getting empty sparse grid quadratures...
    self.SQs[combo] = Quadratures.SparseQuad()
    self.SQs[combo].initialize(combo,iset,distDict,quadDict,self.jobHandler,self.messageHandler)
    #initDict       is for SVL.__init__
    #initializeDict is for SVL.initialize()
    initDict =       {'IndexSet'       : iset.type,         # type of the index set
                      'PolynomialOrder': SVL.maxPolyOrder,  # largest polynomial
                      'Interpolation'  : SVL.itpDict,       # poly,quads per input
                      'Features'       : ','.join(combo),   # input variables
                      'Target'         : SVL.target}        # TODO make it work for multitarget
    initializeDict = {'SG'             : self.SQs[combo],   # sparse grid
                      'dists'          : distDict,          # distributions
                      'quads'          : quadDict,          # quadratures
                      'polys'          : polyDict,          # polynomials
                      'iSet'           : iset}              # index set
    self.ROMs[combo] = SupervisedLearning.returnInstance('GaussPolynomialRom',self,**initDict)
    self.ROMs[combo].initialize(initializeDict)
    #set up for adaptive sampling
    #...make the shell ROM for this combo
    self.romshell[combo] = Models.returnInstance('ROM',self)
    self.romshell[combo].messageHandler = self.messageHandler
    self.romshell[combo].SupervisedEngine[SVL.target] = self.ROMs[combo]
    #...make adaptive SG sampler,simulate localInputAndChecks
    nsamp = AdaptiveSparseGrid()
    nsamp.messageHandler = self.messageHandler
    nsamp.doInParallel = self.doInParallel
    for var in combo: nsamp.axisName.append(var)
    nsamp.convType='variance'
    nsamp.maxPolyOrder = self.maxPolyOrder
    nsamp.persistence = 2
    nsamp.convValue = self.convValue
    nsamp.distDict = distDict
    nsamp.assemblerDict['ROM']              = [['','','',self.romshell[combo] ]]
    nsamp.assemblerDict['TargetEvaluation'] = [['','','',self.solns           ]]
    nsamp.localInitialize()
    #propogate sparse grid back from sampler
    self.SQs[combo] = nsamp.sparseGrid
    self.ROMs[combo].sparseGrid = nsamp.sparseGrid
    self.raiseADebug('SG:',nsamp.sparseGrid)
    self.samplers[combo] = nsamp

  def _updateExisting(self):
    '''
      Collects solutions from TargetEvaluation timepointset.  Also updates timepointset for all subsets.
      @ In, None
      @ Out, None
    '''
    AdaptiveSparseGrid._updateExisting(self)
    #make subset solns for subsets
    #solns={}
    for combo,sampler in self.samplers.items():
      #solns[combo] = []
      counter = 0
      self.raiseADebug('DataObject for combo',combo,self.samplers[combo].solns)
      if self.samplers[combo].solns is None or self.samplers[combo].solns.isItEmpty():
        self.raiseADebug('Creating DataObject for combo',combo)
        dataObj = DataObjects.returnInstance('PointSet',self)
        dataObj.type='PointSet'
        #write XML for intializing
        datanode = ET.Element('PointSet',{'name':str(combo)})
        #inputs
        inpnode = ET.Element('Input')
        inpnode.text=','.join(c for c in combo)
        datanode.append(inpnode)
        #outputs
        outnode = ET.Element('Output')
        outnode.text=','.join(self.targets)
        datanode.append(outnode)
        #initialize
        dataObj.readXML(datanode,self.messageHandler)
      else:
        dataObj = self.samplers[combo].solns
      #add in relevant cut-hyperplane data
      if len(self.existing)>0:
        self.raiseADebug('in existing:',self.existing)
        self.raiseADebug('targets:',self.targets)
        for inp,soln in self.existing.items():
          if self._checkCutPoint(combo,inp): #indicates it's part of the desired cut hyperplane
            #solns[combo].append(soln)
            for i,c in enumerate(combo):
              dataObj.updateInputValue(c,inp[i])
            for i,c in enumerate(self.targets):
               dataObj.updateOutputValue(c,soln[i])
      self.samplers[combo].solns = dataObj

  def _calcActualImpact(self,combo):
    '''
      Calculates the actual impact of adding combo to the HDMR representation.
      @ In, combo, the combo to evaluate
      @ Out, float, the impact parameter
    '''
    contrib=abs(self.ROMs[combo].__variance__())
    denom = 0
    for donecombo in sorted(self.useSet.keys(),key=len):
      if set(donecombo).issubset(set(combo)):
        denom+=self.ROMs[donecombo].__variance__()
    if denom>0: return contrib/abs(denom)
    else: return contrib

  def _calcExpImpact(self,combo):
    '''
    Calculates the expectcted impact of a parameter based on the product of its subsets.
    @ In, combo, the combo to evaluate
    @ Out, float, the expected impact
    '''
    impact=1
    for donecombo in sorted(self.useSet.keys(),key=len):
      if set(donecombo).issubset(set(combo)):
        impact*=self.actImpact[donecombo]
    return impact

  def _calcConvergence(self,newcombo):
    '''
      Calculates the total impact of the current set.
      @ In, None
      @ Out, float, total impact
    '''
    sum1 = 0
    for combo in self.useSet.keys():
      sum1+=self.ROMs[combo].__variance__()
    sum2 = sum1+self.ROMs[newcombo].__variance__()
    self.useSet[newcombo]=self.ROMs[newcombo]
    return abs(sum2-sum1)/abs(sum1)

  def _initializeROM(self):
    #used?
    self.raiseAnError(RuntimeError,'I get used!')
    initdict={'ROMs':self.useSet,
            'dists':self.distDict,
            'quads':self.quadDict,
            'polys':self.polyDict,
            'refs':self.references}
    self.ROM.SupervisedEngine.values()[0].initialize(initdict) #TODO FIXME multitarget

  def localStillReady(self,ready):
    #update existing solutions
    self._updateExisting()
    #if we're already not ready, just return it
    if ready==False: return ready
    #if we still have points to run, submit them
    if len(self.neededPoints)>0: return True
    #look for convergence or new points
    while len(self.neededPoints)<1:
      #check local ready on outstanding ROMs
      alldone = True
      for combo,sampler in self.inTraining.items():
        self.raiseADebug('checking ready on',combo)
        stillrunning=sampler.localStillReady(True) #this might have added more points to sampler.neededPoints
        if not stillrunning: #aka if done sampling
          self.doneTraining[combo]=sampler
          del self.inTraining[combo]
          self.raiseADebug('training dataobj print:',self.samplers[combo].solns.name)
          self.samplers[combo].solns.printCSV()
          self.romshell[combo].train(self.samplers[combo].solns)
          self.actImpact[combo] = self._calcActualImpact(combo)
        else:
          alldone=False #FIXME wait and re-call?
      if not alldone:
        self._collectNeededPoints(self.inTraining.keys())
        self.raiseAnError(IOError,'break')
      #TODO FIXME convergence tests not yet implemented!
      #check total convergence
      if len(self.expImpact)>0: impcombo = self.expImpact.keys()[-1]
      # ELSE? TODO FIXME
      while impcombo in self.doneTraining.keys(): #expected highest-impact is done
        totconv = self._calcConvergence(impcombo)
        del self.expImpact
        if totconv < self.convValue:
          self._finalizeROM()
          return False
      if alldone:
        #make a new layer
        self.makeCombos()
        self._collectNeededPoints(self.expImpact.keys())
    #check for no new points to run #TODO might need to check for runs still running?
    if len(self.neededPoints)==0:
      self.finalizeROM()
      return False
    return True

  def localGenerateInput(self,model,oldInput):
    pt = self.neededPoints.pop()
    for v,varName in enumerate(self.distDict.keys()):
      self.values[varName] = pt[v]
      self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(self.values[varName])
    self.inputInfo['PointsProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['SamplerType'] = 'Adaptive Sobol Sparse Grid'


"""
 Interface Dictionary (factory) (private)
"""
__base = 'Sampler'
__interFaceDict = {}
__interFaceDict['MonteCarlo'              ] = MonteCarlo
__interFaceDict['DynamicEventTree'        ] = DynamicEventTree
__interFaceDict['Stratified'              ] = Stratified
__interFaceDict['Grid'                    ] = Grid
__interFaceDict['LimitSurfaceSearch'      ] = LimitSurfaceSearch
__interFaceDict['AdaptiveDynamicEventTree'] = AdaptiveDET
__interFaceDict['FactorialDesign'         ] = FactorialDesign
__interFaceDict['ResponseSurfaceDesign'   ] = ResponseSurfaceDesign
__interFaceDict['SparseGridCollocation'   ] = SparseGridCollocation
__interFaceDict['AdaptiveSparseGrid'      ] = AdaptiveSparseGrid
__interFaceDict['Sobol'                   ] = Sobol
__interFaceDict['AdaptiveSobol'           ] = AdaptiveSobol
__knownTypes = list(__interFaceDict.keys())

def knownTypes():
  return __knownTypes

def addKnownTypes(newDict):
  for name, value in newDict.items():
    __interFaceDict[name]=value
    __knownTypes.append(name)

def returnInstance(Type,caller):
  """
  function used to generate a Sampler class
  @ In, Type : Sampler type
  @ Out,Instance of the Specialized Sampler class
  """
  try: return __interFaceDict[Type]()
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+Type)

def optionalInputs(Type):
  pass

def mandatoryInputs(Type):
  pass
