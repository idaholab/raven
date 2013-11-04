'''
Module where the base class and the specializzation of different type of sampler are
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
import numpy as np
import xml.etree.ElementTree as ET
from BaseType import BaseType
from itertools import product as iterproduct
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Quadrature
#Internal Modules End--------------------------------------------------------------------------------

class Sampler(BaseType):
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

  --Other external methods--
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
  self.localFinalizeActualSampling(jobObject,model,myInput):
  
  '''

  def __init__(self):
    BaseType.__init__(self) 
    self.counter      = 0           # Counter of the samples performed (better the input generated!!!). It is reset by calling the function self.initialize
    self.limit        = sys.maxsize # maximum number of Samples (for example, Monte Carlo = Number of Histories to run, DET = Unlimited)
    self.toBeSampled  = {}           # Sampling mapping dictionary {'Variable Name':['type of distribution to be used', 'name of the distribution']}
    self.distDict     = {}           # Contains the instance of the distribution to be used, it is created every time the sampler is initialized. keys are the variable names
    self.values       = {}           # for each variable the current value {'var name':value}
    self.inputInfo    = {}           # depending on the sampler several different type of keywarded information could be present only one is mandatory..
    self.inputInfo['SampledVars'  ] = self.values
    self.inputInfo['SampledVarsPb'] = {}
    self.initSeed     = None         #Every time the sampler is call if he possesses a seed it going to reinitialize the seed for the hole environment 
                                     #(if not read remains none meaning the seeds will be the one of the previous run)

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    The text i supposed to contain the info where and which variable to change.
    In case of a code the syntax is specified by the code interface itself
    '''
    try:    self.initSeed = int(xmlNode.attrib['initial_seed'])
    except KeyError: pass
    try:    self.limit    = int(xmlNode.attrib['limit'])
    except KeyError: pass
    for child in xmlNode:
      for childChild in child:
        if childChild.tag =='distribution':
          self.toBeSampled[child.attrib['name']] = [childChild.attrib['type'],childChild.text]
    self.localInputAndChecks(xmlNode)

  def localInputAndChecks(self,xmlNode):
    '''place here the additional reading, remember to add initial parameters in the method localAddInitParams'''
  
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
    tempDict['counter' ] = self.counter
    for key in self.inputInfo:
      if key!='SampledVars':tempDict[key] = self.inputInfo[key]
      else:
        for var in self.inputInfo['SampledVars'].keys():
          tempDict['Variable: '+var+' has value']=tempDict[key][var]
    self.localAddCurrentSetting(tempDict)

  def localAddCurrentSetting(self,tempDict):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''

  def generateDistributions(self,availableDist):
    '''used to restart the random number generators of the distributions that will be used
    the trick is that if you want to add two step of MonteCarlo, to be uncorrelated the second sampler either have a different seed either none.
    In case no seed is specified a random seed is used.
    @in availableDist: {'distribution name':instance}
    '''
    if self.initSeed==None: self.initSeed = np.random.random_integers(0,sys.maxint.bit_length())
    np.random.seed(self.initSeed)
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]]
      self.distDict[key].initializeDistribution()
   
  def initialize(self):
    '''
    This function should be called every time a clean sampler is needed. Called before takeAstep in <Step>
    @ In/Out, None
    '''
    self.counter = 0
    self.localInitialize()
    
  def localInitialize(self):
    '''use this function to export to the printer in the base class the additional PERMANENT your local class have'''
    pass
    
  def amIreadyToProvideAnInput(self):
    '''
    This is a method that should be call from any user of the sampler before requiring the generation of a new sample.
    This method act as a semaphore for generating a new input.
    Reason for not being ready could be for example: exceeding number of samples, waiting for other simulation for providing more information etc. etc.
    @return Boolean
    '''
    if(self.counter < self.limit): ready = True
    else                         : ready = False
    ready = self.localStillReady(ready)
    return ready
  
  def localStillReady(self,ready):
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

  def localGenerateInput(self,model,oldInput):
    '''
    This class need to be overwritten since it is here that the magic of the sampler happens.
    After this method call the self.inputInfo should be ready to be sent to the model
    @in model   : it is the instance of a model
    @in oldInput: [] a list of the original needed inputs for the model (e.g. list of files, etc. etc)
    '''
    raise NotImplementedError('The sampler of type '+self.type+' seems not yet capable to decide how to sample (localGenerateInput not yet implemented)')

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
class StochasticCollocation(Sampler):
  '''
  STOCHASTIC COLLOCATION Sampler 
  '''
  
  def __init__(self):
    Sampler.__init__(self)
    self.min_poly_order = 0 #lowest acceptable polynomial order
    self.var_poly_order = dict() #stores poly orders for each var
    self.availableDist = None #container of all available distributions

  def localInitialize(self):
    # assign values to undefined order variables
    if self.min_poly_order>0:
      if len( set(self.var_poly_order) ^ set(self.distDict) )==0: #keys correspond exactly
        if sum(self.var_poly_order.values())<self.min_poly_order:
          raise IOError('Minimum total polynomial order is set greater than sum of all variable orders!')
      else:
        r_keys = set(self.distDict) - set(self.var_poly_order) #var orders not set yet
        r_order = min(len(r_keys),self.min_poly_order-sum(self.var_poly_order.values())) #min remaining order needed
        for key in r_keys:
          self.var_poly_order[key]=int(round(0.5+r_order/(len(r_keys))))
    self.limit=np.product(self.var_poly_order.values())-1
    #TODO Shouldn't need to -1 here; where should it happen?
    #tried to put it in Steps, MultiRun.initializeStep, set maxNumberIteration, didn't work.
      
    self.generateQuadrature()
    
  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    #Sampler.readMoreXML(self,xmlNode) # overwritten
    # attempt to set minimum total function polynomial expansion order
    try: self.min_poly_order = int(xmlNode.attrib['min_poly_order'])
    except: self.min_poly_order = 0
    
    # attempt to set polynomial expansion order for individual params
    for child in xmlNode:
      self.toBeSampled[child.text]    = [child.attrib['type'],child.attrib['distName']] 
      self.var_poly_order[child.text] = int(child.attrib['poly_order'])
    
    # won't work if there's no uncertain variables!
    if len(self.toBeSampled.keys()) == 0:
      raise IOError('No uncertain variables to sample!')
    return

  def GenerateInput(self,model,myInput):
    '''
    Function used to generate an input.
    It returns the model.createNewInput() passing into it the type of sampler,
    the values to be used and the some add info in the values dict
    @ In, model: Model object instance
    @ Out, myInputs: Original input files
    '''
    if self.debug: print('generate input model:',model)
    quad_pts=self.quad.quad_pts[self.counter-1]
    quad_pt_index = self.quad.quad_pt_index[quad_pts]
    self.inputInfo['quad_pts']       = quad_pts
    self.inputInfo['partial coeffs'] = self.partCoeffs[quad_pts].values()
    self.inputInfo['exp order']      = self.quad.quad_pt_index[quad_pts]
    #values={}
    #values['prefix']={'counter'       :str(self.counter),
    #                  'quad pts'      :str(quad_pts),
    #                  'partial coeffs':str(self.partCoeffs[quad_pts])}
    #values['prefix']=(('counter'   ,'quad pts','partial coeffs'),
    #                  (self.counter, quad_pts,str(self.partCoeffs[quad_pts].values())))
    # TODO would be beneficial to pass the orders of quad pt, too?
    for var in self.distDict.keys():
      self.inputInfo['SampledVars'][var]=self.distDict[var].actual_point(\
          quad_pts[self.quad.dict_quads[self.quad.quads[var]]])
      #print('run',self.counter,'for var '+var+' set value',values['vars'][var])
    return

  def generateQuadrature(self):
    quads={}
    for var in self.distDict.keys():
      #TODO see above, this won't work for quads that need addl params
      #  Example: Laguerre, Jacobi
      #  create a dict for addl params lists to *add to quadrature init call?
      #  this for sure works, even if it's empty!
      quads[var]=self.distDict[var].bestQuad(self.var_poly_order[var])
    self.quad=Quadrature.MultiQuad(quads)
    self.partCoeffs={}
    for quad_pt in self.quad.indx_quad_pt.values(): #quadrature points
      self.partCoeffs[quad_pt]={}
      for ords in list(iterproduct(*[range(self.var_poly_order[var]) for var in self.distDict.keys()])):
        self.partCoeffs[quad_pt][ords]=0
        poly=weight=probNorm=1.
        for v,var in enumerate(self.distDict):
          actVar=self.distDict[var]
          poly*=quads[var].evNormPoly(ords[v],quad_pt[v])
          # Note we this assumes standardToActualWeight is linear!
          probNorm*=actVar.probability_norm(quad_pt[v])
        weight=actVar.actual_weight(self.quad.quad_pt_weight[quad_pt])
        self.partCoeffs[quad_pt][ords]=weight*poly*probNorm
        # summing over each [quad_pt]*soln[quad_pt] will give poly_coeff[ords]
    return

  def initializeDistributions(self,availableDist):
    '''generate the instances of the distribution that will be used'''
    self.availableDist = availableDist
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]]
      self.distDict[key].inDistr()
    return
#
#
#
class MonteCarlo(Sampler):
  '''MONTE CARLO Sampler '''
  
  def localInputAndChecks(self,xmlNode):
    if 'limit' not in  xmlNode.attrib.keys(): raise IOError(' Monte Carlo sampling needs the attribute limit (number of samplings)')


  def localGenerateInput(self,model,myInput):
    '''set up self.inputInfo before being sent to the model'''
    for key in self.distDict: self.values[key]=self.distDict[key].distribution.rvs()
    #self.inputInfo['sampledVars'] =  self.values
    # create values dictionary
    self.inputInfo['initial_seed'] = str(self.initSeed)
#
#
#
class Grid(Sampler):
  '''Samples the model on a given (by input) set of points'''
  def __init__(self):
    Sampler.__init__(self) 
    self.gridCoordinate       = [] #the grid point to be used for each distribution (changes at each step)
    self.axisName             = [] #the name of each axis (variable)
    self.gridInfo             = {} # {'name of the variable':('Type',Construction,[values])} gridType: Probability/Value, gridConstruction:Custom/Equal    

  def localInputAndChecks(self,xmlNode):
    self.limit = 1
    for child in xmlNode:
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
           
  def localGenerateInput(self,model,myInput):
    ''' '''
    remainder = self.counter - 1
    total = self.limit+1
    for i in range(len(self.gridCoordinate)):
      varName = self.axisName[i]
      temp, remainder = divmod(remainder, int(total/ len(self.gridInfo[varName][2]) ) )
      total = total/len(self.gridInfo[varName][2])
      if   remainder == 0 and temp != 0 : self.gridCoordinate[i] = temp-1
      elif remainder == 0 and temp == 0 : self.gridCoordinate[i] = len(self.gridInfo[varName][2])-1
      else: self.gridCoordinate[i] = temp
      if self.gridInfo[varName][0]=='CDF':
        self.values[varName] = self.distDict[varName].distribution.ppf(self.gridInfo[varName][2][self.gridCoordinate[i]])
      elif self.gridInfo[varName][0]=='value':
        self.values[varName] = self.gridInfo[varName][2][self.gridCoordinate[i]]

#
#
#
class LHS(Grid):
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
    self.buildLHSSamplingPatter()
  
  def buildLHSSamplingPatter(self):
    tempFillingCheck = [None]*len(self.axisName) #for all variables
    for i in range(len(tempFillingCheck)):
      tempFillingCheck[i] = [None]*(self.pointByVar-1) #intervals are n-points-1
      tempFillingCheck[i][:] = np.random.choice((self.pointByVar-1),size=(self.pointByVar-1),replace=False) #pick a random interval sequence
    self.sampledCoordinate = [None]*(self.pointByVar-1)
    for i in range(self.pointByVar-1):
      self.sampledCoordinate[i] = [None]*len(self.axisName)
      self.sampledCoordinate[i][:] = tempFillingCheck[:][i]

  def localGenerateInput(self,model,myInput):
    j=0
    for varName in self.axisName:
      upper = self.gridInfo[varName][2][self.sampledCoordinate[j][self.counter-2]+1]
      lower = self.gridInfo[varName][2][self.sampledCoordinate[j][self.counter-2]  ]
      j +=1
      intervalFraction = np.random.random()
      coordinate = lower + (upper-lower)*intervalFraction
      if self.gridInfo[varName][0] =='CDF':
        self.values[varName] = self.distDict[varName].distribution.ppf(coordinate)
        self.inputInfo['upper'][varName] = self.distDict[varName].distribution.ppf(max(upper,lower))
        self.inputInfo['lower'][varName] = self.distDict[varName].distribution.ppf(min(upper,lower))
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
    else: return False

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
    if(jobObject.identifier == self.TreeInfo.getroot().tag): endInfo['parent_node'] = self.TreeInfo.getroot()
    else: endInfo['parent_node'] = list(self.TreeInfo.getroot().iter(jobObject.identifier))[0] 
    # get the branchedLevel dictionary 
    branchedLevel = endInfo['parent_node'].get('branchedLevel')
    # Loop of the parameters that have been changed after a trigger gets activated
    for key in endInfo['branch_changed_params']:
      endInfo['n_branches'] = 1 + int(len(endInfo['branch_changed_params'][key]['actual_value']))
      if(len(endInfo['branch_changed_params'][key]['actual_value']) > 1):
        #  Multi-Branch mode => the resulting branches from this parent calculation (just ended)
        # will be more then 2
        # unchanged_pb = probablity (not conditional probability yet) that the event does not occur  
        unchanged_pb = 0.0
        try:
          # changed_pb = probablity (not conditional probability yet) that the event A occurs and the final state is 'alpha' ''' 
          for pb in xrange(len(endInfo['branch_changed_params'][key]['associated_pb'])): unchanged_pb = unchanged_pb + endInfo['branch_changed_params'][key]['associated_pb'][pb]
        except: pass
        if(unchanged_pb <= 1): endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0-unchanged_pb
      else:
        # Two-Way mode => the resulting branches from this parent calculation (just ended) = 2
        if branchedLevel[endInfo['branch_dist']] > len(self.branchProbabilities[endInfo['branch_dist']])-1: pb = 1.0
        else: pb = self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]
        endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0 - pb
        endInfo['branch_changed_params'][key]['associated_pb'] = [pb]
  
    self.branchCountOnLevel = 0
    # set runEnded and running to true and false respectively
    endInfo['parent_node'].set('runEnded',True)
    endInfo['parent_node'].set('running',False)
    endInfo['parent_node'].set('end_time',self.actual_end_time)
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
    parent_cond_pb = 0.0  
    try:
      parent_cond_pb = self.endInfo[index]['parent_node'].get('conditional_pb')
      if not parent_cond_pb: parent_cond_pb = 1.0
    except: parent_cond_pb = 1.0
    # for all the branches the conditional pb is computed 
    # unchanged_cond_pb = Conditional Probability of the branches in which the event has not occurred
    # changed_cond_pb   = Conditional Probability of the branches in which the event has occurred  
    for key in self.endInfo[index]['branch_changed_params']:
       try:
         self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'] = []
         testpb = self.endInfo[index]['branch_changed_params'][key]['unchanged_pb']
         self.endInfo[index]['branch_changed_params'][key]['unchanged_cond_pb'] = parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['unchanged_pb'])
         for pb in xrange(len(self.endInfo[index]['branch_changed_params'][key]['associated_pb'])): self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'].append(parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['associated_pb'][pb]))
       except: pass
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
    try:
      branch_info_tree = ET.parse(filename)
      print('SAMPLER DET   : Done parsing '+filename)
    except:raise IOError ('not able to parse ' + filename)
    root = branch_info_tree.getroot()  
    # Check if end_time and end_ts (time step)  are present... In case store them in the relative working vars 
    try:
      self.actual_end_time = float(root.attrib['end_time'])
      self.actual_end_ts   = int(root.attrib['end_ts'])
    except: pass
    # Store the information in a dictionary that has as keywords the distributions that triggered
    for node in root:
      if node.tag == "Distribution_trigger":
        dist_name = node.attrib['name'].strip()
        self.actualBranchInfo[dist_name] = {}
        for child in node:
          self.actualBranchInfo[dist_name][child.text.strip()] = {'varType':child.attrib['type'].strip(),'actual_value':child.attrib['actual_value'].strip().split(),'old_value':child.attrib['old_value'].strip()}
          try:
            as_pb = child.attrib['probability'].strip().split()
            self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'] = []
            #self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb)) 
            for index in range(len(as_pb)): self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb[index]))
          except: pass
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
    if self.counter > 1:
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
      for i in xrange(n_branches):
        del self.inputInfo
        self.counter += 1
        self.branchCountOnLevel += 1
        branchedLevel = copy.deepcopy(branchedLevelG)
        # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
        rname = copy.deepcopy(endInfo['parent_node'].get('name') + '-' + str(self.branchCountOnLevel))
        # create a subgroup that will be appended to the parent element in the xml tree structure 
        subGroup = ET.Element(rname)
        subGroup.set('parent', endInfo['parent_node'].get('name'))
        subGroup.set('name', rname)
        # cond_pb_un = conditional probability event not occur
        # cond_pb_c  = conditional probability event/s occur/s
        cond_pb_un = 0.0
        cond_pb_c  = 0.0
        # Loop over  branch_changed_params (events) and start storing information, 
        # such as conditional pb, variable values, into the xml tree object
        for key in endInfo['branch_changed_params'].keys():
          subGroup.set('branch_changed_param',key)
          if self.branchCountOnLevel != 1:
            subGroup.set('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2]))
            subGroup.set('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2]))
            try: cond_pb_c = cond_pb_c + copy.deepcopy(endInfo['branch_changed_params'][key]['changed_cond_pb'][self.branchCountOnLevel-2])
            except: pass
          else:
            subGroup.set('branch_changed_param_value',copy.deepcopy(endInfo['branch_changed_params'][key]['old_value']))
            subGroup.set('branch_changed_param_pb',copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_pb']))
            try:cond_pb_un =  cond_pb_un + copy.deepcopy(endInfo['branch_changed_params'][key]['unchanged_cond_pb'])
            except: pass
        # add conditional probability
        if self.branchCountOnLevel != 1: subGroup.set('conditional_pb',copy.deepcopy(cond_pb_c))
        else: subGroup.set('conditional_pb',copy.deepcopy(cond_pb_un)) 
        # add initiator distribution info, start time, etc. 
        subGroup.set('initiator_distribution',copy.deepcopy(endInfo['branch_dist']))
        subGroup.set('start_time', copy.deepcopy(endInfo['parent_node'].get('end_time')))
        # initialize the end_time to be equal to the start one... It will modified at the end of this branch
        subGroup.set('end_time', copy.deepcopy(endInfo['parent_node'].get('end_time')))
        # add the branchedLevel dictionary to the subgroup
        if self.branchCountOnLevel != 1: branchedLevel[endInfo['branch_dist']] = branchedLevel[endInfo['branch_dist']] - 1
        
        subGroup.set('branchedLevel', copy.deepcopy(branchedLevel))
        # branch calculation info... running, queue, etc are set here
        subGroup.set('runEnded',False)
        subGroup.set('running',False)
        subGroup.set('queue',True)
#        subGroup.set('restartFileRoot',endInfo['restartRoot'])
        # Append the new branch (subgroup) info to the parent_node in the xml tree object
        endInfo['parent_node'].append(subGroup)
        
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
      rname = self.TreeInfo.getroot().tag 
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
      if self.print_end_xml:
        self.TreeInfo.write(self.name + "_xml_output_summary")
      return None
    else:
      # Pop out the first input in queue
      jobInput = self.RunQueue['queue'].pop(0)
      id       = self.RunQueue['identifiers'].pop(0)
      #set running flags in self.TreeInfo
      root = self.TreeInfo.getroot()
      # Update the run information flags
      if (root.tag == id):
        root.set('runEnded',str(False))
        root.set('running',str(True)) 
        root.set('queue',str(False))
      else:
        subElm = list(root.iter(id))[0]
        if(subElm is not None):
          subElm.set('runEnded',str(False))
          subElm.set('running',str(True))
          subElm.set('queue',str(False))

    return jobInput

  def localGenerateInput(self,model,myInput):
    '''
    Function used to generate a input. In this case it just calls 
    the function '__getQueueElement' to retrieve the first input
    in the queue
    @ In, model    : Model object instance
    @ In, myInput  : Original input files
    @ Out, input   : First input in the queue 
    '''
    if self.counter <= 1:
      # If first branch input, create the queue 
      self.__createRunningQueue(model, myInput)
    # retrieve the input from the queue
    input = self.__getQueueElement()
    if not input:
      # If no inputs are present in the queue => a branch is finished 
      print("SAMPLER DET   : A Branch ended!!!!")
    return input

  def localInputAndChecks(self,xmlNode):

    try:    self.print_end_xml = (xmlNode.attrib['print_end_xml'].lower() in ['true','t','yes','si','y'])
    except KeyError: self.print_end_xml = False

    # retrieve max simulation time, if inputted
    try:    self.maxSimulTime = children.attrib['maxSimulationTime']
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
          if   childChild.attrib.has_key('ProbabilityThresholds'): 
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
          elif childChild.attrib.has_key('ValueThresholds'):
            self.branchValues[childChild.text] = [float(x) for x in childChild.attrib['ValueThresholds'].split()]
            self.branchValues[childChild.text].sort(key=float)
            for index in range(len(sorted(self.branchValues[childChild.text], key=float))):
              if sorted(self.branchValues[childChild.text], key=float).count(sorted(self.branchValues[childChild.text], key=float)[index]) > 1:
                print("SAMPLER DET   : ERROR -> In distribution " + str(childChild.text) + " the Threshold " + str(sorted(self.branchValues[childChild.text], key=float)[index])+" appears multiple times!!")
                error_found = True
            # the associated CDF of the branchValues are computed in the initialize method (called right before the Sampler gets used)
          else:      
            raise IOError('The DynamicEventTree sampler needs that all distributions have either Probability Thresholds or Value Thresholds!!!!')
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
    elm = ET.Element(self.name + '_1')
  
    elm.set('name', self.name + '_1')
    elm.set('start_time', str(0.0))
    # Initialize the end_time to be equal to the start one... 
    # It will modified at the end of each branch
    elm.set('end_time', str(0.0))
    elm.set('runEnded',str(False))
    elm.set('running',str(True))
    elm.set('queue',str(False))   
    # The dictionary branchedLevel is stored in the xml tree too. That's because
    # the advancement of the thresholds must follow the tree structure
    elm.set('branchedLevel', self.branchedLevel[0])
    # Here it is stored all the info regarding the DET => we create the info for all the
    # branchings and we store them
    self.TreeInfo = ET.ElementTree(elm) 
    
    if (len(self.branchProbabilities.keys()) != 0):
      #compute the associated invCDF values
      for key in self.branchProbabilities.keys():
        self.branchValues[key] = [copy.deepcopy(self.distDict[key].distribution.ppf(float(self.branchProbabilities[key][index]))) for index in range(len(self.branchProbabilities[key]))]
    else:
      #compute the associated CDF values
      for key in self.branchValues.keys():
        self.branchProbabilities[key] = [copy.deepcopy(self.distDict[key].distribution.ppf(float(self.branchValues[key][index]))) for index in range(len(self.branchValues[key]))]
    return

'''
 Interface Dictionary (factory) (private)
'''
__base = 'Sampler'
__interFaceDict = {}
__interFaceDict['MonteCarlo'            ] = MonteCarlo
__interFaceDict['DynamicEventTree'      ] = DynamicEventTree
__interFaceDict['StochasticCollocation' ] = StochasticCollocation
__interFaceDict['LHS'                   ] = LHS
__interFaceDict['Grid'                  ] = Grid
__knownTypes = __interFaceDict.keys()

def knonwnTypes():
  return __knownTypes

def returnInstance(Type):
  '''
  function used to generate a Sampler class
  @ In, Type : Sampler type
  @ Out,Instance of the Specialized Sampler class
  '''
  try: return __interFaceDict[Type]()
  except: raise NameError('not known '+__base+' type '+Type)

def optionalInputs(Type):
  pass

def mandatoryInputs(Type):
  pass




