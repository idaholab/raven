'''
Created on May 8, 2013

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range

import sys
import time
import Datas
from BaseType import BaseType
import xml.etree.ElementTree as ET
import os
#import Queue
import copy
import numpy as np
try:
  import cPickle as pk
except:
  import pickle as pk
import Quadrature
import Distributions
from itertools import product as iterproduct

class Sampler(BaseType):
  ''' this is the base class for samplers'''

  def __init__(self):
    BaseType.__init__(self)
    # Counter of the samples performed
    self.counter = 0
    # maximum number of Samples (for example, Montecarlo = Number of Histories to run, DET = Unlimited)
    self.limit   = sys.maxsize 
    # Working directory (Path of the directory in which all the outputs,etc. are stored)
    self.workingDir = ""
    #  Dictionary of sampling variables.
    #  key=feature to be sampled, value = ['type of distribution to be used', 'name of the distribution']
    self.toBeSampled = {}  
    # Contains the instance of the distribution to be used, it is created every time the sampler is initialize
    self.distDict    = {}  

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    for child in xmlNode:
      sampleVar = str(child.text).split(':')
      #self.toBeSampled[sampleVar[0]] = [child.attrib['type'],child.attrib['distName']]
      self.toBeSampled[child.text] = [child.attrib['type'],child.attrib['distName']]
      # we try to append the position =>  if the user wants to add a position 
      #(i.e. word number in a RELAP5 card or array position for RAVEN), the sampledVariable would be 
      # variableName:position (example wolf:6)
      #try: self.toBeSampled[sampleVar[0]].append(sampleVar[1])
      #except: self.toBeSampled[child.text].append(0)   #append a default value of the position

  def addInitParams(self,tempDict):
    '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    for value in self.toBeSampled.items():
      tempDict[value[0]] = value[1][0]+'-'+value[1][1]
    tempDict['limit' ]        = self.limit

  def addCurrentSetting(self,tempDict):
    '''
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    tempDict['counter' ] = self.counter

  def initialize(self):
    '''
    Function used to initialize the Latin Hyper Cube Sampler
    @ In, None
    @ Out, None
    '''
    self.counter = 0

  def amIreadyToProvideAnInput(self):
    '''this check if the sampler is ready to generate a new input
       it might not since it is waiting for more information or has 
       run out of limit, depends from the type of sampler'''
    if(self.counter <= self.limit): return True
    else: return False

  def fillDistribution(self,availableDist):
    '''generate the instances of the distribution that will be used'''
    for key in self.toBeSampled.keys():
      self.distDict[key] = availableDist[self.toBeSampled[key][1]]
      self.distDict[key].inDistr()
    return

  def finalizeActualSampling(self,jobObject,model,myInput):
    ''' This function is used by samplers that need to finalize the just ended sample
        For example, In a MonteCarlo simulation it can be used to sample parameters at 
        the end of a previous calculation in order to make the Model aware of the parameters
        it needs to change. For a Dynamic Event Tree case, this function can be used to retrieve
        the information from the just finished run of a branch in order to retrieve, for example,
        the distribution name that caused the trigger, etc.
    '''
    pass

  def generateInput(self):
    '''override this method to place your input generation'''
    raise IOError('This sampler has not an input generation?!?!')
    return

  def generateInputBatch(self,myInput,model,batchSize,projector=None):
    '''for the first set of run a set of run are started in // so we need more than one input '''
    newInputs = []
    while self.amIreadyToProvideAnInput() and (self.counter < batchSize):
      if projector==None: newInputs.append(self.generateInput(model,myInput))
      else             : newInputs.append(self.generateInput(model,myInput,projector))
    return newInputs
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

  def initialize(self):
    Sampler.initialize(self)
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

  def generateInput(self,model,myInput):
    '''
    Function used to generate an input.
    It returns the model.createNewInput() passing into it the type of sampler,
    the values to be used and the some add info in the values dict
    @ In, model: Model object instance
    @ Out, myInputs: Original input files
    '''
    print('generate input model:',model)
    self.counter+=1
    quad_pts=self.quad.quad_pts[self.counter-1]
    quad_pt_index = self.quad.quad_pt_index[quad_pts]
    values={'prefix'        :str(self.counter),
            'quad_pts'      :quad_pts,
            'partial coeffs':self.partCoeffs[quad_pts].values(),
            'exp order'    :self.quad.quad_pt_index[quad_pts]}
    #values={}
    #values['prefix']={'counter'       :str(self.counter),
    #                  'quad pts'      :str(quad_pts),
    #                  'partial coeffs':str(self.partCoeffs[quad_pts])}
    #values['prefix']=(('counter'   ,'quad pts','partial coeffs'),
    #                  (self.counter, quad_pts,str(self.partCoeffs[quad_pts].values())))
    values['vars']={}
    # TODO would be beneficial to pass the orders of quad pt, too?
    for var in self.distDict.keys():
      values['vars'][var]=self.distDict[var].actual_point(\
          quad_pts[self.quad.dict_quads[self.quad.quads[var]]])
      #print('run',self.counter,'for var '+var+' set value',values['vars'][var])
    return model.createNewInput(myInput,self.type,**values)

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

  def fillDistribution(self,availableDist):
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
  '''
  MONTE CARLO Sampler 
  '''
  def __init__(self):
    Sampler.__init__(self)
    self.limit       = 0        #maximum number of samples it will perform every time it is used
    self.init_seed   = 0

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    try: self.init_seed    = xmlNode.attrib['initial_seed']
    except: self.init_seed = 0 
    try: self.limit    = int(xmlNode.attrib['limit'])
    except: raise IOError(' Monte Carlo sampling needs the attribute limit (number of samplings)')
    #  stores variables for random sampling  added by nieljw to allow for RELAP5 
    self.variables={}
    Sampler.readMoreXML(self, xmlNode)

  def addInitParams(self,tempDict):
    '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    Sampler.addInitParams(self,tempDict)
    tempDict['initial seed' ] = self.init_seed

  def generateInput(self,model,myInput):
    '''
    Function used to generate an input 
    @ In, model: Model object instance
    @ Out, myInputs: Original input files
    '''
    self.counter += 1
    #evaluate the distributions and fill values{}
    sampledVar = {}
    for key in self.distDict: sampledVar[key]=self.distDict[key].distribution.rvs()
    # create values dictionary
    values = {'prefix':str(self.counter),'initial_seed':str(self.init_seed),'sampledVars':sampledVar}
       #values[key]={'value':value,'position':self.toBeSampled[key][2]}
    return model.createNewInput(myInput,self.type,**values)
#
#
#
class LatinHyperCube(Sampler):
  '''
  LATIN HYPER CUBE Sampler 
  implement a latin hyper cube approach only with random picking of the intervals'''

  def __init__(self):
    Sampler.__init__(self)
    # maximum number of sampler it will perform every time it is used 
    self.limit        = 0
    # is a list that for each point in the grid return a dictionary of the 
    #  distributions where values are the bounds in terms of the random variate 
    self.grid         = []       #
    self.init_seed    = 0

  def addInitParams(self,tempDict):
    '''
    Function adds the initial parameter in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    Sampler.addInitParams(self,tempDict)
    tempDict['initial seed' ] = self.init_seed

  def addCurrentSetting(self,tempDict):
    '''
    Function adds the current settings in a temporary dictionary
    @ In, tempDict
    @ Out, tempDict 
    '''
    i = 0
    for distribution in self.distDict.keys():
      tempDict['interval '+ int(i) + ', distribution ' +distribution+' is in range'] = self.grid[i].distBounds[distribution]

  def initialize(self):
    '''
    Function used to initialize the Latin Hyper Cube Sampler
    @ In, None
    @ Out, None
    '''
    # Initialize Sampler 
    Sampler.initialize(self)
    #Initialize the grid to have a size equal to the max number of samplings 
    self.grid = [None]*self.limit
    # Grep number of distributions for sampling 
    nDimension = len(self.distDict)
    takenGlobal = np.zeros((self.limit,nDimension),ndmin=2,dtype=int)
    # Create a list of the distributions' names 
    distList = list(self.distDict.keys())
    # Initialize the grid dictionary to a set of 
    # dictionaries with keys = distribution keys and 
    # a value container size of 2 (Lower and Upper bounds)
    for i in range(self.grid):
      self.grid[i] = dict.fromkeys(self.distDict.keys(),[None]*2)
    # Construct the grid
    for j in range(self.limit):
      for i in range(nDimension):
        placed = False
        while placed == False:
          indexInterval = int(np.random.rand(1)*self.limit)
          if takenGlobal[indexInterval][i] == 0:
            takenGlobal[indexInterval][i] = 1
            distName = distList[i]
            # if equally spaced do not use ppt
            lowerBound = self.distDict[distName].ppt(float((indexInterval-1)/self.limit))
            upperBound = self.distDict[distName].ppt(float((indexInterval)/self.limit))
            self.grid[j].distBounds[distName] = [lowerBound,upperBound]

  def generateInput(self,model,myInput):
    '''
    Function used to generate an input.
    It returns the model.createNewInput() passing into it the type of sampler,
    the values to be used and the some add info in the values dict
    @ In, model: Model object instance
    @ Out, myInputs: Original input files
    '''
    # increase the counter
    self.counter += 1
    # Fill the values dictionary that will be passed into the model in order to create an input
    # In this dictionary the info for changing the original input is stored    
    values = {'prefix':str(self.counter),'initial_seed':str(self.init_seed)}
    # evaluate the distributions and put the results into values
    for key in self.distDict:
      upper = self.grid[self.counter][key][1]
      lower = self.grid[self.counter][key][1]
      values[key] = [self.distDict[key].distribution.rvsWithinbounds(lower,upper),lower,upper]
    return model.createNewInput(myInput,self.type,**values)
#
#
#
class EquallySpaced(Sampler):
  '''
  EQUALLY SPACED Sampler - TO BE IMPLEMENTED 
  '''
  pass
#
#
#
class DynamicEventTree(Sampler):
  '''
  DYNAMIC EVEN TREE Sampler - "ANalysis of Dynamic REactor Accident evolution" module (DET      ) :D
  '''
  def __init__(self):
    Sampler.__init__(self)

    # (optional) if not present, the sampler will not change the relative keyword in the input file
    self.maxSimulTime            = None  

    # print the xml tree representation of the dynamic event tree calculation
    # see variable 'self.TreeInfo'
    self.print_end_xml           = False 
    
    # Dictionary of the probability bins for each distribution that have been 
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    self.branchProbabilities     = {}

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
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0):
      return True
    else:
      return False

  def finalizeActualSampling(self,jobObject,model,myInput):
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
    endInfo = {}
    endInfo['end_time']               = self.actual_end_time
    endInfo['end_ts']                 = self.actual_end_ts
    endInfo['branch_dist']            = list(self.actualBranchInfo.keys())[0]
    endInfo['branch_changed_params']  = self.actualBranchInfo[endInfo['branch_dist']]
    # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    if(jobObject.identifier == self.TreeInfo.getroot().tag):
      endInfo['parent_node'] = self.TreeInfo.getroot()
    else:
      endInfo['parent_node'] = list(self.TreeInfo.getroot().iter(jobObject.identifier))[0]
    
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
           for pb in xrange(len(endInfo['branch_changed_params'][key]['associated_pb'])):
             unchanged_pb = unchanged_pb + endInfo['branch_changed_params'][key]['associated_pb'][pb]
         except:
          pass
         if(unchanged_pb <= 1):
           endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0-unchanged_pb
      else:
        # Two-Way mode => the resulting branches from this parent calculation (just ended) = 2
        if branchedLevel[endInfo['branch_dist']] > len(self.branchProbabilities[endInfo['branch_dist']])-1:
          pb = 1.0
        else:
          pb = self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]
        endInfo['branch_changed_params'][key]['unchanged_pb'] = 1.0 - pb
        endInfo['branch_changed_params'][key]['associated_pb'] = [pb]
  
    self.branchCountOnLevel = 0
    # set runEnded and running to true and false respectively
    endInfo['parent_node'].set('runEnded',True)
    endInfo['parent_node'].set('running',False)
    endInfo['parent_node'].set('end_time',self.actual_end_time)
    # The branchedLevel counter is updated
    if branchedLevel[endInfo['branch_dist']] < len(self.branchProbabilities[endInfo['branch_dist']]):
      branchedLevel[endInfo['branch_dist']]       += 1
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
    if not index:
      index = len(self.endInfo)-1
    # parent_cond_pb = associated conditional probability of the Parent branch 
    parent_cond_pb = 0.0  
    try:
      parent_cond_pb = self.endInfo[index]['parent_node'].get('conditional_pb')
      if not parent_cond_pb:
        parent_cond_pb = 1.0
    except:
      parent_cond_pb = 1.0
    # for all the branches the conditional pb is computed 
    # unchanged_cond_pb = Conditional Probability of the branches in which the event has not occurred
    # changed_cond_pb   = Conditional Probability of the branches in which the event has occurred  
    for key in self.endInfo[index]['branch_changed_params']:
       try:
         self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'] = []
         testpb = self.endInfo[index]['branch_changed_params'][key]['unchanged_pb']
         self.endInfo[index]['branch_changed_params'][key]['unchanged_cond_pb'] = parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['unchanged_pb'])
         for pb in xrange(len(self.endInfo[index]['branch_changed_params'][key]['associated_pb'])):
           self.endInfo[index]['branch_changed_params'][key]['changed_cond_pb'].append(parent_cond_pb*float(self.endInfo[index]['branch_changed_params'][key]['associated_pb'][pb]))
       except:
         pass
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
    if out_base:
      filename = out_base + "_actual_branch_info.xml"
    else:
      filename = "actual_branch_info.xml"
    if not os.path.isabs(filename):
      filename = os.path.join(self.workingDir,filename)
    if not os.path.exists(filename):
      print('SAMPLER DET   : branch info file ' + filename +' has not been found. => No Branching.')
      branch_present = False
      return branch_present
    
    # Parse the file and create the xml element tree object
    try:
      branch_info_tree = ET.parse(filename)
      print('SAMPLER DET   : Done parsing '+filename)
    except:
      #branch_info_tree = ET.parse(filename) #This could cause a second exception
      raise IOError ('not able to parse ' + filename)
    root = branch_info_tree.getroot()
    
    # Check if end_time and end_ts (time step)  are present... In case store them in the relative working vars 
    try:
      self.actual_end_time = float(root.attrib['end_time'])
      self.actual_end_ts   = int(root.attrib['end_ts'])
    except:
      pass
    # Store the information in a dictionary that has as keywords the distributions that triggered
    
    for node in root:
      if node.tag == "Distribution_trigger":
        dist_name = node.attrib['name'].strip()
        self.actualBranchInfo[dist_name] = {}
        for child in node:
          self.actualBranchInfo[dist_name][child.text.strip()] = {}
          self.actualBranchInfo[dist_name][child.text.strip()]['varType'] = child.attrib['type'].strip()
          #self.actualBranchInfo[dist_name][child.text.strip()]['actual_value'] = []
          #self.actualBranchInfo[dist_name][child.text.strip()]['actual_value'].append(child.attrib['actual_value'].strip())
          self.actualBranchInfo[dist_name][child.text.strip()]['actual_value'] = child.attrib['actual_value'].strip().split()
          self.actualBranchInfo[dist_name][child.text.strip()]['old_value'] = child.attrib['old_value'].strip()
          try:
            as_pb = child.attrib['probability'].strip().split()
            self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'] = []
            #self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb)) 
            for index in range(len(as_pb)):
              self.actualBranchInfo[dist_name][child.text.strip()]['associated_pb'].append(float(as_pb[index]))
          except:
            pass
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
      for i in xrange(n_branches):
        self.counter += 1
        self.branchCountOnLevel += 1
        branchedLevel = copy.deepcopy(branchedLevelG)
        # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
        rname = endInfo['parent_node'].get('name') + '-' + str(self.branchCountOnLevel)
        
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
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['actual_value'][self.branchCountOnLevel-2])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['associated_pb'][self.branchCountOnLevel-2])
            try:
              cond_pb_c = cond_pb_c + endInfo['branch_changed_params'][key]['changed_cond_pb'][self.branchCountOnLevel-2] 
            except:
              pass
          else:
            subGroup.set('branch_changed_param_value',endInfo['branch_changed_params'][key]['old_value'])
            subGroup.set('branch_changed_param_pb',endInfo['branch_changed_params'][key]['unchanged_pb'])
            try:
              cond_pb_un =  cond_pb_un + endInfo['branch_changed_params'][key]['unchanged_cond_pb']
            except:
              pass

        # add conditional probability
        if self.branchCountOnLevel != 1:
          subGroup.set('conditional_pb',cond_pb_c)
        else:
          subGroup.set('conditional_pb',cond_pb_un)
          
        # add initiator distribution info, start time, etc. 
        subGroup.set('initiator_distribution',endInfo['branch_dist']) 
        subGroup.set('start_time', endInfo['parent_node'].get('end_time'))
        # initialize the end_time to be equal to the start one... It will modified at the end of this branch
        subGroup.set('end_time', endInfo['parent_node'].get('end_time'))
        # add the branchedLevel dictionary to the subgroup
        if self.branchCountOnLevel != 1:
          branchedLevel[endInfo['branch_dist']] = branchedLevel[endInfo['branch_dist']] - 1
        
        subGroup.set('branchedLevel', branchedLevel)
        # branch calculation info... running, queue, etc are set here
        subGroup.set('runEnded',False)
        subGroup.set('running',False)
        subGroup.set('queue',True)
#        subGroup.set('restartFileRoot',endInfo['restartRoot'])
        # Append the new branch (subgroup) info to the parent_node in the xml tree object
        endInfo['parent_node'].append(subGroup)
        
        # Fill the values dictionary that will be passed into the model in order to create an input
        # In this dictionary the info for changing the original input is stored
        values = {'prefix':rname,'end_ts':endInfo['end_ts'],
                  'branch_changed_param':[subGroup.get('branch_changed_param')],
                  'branch_changed_param_value':[subGroup.get('branch_changed_param_value')],
                  'conditional_prb':[subGroup.get('conditional_pb')],
                  'start_time':endInfo['parent_node'].get('end_time'),
                  'parent_id':subGroup.get('parent')}

        # Check if the distribution that just triggered hitted the last probability threshold . 
        #  In this case there is not a probability threshold that needs to be added in the input
        #  for this particular distribution 
        if not (branchedLevel[endInfo['branch_dist']] >= len(self.branchProbabilities[endInfo['branch_dist']])):
          values['initiator_distribution'] = [endInfo['branch_dist']]
          values['PbThreshold']            = [self.branchProbabilities[endInfo['branch_dist']][branchedLevel[endInfo['branch_dist']]]]
        #  For the other distributions, we put the unbranched thresholds
        #  Before adding these thresholds, check if the keyword 'initiator_distribution' is present... 
        #  (In the case the previous if statement is true, this keyword is not present yet
        #  Add it otherwise 
        if not ('initiator_distribution' in values.keys()):
          values['initiator_distribution'] = []
          values['PbThreshold'           ] = []
        # Add the unbranched thresholds
        for key in self.branchProbabilities.keys():
          if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])):
            values['initiator_distribution'].append(key.encode())
        for key in self.branchProbabilities.keys():
          if not (key in endInfo['branch_dist']) and (branchedLevel[key] < len(self.branchProbabilities[key])):
            values['PbThreshold'].append(self.branchProbabilities[key][branchedLevel[key]])
        # Call the model function "createNewInput" with the "values" dictionary just filled.
        # Add the new input path into the RunQueue system  
        self.RunQueue['queue'].append(copy.deepcopy(model.createNewInput(myInput,self.type,**values)))
        self.RunQueue['identifiers'].append(values['prefix'])
        del values
        del branchedLevel

    else:
      # We construct the input for the first DET branch calculation'
      # Increase the counter
      self.counter += 1
      # The root name of the xml element tree is the starting name for all the branches 
      # (this root name = the user defined sampler name)
      rname = self.TreeInfo.getroot().tag 
      # Get the initial branchedLevel dictionary (=> the list gets empty)
      branchedLevelG = copy.deepcopy(self.branchedLevel.pop(0))
      branchedLevel = copy.deepcopy(branchedLevelG)
      # Fill th values dictionary in
      values = {'prefix':rname}
      values['initiator_distribution']     = []
      values['PbThreshold']                = []
      values['branch_changed_param']       = [b'None']
      values['branch_changed_param_value'] = [b'None']
      values['start_time']                 = b'Initial'
      values['end_ts']                     = 0
      values['parent_id']                  = b'root'
      values['conditional_prb']            = [1.0]
      for key in self.branchProbabilities.keys():
        values['initiator_distribution'].append(key.encode())
      for key in self.branchProbabilities.keys():  
        values['PbThreshold'].append(self.branchProbabilities[key][branchedLevel[key]])
      if(self.maxSimulTime): values['end_time'] = self.maxSimulTime
      # Call the model function "createNewInput" with the "values" dictionary just filled.
      # Add the new input path into the RunQueue system  
      newInputs = model.createNewInput(myInput,self.type,**values)
      self.RunQueue['queue'].append(newInputs)
      self.RunQueue['identifiers'].append(values['prefix'])
      del values
      del newInputs
      del branchedLevel
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

  def generateInput(self,model,myInput):
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

  def readMoreXML(self,xmlNode):
    '''
    Function to read the portion of the xml input that belongs to this specialized class
    and initialize some stuff based on the inputs got
    @ In, xmlNode    : Xml element node
    @ Out, None
    '''
    elm = ET.Element(xmlNode.attrib['name'] + '_1')
    try:
      flag = ""
      flag = xmlNode.attrib['print_end_xml']
      self.print_end_xml = (flag.lower() in ['true','t','yes','si','y'])
    except:
      self.print_end_xml = False
      
    elm.set('name', xmlNode.attrib['name'] + '_1')
    elm.set('start_time', str(0.0))
    # Initialize the end_time to be equal to the start one... 
    # It will modified at the end of each branch
    elm.set('end_time', str(0.0))
    elm.set('runEnded',str(False))
    elm.set('running',str(True))
    elm.set('queue',str(False))

    # Read branching settings
    children = xmlNode.find("BranchingSettings")
    try: self.maxSimulTime = children.attrib['maxSimulationTime']
    except: self.maxSimulTime = None
    Sampler.readMoreXML(self,children)
    branchedLevel = {}
    error_found = False
    for child in children:
      bv = child.attrib['BranchProbs']
      bvalues = [float(x) for x in bv.split()]
      bvalues.sort()
      self.branchProbabilities[child.attrib['distName']] = bvalues
      branchedLevel[child.attrib['distName']]       = 0
      #error check
      if max(bvalues) > 1:
        print("SAMPLER DET   : ERROR -> One of the Thresholds for distribution " + str(child.attrib['distName']) + " is > 1")
        error_found = True
      templist = sorted(bvalues, key=float)
      for index in range(len(templist)):
        if templist.count(templist[index]) > 1:
          print("SAMPLER DET   : ERROR -> In distribution " + str(child.attrib['distName']) + " the Threshold " + str(templist[index])+" appears multiple times!!")
          error_found = True
    if error_found: raise IOError("In Sampler " + self.name+' ERRORS have been found!!!' )

    # Append the branchedLevel dictionary in the proper list
    self.branchedLevel.append(branchedLevel)
    # The dictionary branchedLevel is stored in the xml tree too. That's because
    # the advancement of the thresholds must follow the tree structure
    elm.set('branchedLevel', branchedLevel)
    # Here it is stored all the info regarding the DET => we create the info for all the
    # branchings and we store them
    self.TreeInfo = ET.ElementTree(elm)

def returnInstance(Type):
  '''
  function used to generate a Sampler class
  @ In, Type : Sampler type
  @ Out,Instance of the Specialized Sampler class
  '''
  base = 'Sampler'
  InterfaceDict = {}
  InterfaceDict['MonteCarlo'            ] = MonteCarlo
  InterfaceDict['LatinHyperCube'        ] = LatinHyperCube
  InterfaceDict['EquallySpaced'         ] = EquallySpaced
  InterfaceDict['DynamicEventTree'      ] = DynamicEventTree
  InterfaceDict['StochasticCollocation' ] = StochasticCollocation
  try: return InterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)
