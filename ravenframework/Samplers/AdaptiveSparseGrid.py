# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  This module contains the Adaptive Stochastic Collocation sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from talbpw
"""
# External Modules----------------------------------------------------------------------------------
import sys
import copy
from operator import mul
from functools import reduce
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from .SparseGridCollocation import SparseGridCollocation
from .AdaptiveSampler import AdaptiveSampler
from ..utils import utils
from ..utils import InputData, InputTypes
from .. import Quadratures
from .. import IndexSets
# Internal Modules End------------------------------------------------------------------------------

# get appropriate version of pickle
if sys.version_info.major > 2:
  import pickle
else:
  import cPickle as pickle

class AdaptiveSparseGrid(SparseGridCollocation, AdaptiveSampler):
  """
   Adaptive Sparse Grid Collocation sampling strategy
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(AdaptiveSparseGrid, cls).getInputSpecification()

    convergenceInput = InputData.parameterInputFactory("Convergence", contentType=InputTypes.StringType)
    convergenceInput.addParam("target", InputTypes.StringType, True)
    convergenceInput.addParam("maxPolyOrder", InputTypes.IntegerType)
    convergenceInput.addParam("persistence", InputTypes.IntegerType)

    inputSpecification.addSub(convergenceInput)

    inputSpecification.addSub(InputData.parameterInputFactory("logFile"))
    inputSpecification.addSub(InputData.parameterInputFactory("maxRuns", contentType=InputTypes.IntegerType))

    targetEvaluationInput = InputData.parameterInputFactory("TargetEvaluation", contentType=InputTypes.StringType)
    targetEvaluationInput.addParam("type", InputTypes.StringType)
    targetEvaluationInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(targetEvaluationInput)

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    super().__init__()
    # identification
    self.type                    = 'AdaptiveSparseGridSampler'
    self.printTag                = self.type
    # assembler objects
    self.solns                   = None   # TimePointSet of solutions -> assembled
    self.ROM                     = None   # eventual final ROM object
    # input parameters
    self.maxPolyOrder            = 0      # max size of polynomials to allow
    self.persistence             = 0      # number of forced iterations, default 2
    self.convType                = None   # convergence criterion to use
    self.logFile                 = None   # file to print log to, optional
    # convergence/training tools
    self.expImpact               = {}     # dict of potential included polynomials and their estimated impacts, [target][index]
    self.actImpact               = {}     # dict of included polynomials and their current impact, [target][index] = impact
    self.sparseGrid              = None   # current sparse grid
    self.oldSG                   = None   # previously-accepted sparse grid
    self.error                   = 0      # estimate of percent of moment calculated so far
    self.logCounter              = 0      # when printing the log, tracks the number of prints

    # solution storage
    self.neededPoints            = []     # queue of points to submit
    self.submittedNotCollected   = []     # list of points submitted but not yet collected and used
    self.pointsNeededToMakeROM   = set()  # list of distinct points needed in this process
    self.unfinished              = 0      # number of runs still running when convergence complete
    self.batchDone               = True   # flag for whether jobHandler has complete batch or not
    self.done                    = False  # flipped when converged
    self.newSolutionSizeShouldBe = None   # used to track and debug intended size of solutions
    self.inTraining              = set()  # list of index set points for whom points are being run

    # attributes set later
    self.maxRuns                 = None
    self.convValue               = None
    self.targets                 = None
    self.converged               = None

    # convergence study -> currently suspended since it doesn't follow RAVEN I/O protocol.
    # self.doingStudy              = False  #true if convergenceStudy node defined for sampler
    # self.studyFileBase           = 'out_' #can be replaced in input, not used if not doingStudy
    # self.studyPoints             = []     #list of ints, runs at which to record a state
    # self.studyPickle             = False  #if true, dumps ROM to pickle at each step

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    SparseGridCollocation.localInputAndChecks(self,xmlNode, paramInput)
    if 'Convergence' not in list(c.tag for c in xmlNode):
      self.raiseAnError(IOError,'Convergence node not found in input!')
    convnode  = xmlNode.find('Convergence')
    logNode   = xmlNode.find('logFile')
    # studyNode = xmlNode.find('convergenceStudy')
    self.convType     = convnode.attrib.get('target', 'variance')
    self.maxPolyOrder = int(convnode.attrib.get('maxPolyOrder',10))
    self.persistence  = int(convnode.attrib.get('persistence',2))
    maxRunsNode = xmlNode.find('maxRuns')
    if maxRunsNode is not None:
      self.maxRuns = int(maxRunsNode.text)
    else:
      self.maxRuns = None

    self.convValue    = float(convnode.text)
    if logNode is not None:
      self.logFile = logNode.text
    if self.maxRuns is not None:
      self.maxRuns = int(self.maxRuns)
    # studyNode for convergence study is removed for now, since it doesn't follow the RAVEN pattern of I/O
    #   since it writes directy to a file. However, it could be configured to work in the future, so leaving
    #   it for now.
    #if studyNode is not None:
    #  self.doingStudy = True
    #  self.studyPoints = studyNode.find('runStatePoints').text
    #  filebaseNode = studyNode.find('baseFilename')
    #  self.studyPickle = studyNode.find('pickle') is not None
    #  if filebaseNode is None:
    #    self.raiseAWarning('No baseFilename specified in convergenceStudy node!  Using "%s"...' %self.studyFileBase)
    #  else:
    #    self.studyFileBase = studyNode.find('baseFilename').text
    #  if self.studyPoints is None:
    #    self.raiseAnError(IOError,'convergenceStudy node was included, but did not specify the runStatePoints node!')
    #  else:
    #    try:
    #      self.studyPoints = list(int(i) for i in self.studyPoints.split(','))
    #    except ValueError as e:
    #      self.raiseAnError(IOError,'Convergence state point not recognizable as an integer!',e)
    #    self.studyPoints.sort()

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    # this model doesn't use restarts, it uses the TargetEvaluation
    if self.restartData is not None:
      self.raiseAnError(IOError, 'AdaptiveSparseGrid does not use Restart nodes! Try TargetEvaluation instead.')
    # obtain the DataObject that contains evaluations of the model
    self.solns = self.assemblerDict['TargetEvaluation'][0][3]
    # set a pointer to the GaussPolynomialROM object
    SVL = self.readFromROM()
    self.targets = SVL.target # the output space variables
    # initialize impact dictionaries by target
    self.expImpact = {key: dict({}) for key in self.targets}
    self.actImpact = {key: dict({}) for key in self.targets}

    mpo = self.maxPolyOrder # save it to re-set it after calling generateQuadsAndPolys
    self._generateQuadsAndPolys(SVL) # lives in GaussPolynomialRom object
    self.maxPolyOrder = mpo # re-set it

    # print out the setup for each variable.
    self.raiseADebug(' INTERPOLATION INFO:')
    self.raiseADebug('    Variable | Distribution | Quadrature | Polynomials')
    for v in self.quadDict:
      self.raiseADebug('   '+' | '.join([v,self.distDict[v].type,self.quadDict[v].type,self.polyDict[v].type]))
    self.raiseADebug('    Polynomial Set Type  : adaptive')

    #create the index set
    self.raiseADebug('Starting index set generation...')
    self.indexSet = IndexSets.factory.returnInstance('AdaptiveSet')
    self.indexSet.initialize(self.features,self.importanceDict,self.maxPolyOrder)
    for pt in self.indexSet.active:
      self.inTraining.add(pt)
      for t in self.targets:
        self.expImpact[t][pt] = 1.0 # dummy, just to help algorithm be consistent

    #make the first sparse grid
    self.sparseGrid = self._makeSparseQuad(self.indexSet.active)

    #set up the points we need RAVEN to run before we can continue
    self.newSolutionSizeShouldBe = len(self.solns)
    self._addNewPoints()

  def localStillReady(self, ready, skipJobHandlerCheck=False):
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      Determines what additional points are necessary for RAVEN to run.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ In, skipJobHandlerCheck, bool, optional, if true bypasses check on active runs in jobHandler
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    # if we're done, be done
    if self.done:
      return False
    # if we're not ready elsewhere, just be not ready
    if ready is False:
      return ready
    # if we still have a list of points to sample, just keep on trucking.
    if len(self.neededPoints) > 0:
      return True
    # if points all submitted but not all done, not ready for now.
    if (not self.batchDone) or (not skipJobHandlerCheck and not self.jobHandler.isFinished()):
      return False
    if len(self.solns) < self.newSolutionSizeShouldBe:
      return False
    # if no points to check right now, search for points to sample
    # this should also mean the points for the poly-in-training are done
    while len(self.neededPoints) < 1:
      # update sparse grid and set active impacts
      self._updateQoI()
      # move the index set forward -> that is, find the potential new indices
      self.indexSet.forward(self.maxPolyOrder)
      # estimate impacts of all potential indices
      for pidx in self.indexSet.active:
        self._estimateImpact(pidx)
      # check error convergence, using the largest impact from each target
      self.error = 0
      for pidx in self.indexSet.active:
        self.error += max(self.expImpact[t][pidx] for t in self.targets)
      # if logging, print to file
      if self.logFile is not None:
        self._printToLog()
      # if doing a study and past a statepoint, record the statepoint
      # discontinued temporarily, see notes above in localInputsAndChecks
      # if self.doingStudy:
      #  while len(self.studyPoints)>0 and len(self.pointsNeededToMakeROM) > self.studyPoints[0]:
      #    self._writeConvergencePoint(self.studyPoints[0])
      #    if self.studyPickle:
      #      self._writePickle(self.studyPoints[0])
      #    #remove the point
      #    if len(self.studyPoints)>1:
      #      self.studyPoints=self.studyPoints[1:]
      #    else:
      #      self.studyPoints = []
      # if error small enough, converged!
      if abs(self.error) < self.convValue:
        self.done = True
        self.converged = True
        break
      # if maxRuns reached, no more samples!
      if self.maxRuns is not None and len(self.pointsNeededToMakeROM) >= self.maxRuns:
        self.raiseAMessage('Maximum runs reached!  No further polynomial will be added.')
        self.done = True
        self.converged = True
        self.neededPoints=[]
        break
      # otherwise, not converged...
      # what if we have no polynomials to consider...
      if len(self.indexSet.active)<1:
        self.raiseADebug('No new polynomials to consider!')
        break
      # find the highest overall impact to run next
      idx = self._findHighestImpactIndex()
      # add it to the training list, and append its points to the requested ones
      self.inTraining.add(idx)
      newSG = self._makeSparseQuad([idx])
      self._addNewPoints(newSG)
    # if we exited while loop without finding points, we must be done!
    if len(self.neededPoints) < 1:
      self.converged = True
      self.raiseADebug('Index points in use, and their impacts:')
      for p in self.indexSet.points:
        self.raiseADebug('   ',p,list(self.actImpact[t][p] for t in self.targets))
      self._finalizeROM()
      self.unfinished = self.jobHandler.numRunning()
      self.jobHandler.terminateAll()
      self.neededPoints=[]
      self.done = True
      # suspended, see notes above
      # if self.doingStudy and len(self.studyPoints)>0:
      #  self.raiseAWarning('In the convergence study, the following numbers of runs were not reached:',self.studyPoints)
      return False
    # if we got here, we still have points to run!
    # print a status update...
    self.raiseAMessage(f'  Next: {idx} | error: {self.error:1.4e} | runs: {len(self.pointsNeededToMakeROM)}')

    return True

  def localGenerateInput(self, model, oldInput):
    """
      Function to select the next most informative point
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    self.inputInfo['ProbabilityWeight'] = 1.0
    pt = self.neededPoints.pop()
    self.submittedNotCollected.append(pt)
    for v, varName in enumerate(self.sparseGrid.varNames):
      # compute the SampledVarsPb for 1-D distribution
      if self.variables2distributionsMapping[varName]['totDim'] == 1:
        for key in varName.strip().split(','):
          self.values[key] = pt[v]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(pt[v])
        self.inputInfo['ProbabilityWeight-'+varName] = self.inputInfo['SampledVarsPb'][varName]
        # compute the SampledVarsPb for N-D distribution
      elif self.variables2distributionsMapping[varName]['totDim'] > 1 and self.variables2distributionsMapping[varName]['reducedDim'] == 1:
        dist = self.variables2distributionsMapping[varName]['name']
        ndCoordinates = np.zeros(len(self.distributions2variablesMapping[dist]))
        positionList = self.distributions2variablesIndexList[dist]
        for varDict in self.distributions2variablesMapping[dist]:
          var = utils.first(varDict.keys())
          position = utils.first(varDict.values())
          location = -1
          for key in var.strip().split(','):
            if key in self.sparseGrid.varNames:
              location = self.sparseGrid.varNames.index(key)
              break
          if location > -1:
            ndCoordinates[positionList.index(position)] = pt[location]
          else:
            self.raiseAnError(IOError,'The variables ' + var + ' listed in sparse grid collocation sampler, but not used in the ROM!' )
          for key in var.strip().split(','):
            self.values[key] = pt[location]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(ndCoordinates)
        self.inputInfo[f'ProbabilityWeight-{dist}'] = self.inputInfo['SampledVarsPb'][varName]
        self.inputInfo['ProbabilityWeight'] *= self.inputInfo[f'ProbabilityWeight-{dist}']
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['SamplerType'] = self.type

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    # check if all sampling is done
    if self.jobHandler.isFinished():
      self.batchDone = True
    else:
      self.batchDone = False
    # batchDone is used to check if the sampler should find new points.

  def _addNewPoints(self, SG=None):
    """
      Sort through sparse grid and add any new needed points
      @ In, SG, SparseGrid, optional, sparse grid to comb for new points
      @ Out, None
    """
    if SG is None:
      SG = self.sparseGrid
    for pt in SG.points()[:]:
      self.pointsNeededToMakeROM.add(pt) # sets won't store redundancies
      # if pt isn't already in needed, and it hasn't already been solved, add it to the queue
      if pt not in self.neededPoints and self.solns.realization(matchDict=self._tupleToDict(pt))[1] is None:
        self.newSolutionSizeShouldBe+=1
        self.neededPoints.append(pt)

  def _convergence(self, poly, rom, target):
    """
      Checks the convergence of the adaptive index set via one of (someday) several ways, currently "variance"
      @ In, poly, list(int), the polynomial index to check convergence for
      @ In, rom, supervisedContainer, the GaussPolynomialROM object with respect to which we check convergence
      @ In, target, string, target to check convergence with respect to
      @ Out, impact, float, estimated impact factor for this index set and sparse grid
    """
    if self.convType.lower() == 'variance':
      impact = rom.polyCoeffDict[target][poly]**2 / sum(rom.polyCoeffDict[target][p]**2 for p in rom.polyCoeffDict[target].keys())
    #FIXME 'coeffs' has to be updated to fit in the new rework before it can be used.
    # elif self.convType.lower()=='coeffs':
    #   #new = self._makeARom(rom.sparseGrid,rom.indexSet).supervisedContainer[target]
    #   tot = 0 #for L2 norm of coeffs
    #   if self.oldSG != None:
    #     oSG,oSet = self._makeSparseQuad()
    #     old = self._makeARom(oSG,oSet).supervisedContainer[target]
    #   else: old=None
    #   for coeff in new.polyCoeffDict.keys():
    #     if old!=None and coeff in old.polyCoeffDict.keys():
    #       n = new.polyCoeffDict[coeff]
    #       o = old.polyCoeffDict[coeff]
    #       tot+= (n - o)**2
    #     else:
    #       tot+= new.polyCoeffDict[coeff]**2
    #   impact = np.sqrt(tot)
    else:
      self.raiseAnError(KeyError, f'Unexpected convergence criteria: {self.convType}')

    return impact

  def _estimateImpact(self, idx):
    """
      Estimates the impact of polynomial with index idx by considering the product of its predecessor impacts.
      @ In, idx, tuple(int), polynomial index
      @ Out, None
    """
    # initialize
    for t in self.targets:
      self.expImpact[t][idx] = 1.
    # have = 0 # tracks the number of preceeding terms I have (e.g., terms on axes have less preceeding terms)
    # create a list of actual impacts for predecessors of idx
    predecessors = {}
    for t in self.targets:
      predecessors[t]=[]
    for i in range(len(self.features)):
      subidx = list(idx)
      if subidx[i]>0:
        subidx[i] -= 1
        for t in self.targets:
          predecessors[t].append(self.actImpact[t][tuple(subidx)])
      else:
        continue # on an axis or axial plane
    # estimated impact is the product of the predecessor impacts raised to the power of the number of predecessors
    for t in self.targets:
      # raising each predecessor to the power of the predecessors makes a more fair order-of-magnitude comparison
      #  for indices on axes -> otherwise, they tend to be over-emphasized
      self.expImpact[t][idx] = np.prod(np.power(np.array(predecessors[t]),1.0/len(predecessors[t])))

  def _finalizeROM(self, rom=None):
    """
      Initializes final target ROM with necessary objects for training.
      @ In, rom, GaussPolynomailROM object, optional, the rom to initialize, defaults to target rom
      @ Out, None
    """
    if rom is None:
      rom = self.ROM
    self.raiseADebug('No more samples to try! Declaring sampling complete.')
    # initialize final rom with final sparse grid and index set
    for SVL in rom.supervisedContainer:
      SVL.initialize({'SG':self.sparseGrid,
                      'dists':self.dists,
                      'quads':self.quadDict,
                      'polys':self.polyDict,
                      'iSet':self.indexSet,
                      'numRuns':len(self.pointsNeededToMakeROM)-self.unfinished})

  def _findHighestImpactIndex(self, returnValue=False):
    """
      Finds and returns the index with the highest average expected impact factor across all targets
      Can optionally return the value of the highest impact, as well.
      @ In, returnValue, bool, optional, returns the value of the index if True
      @ Out, point, tuple(int), polynomial index with greatest expected effect
    """
    point = None
    avg = 0
    # This finds a prototype of the samples from which the points can be found
    prototype = self.expImpact[self.targets[0]]
    for pt in sorted(prototype.keys()):
      new = sum(self.expImpact[t][pt] for t in self.targets)/len(self.targets)
      if avg < new:
        avg = new
        point = pt
    self.raiseADebug(f'Highest impact point is {point} with expected average impact {avg}')
    if returnValue:
      return point, avg
    else:
      return point

  def _integrateFunction(self, sg, r, i):
    """
      Uses the sparse grid sg to effectively integrate the r-th moment of the model.
      @ In, sg, SparseGrid, sparseGrid object
      @ In, r, int, integer moment
      @ In, i, int, index of target to evaluate
      @ Out, tot, float, approximate integral
    """
    tot = 0
    for n in range(len(sg)):
      pt, wt = sg[n]
      _, inExisting = self.solns.realization(matchDict=self._tupleToDict(pt))
      if inExisting is None:
        self.raiseAnError(RuntimeError, f'Trying to integrate with point {pt} but it is not in the solutions!')
      tot+=inExisting['outputs'][self.targets[i]]**r*wt

    return tot

  def _makeARom(self, grid, inset):
    """
      Generates a GaussPolynomialRom object using the passed in sparseGrid and indexSet,
      otherwise fundamentally a copy of the end-target ROM.
      @ In, grid, SparseGrid, sparseGrid
      @ In, inset, IndexSet, indexSet
      @ Out, rom, GaussPolynomialROM object, the constructed rom
    """
    # deepcopy prevents overwriting
    rom  = copy.deepcopy(self.ROM) # preserves interpolation requests via deepcopy
    sg   = copy.deepcopy(grid)
    iset = copy.deepcopy(inset)
    # reset supervisedContainer since some information is lost during deepcopy, such as 'features' and 'target'
    rom.supervisedContainer = [rom._interfaceROM]
    for svl in rom.supervisedContainer:
      svl.initialize({'SG'   : sg,
                      'dists': self.dists,
                      'quads': self.quadDict,
                      'polys': self.polyDict,
                      'iSet' : iset
                      })
    # while the training won't always need all of solns, it is smart enough to take what it needs
    rom.train(self.solns)

    return rom

  def _makeSparseQuad(self, points=[]):
    """
      Generates a sparseGrid object using the self.indexSet adaptively established points
      as well as and additional points passed in (often the indexSet's adaptive points).
      @ In, points, list(tuple(int)), optional, points
      @ Out, sparseGrid, SparseGrid object, new sparseGrid using self's points plus points' points
    """
    sparseGrid = Quadratures.factory.returnInstance(self.sparseGridType)
    iset = IndexSets.factory.returnInstance('Custom')
    iset.initialize(self.features,self.importanceDict,self.maxPolyOrder)
    iset.setPoints(self.indexSet.points)
    iset.addPoints(points)
    sparseGrid.initialize(self.features,iset,self.dists,self.quadDict,self.jobHandler)

    return sparseGrid

  def _printToLog(self):
    """
      Prints adaptive state of this sampler to the log file.
      @ In, None
      @ Out, None
    """
    self.logCounter += 1
    pl = 4*len(self.features)+1
    f = open(self.logFile,'a')
    f.writelines(f'===================== STEP {self.logCounter} =====================\n')
    f.writelines(f'\nNumber of Runs: {len(self.pointsNeededToMakeROM)}\n')
    f.writelines(f'Error: {self.error:1.9e}\n')
    f.writelines(f'Features: {", ".join(self.features)}\n')
    f.writelines('\nExisting indices:\n')
    f.writelines(f'    {"poly":^{pl}}:')
    for t in self.targets:
      f.writelines(f'  {t:<16}')
    f.writelines('\n')
    for idx in self.indexSet.points:
      f.writelines(f'    {idx:^{pl}}:')
      for t in self.targets:
        f.writelines(f'  {self.actImpact[t][idx]:<9}')
      f.writelines('\n')
    f.writelines('\nPredicted indices:\n')
    f.writelines(f'    {"poly":^{pl}}:')
    for t in self.targets:
      f.writelines(f'  {t:<16}')
    f.writelines('\n')
    for idx in utils.first(self.expImpact.values()):
      f.writelines(f'    {idx:^{pl}}:')
      for t in self.targets:
        f.writelines(f'  {self.expImpact[t][idx]:<9}')
      f.writelines('\n')
    f.writelines('===================== END STEP =====================\n')
    f.close()

  def _tupleToDict(self, pt, output=False):
    """
      Converts tuple in order of self.features into a dictionary varName:varValue
      @ In, pt, tuple(float), point
      @ In, output, bool, if True use self.targets instead
      @ Out, _tupleToDict, dict, dictionary
    """
    if output:
      return dict((k,v) for (k,v) in zip(self.targets, pt))

    return dict((k,v) for (k,v) in zip(self.features, pt))

  def _dictToTuple(self, pt, output=False):
    """
      Converts dictionary to tuple in order of self.features
      @ In, pt, dict, point
      @ In, output, bool, if True use self.targets instead
      @ Out, _dictToTuple, tuple(float), point
    """
    if output:
      return tuple(pt[v] for v in self.targets)

    return tuple(pt[v] for v in self.features)

  def _updateQoI(self):
    """
      Updates Reduced Order Models (ROMs) for Quantities of Interest (QoIs), as well as impact parameters and estimated error.
      @ In, None
      @ Out, None
    """
    # add active (finished) points to the sparse grid
    for active in list(self.inTraining):
      # add point to index set
      self.indexSet.accept(active)
      self.sparseGrid = self._makeSparseQuad()
      for t in self.targets:
        del self.expImpact[t][active]
      self.inTraining.remove(active)
    # update all the impacts
    rom = self._makeARom(self.sparseGrid,self.indexSet)
    for poly in self.indexSet.points:
      for t in self.targets:
        impact = self._convergence(poly,rom.supervisedContainer[0],t)
        self.actImpact[t][poly] = impact

  # disabled until we determine a consistent way to do this without bypassing dataobjects
  #def _writeConvergencePoint(self,runPoint):
  #  """
  #    Writes XML out for this ROM at this point in the run
  #    @ In, runPoint, int, the target runs for this statepoint
  #    @ Out, None
  #  """
  #  fname = self.studyFileBase+str(runPoint)
  #  self.raiseAMessage('Preparing to write state %i to %s.xml...' %(runPoint,fname))
  #  rom = copy.deepcopy(self.ROM)
  #  self._finalizeROM(rom)
  #  rom.train(self.solns)
  #  options = {'filenameroot':fname, 'what':'all'}
  #  rom.printXML(options)

  def _writePickle(self, runPoint):
    """
      Writes pickle for this ROM at this point in the run
      @ In, runPoint, int, the target runs for this statepoint
      @ Out, None
    """
    fname = self.studyFileBase + str(runPoint)
    self.raiseAMessage(f'Writing ROM at state {runPoint} to {fname}.pk...')
    rom = copy.deepcopy(self.ROM)
    self._finalizeROM(rom)
    rom.train(self.solns)
    pickle.dump(rom,open(fname+'.pk','wb'))

  def flush(self):
    """
      Reset SparsGridCollocation attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.ROM = None
    self.solns = None
    self.expImpact = {}
    self.actImpact = {}
    self.sparseGrid = None
    self.error = 0
    self.logCounter = 0
    self.neededPoints = []
    self.submittedNotCollected = []
    self.pointsNeededToMakeROM = set()
    self.unfinished = 0
    self.batchDone = True
    self.done = False
    self.newSolutionSizeShouldBe = None
    self.inTraining = set()
