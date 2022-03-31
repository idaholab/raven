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
  This module contains the Adaptive Sobol sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from talbpw
"""
import copy
import numpy as np
from operator import mul,itemgetter
from collections import OrderedDict
from functools import reduce
import xml.etree.ElementTree as ET
import itertools

from .Sobol import Sobol
from .AdaptiveSparseGrid import AdaptiveSparseGrid
from ..utils import utils
from ..utils import InputData, InputTypes
from .. import DataObjects
from .. import SupervisedLearning
from .. import Quadratures
from .. import IndexSets
from .. import Models

class AdaptiveSobol(Sobol, AdaptiveSparseGrid):
  """
    Adaptive Sobol sampler to obtain points adaptively for training a HDMR ROM.
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
    inputSpecification = super(AdaptiveSobol, cls).getInputSpecification()

    #Remove old convergence and convergenceStudy from AdaptiveSparseGrid
    inputSpecification.popSub("Convergence")
    inputSpecification.popSub("convergenceStudy")
    convergenceInput = InputData.parameterInputFactory("Convergence")

    convergenceInput.addSub(InputData.parameterInputFactory("relTolerance", contentType=InputTypes.FloatType))
    convergenceInput.addSub(InputData.parameterInputFactory("maxRuns", contentType=InputTypes.IntegerType))
    convergenceInput.addSub(InputData.parameterInputFactory("maxSobolOrder", contentType=InputTypes.IntegerType))
    convergenceInput.addSub(InputData.parameterInputFactory("progressParam", contentType=InputTypes.FloatType))
    convergenceInput.addSub(InputData.parameterInputFactory("logFile", contentType=InputTypes.StringType))
    convergenceInput.addSub(InputData.parameterInputFactory("subsetVerbosity", contentType=InputTypes.StringType))

    inputSpecification.addSub(convergenceInput)

    convergenceStudyInput = InputData.parameterInputFactory("convergenceStudy")

    convergenceStudyInput.addSub(InputData.parameterInputFactory("runStatePoints", contentType=InputTypes.StringType))
    convergenceStudyInput.addSub(InputData.parameterInputFactory("baseFilename", contentType=InputTypes.StringType))
    convergenceStudyInput.addSub(InputData.parameterInputFactory("pickle"))

    inputSpecification.addSub(convergenceStudyInput)

    return inputSpecification

  def __init__(self):
    """
      The constructor.
      @ In, None
      @ Out, None
    """
    AdaptiveSparseGrid.__init__(self)
    Sobol.__init__(self)

    #identification
    self.type            = 'AdaptiveSobolSampler'
    self.printTag        = 'SAMPLER ADAPTIVE SOBOL'
    self.stateCounter    = 0       #counts number of times adaptive step moves forward

    #input parameters
    self.maxSobolOrder   = None    #largest dimensionality of a subset combination
    #self.maxPolyOrder    = None   #largest polynomial order to use in subset sparse grids #TODO maybe someday
    self.maxRuns         = None    #most runs to allow total before quitting
    self.convValue       = None    #value to converge adaptive sampling to
    self.tweakParam      = 1.0     #ranges 0 (only polynomials) to 2 (only subsets)
    self.statesFile      = None    #file to log the progression of the adaptive sampling
    self.subVerbosity    = 'quiet' #verbosity level for the ROMs, samplers, dataobjects created within this sampler

    #assembly objects
    self.solns           = None    #solution database, PointSet data object
    self.ROM             = None    #HDMR rom that will be constructed with the samples found here

    #storage dictionaries
    self.ROMs            = {} #subset reduced-order models by subset: self.ROMs[target][subset]
    self.SQs             = {} #stores sparse grid quadrature objects
    self.samplers        = {} #stores adaptive sparse grid sampling objects
    self.romShell        = {} #stores Model.ROM objects for each subset
    self.iSets           = {} #adaptive index set objects by target,subset
    self.pointsNeeded    = {} #by subset, the points needed for next step in adaptive SG sampler
    self.pointsCollected = {} #by subset, the points collected for next stip in adaptive SG sampler
    self.subsets         = {} #subset gPC ROMs to be used in full HDMR ROM that have at least started training
    self.references      = {} #mean-value cut reference points by variable
    self.useSet          = {} #accepted subsets and the associated ROMs, as useSet[subset][target]

    #convergence parameters
    self.subsetImpact    = {}    #actual impact on variance by subset combo
    self.subsetExpImpact = {}    #estimated impact on variance by subset combo
    self.done            = False #boolean to track if we've converged, or gone over limit
    self.distinctPoints  = set() #list of points needed to make this ROM, for counting purposes
    self.numConverged    = 0     #tracking for persistance
    self.persistence     = 2     #set in input, the number of successive converges to require

    #convergence study
    self.doingStudy      = False  #true if convergenceStudy node defined for sampler
    self.studyFileBase   = 'out_' #can be replaced in input, not used if not doingStudy
    self.studyPoints     = []     #list of ints, runs at which to record a state
    self.studyPickle     = False  #if true, creates a pickle of rom at statepoints

    #attributes
    self.features        = None #ROM features of interest, also input variable list
    self.targets         = None #ROM outputs of interest

    #point lists
    self.sorted          = []       #points that have been sorted into appropriate objects
    self.submittedNotCollected = [] #list of points that have been generated but not collected
    self.inTraining      = []       #usually just one tuple, unless multiple items in simultaneous training

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    Sobol.localInputAndChecks(self,xmlNode, paramInput)
    conv = xmlNode.find('Convergence')
    studyNode = xmlNode.find('convergenceStudy')
    if conv is None:
      self.raiseAnError(IOError,'"Convergence" node not found in input!')
    #self.convType      = conv.get('target',None) #TODO not implemented.  Currently only does variance.
    for child in conv:
      if child.tag == 'relTolerance':
        self.convValue = float(child.text)
      elif child.tag == 'maxRuns':
        self.maxRuns = int(child.text)
      elif child.tag == 'maxSobolOrder':
        self.maxSobolOrder = int(child.text)
      #elif child.tag== 'maxPolyOrder'   : self.maxPolyOrder  =   int(child.text) #TODO someday maybe.
      elif child.tag == 'progressParam':
        self.tweakParam = float(child.text)
      elif child.tag == 'logFile':
        self.statesFile = open(child.text,'w')
      elif child.tag == 'subsetVerbosity':
        self.subVerbosity = child.text.lower()
    if not 0 <= self.tweakParam <= 2:
      self.raiseAnError(IOError,'progressParam must be between 0 (only add polynomials) and 2 (only add subsets) (default 1).  Input value was',self.tweakParam,'!')
    if self.subVerbosity not in ['debug','all','quiet','silent']:
      self.raiseAWarning('subsetVerbosity parameter not recognized:',self.subVerbosity,' -> continuing with "quiet"')
      self.subVerbosity = 'quiet'
    if studyNode is not None:
      self.doingStudy = True
      self.studyPoints = studyNode.find('runStatePoints').text
      filebaseNode = studyNode.find('baseFilename')
      self.studyPickle = studyNode.find('pickle') is not None
      if filebaseNode is None:
        self.raiseAWarning('No baseFilename specified in convergenceStudy node!  Using "%s"...' %self.studyFileBase)
      else:
        self.studyFileBase = studyNode.find('baseFilename').text
      if self.studyPoints is None:
        self.raiseAnError(IOError,'convergenceStudy node was included, but did not specify the runStatePoints node!')
      else:
        try:
          self.studyPoints = list(int(i) for i in self.studyPoints.split(','))
        except ValueError as e:
          self.raiseAnError(IOError,'Convergence state point not recognizable as an integer!',e)
        self.studyPoints.sort()

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    #we don't use restarts in adaptive sampling
    if self.restartData is not None:
      self.raiseAnError(IOError,'AdaptiveSobol does not take Restart node!  Use TargetEvaluation instead.')
    #set up assembly-based objects
    self.solns = self.assemblerDict['TargetEvaluation'][0][3]
    SVL = self.readFromROM()
    self.targets = SVL.target
    self.subsetImpact = {key: dict({}) for key in self.targets}
    #generate quadratures and polynomials
    self._generateQuadsAndPolys(SVL)
    #set up reference case
    for var,dist in self.dists.items():
      self.references[var] = dist.untruncatedMean()
    #set up first subsets, the mono-dimensionals
    self.firstCombos = list(itertools.chain.from_iterable(itertools.combinations(self.features,r) for r in [0,1]))
    for c in self.firstCombos[:]:
      #already did reference case, so remove it
      if len(c)<1:
        self.firstCombos.remove(c)
        continue
      self._makeSubsetRom(c)
      self.inTraining.append( ('poly',c,self.samplers[c]._findHighestImpactIndex()) )
      #get the points needed to push this subset forward
      self._retrieveNeededPoints(c)
    #set up the nominal point for a run
    #  Note: neededPoints is not going to be the main method for queuing points, but it will take priority.
    self.neededPoints = [tuple(self.references[var] for var in self.features)]

  def localStillReady(self,ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    #if we've already capped runs or are otherwise done, return False
    if self.done:
      self.raiseADebug('Sampler is already done; no more runs available.')
      return False
    #if for some reason we're not ready already, just return that
    if not ready:
      return ready
    #collect points that have been run
    self._sortNewPoints()
    #if starting set of points is not done, just return
    if len(self.neededPoints)>0:
      return True
    #look for any new points to run, if we don't have any
    while sum(len(self.pointsNeeded[s[1]]) for s in self.inTraining)<1:
      #since we don't need any points to sample, we can train
      for item in self.inTraining:
        sub = item[1]
        # whether we were training a poly or a new subset, we need to update the subset
        self._updateSubset(sub)
      # now that we've updated the subsets, we can train them and update the actual and expected impacts
      for item in self.inTraining:
        sub = item[1]
        #train it
        self.samplers[sub]._finalizeROM()
        self.romShell[sub].train(self.samplers[sub].solns)
        #update the actual impacts
        for t in self.targets:
          self.subsetImpact[t][sub] = self._calcActualImpact(sub,t)
          if sub in self.subsetExpImpact.keys():
            del self.subsetExpImpact[sub]
        #add new/update expected impacts of subsets
        self._generateSubsets(sub)
        #remove this item from the training queue
        self.inTraining.remove(item)
      #are we at maxRuns?  If so, we need to be done.
      if self.maxRuns is not None and len(self.distinctPoints)>self.maxRuns:
        self.raiseAMessage('Maximum runs reached!  No new polynomials or subsets will be added...')
        self._earlyExit()
        return False
      #get next-most influential poly/subset to add, update global error estimate
      which, toDoSub, poly = self._getLargestImpact()
      self.raiseAMessage('Next: %6s %8s%12s' %(which,','.join(toDoSub),str(poly)),'| error: %1.4e' %self.error,'| runs: %i' %len(self.distinctPoints))
      if self.statesFile is not None:
        self._printState(which,toDoSub,poly)
      #if doing a study and past a statepoint, record the statepoint
      if self.doingStudy:
        while len(self.studyPoints)>0 and len(self.distinctPoints) > self.studyPoints[0]:
          self._writeConvergencePoint(self.studyPoints[0])
          if self.studyPickle:
            self._writePickle(self.studyPoints[0])
          #remove the point
          if len(self.studyPoints)>1:
            self.studyPoints=self.studyPoints[1:]
          else:
            self.studyPoints = []
      #are we converged?
      if self.error < self.convValue:
        self.raiseAMessage('Convergence achieved!  No new polynomials or subsets will be added...')
        self._earlyExit()
        return False
      #otherwise, we're not done...
      #  -> use the information from _getLargestImpact to add either a poly or a subset
      if which == 'poly':
        self.inTraining.append(('poly',toDoSub,self.samplers[toDoSub]._findHighestImpactIndex()))
        samp = self.samplers[toDoSub]
        #add the poly to the subset sampler's training queue
        samp.inTraining.add(self.inTraining[-1][2])
        #add new necessary points to subset sampler
        samp._addNewPoints(samp._makeSparseQuad([self.inTraining[-1][2]]))
        #get those new needed points and store them locally
        self._retrieveNeededPoints(toDoSub)
      elif which == 'subset':
        self._makeSubsetRom(toDoSub)
        self.ROMs[toDoSub] = self.romShell[toDoSub].supervisedContainer[0]
        self.inTraining.append(('subset',toDoSub,self.romShell[toDoSub]))
        #get initial needed points and store them locally
        self._retrieveNeededPoints(toDoSub)
    #END while loop
    #if all the points we need are currently submitted but not collected, we have no points to offer
    if not self._havePointsToRun():
      return False
    #otherwise, we can submit points!
    return True

  def localGenerateInput(self,model,oldInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    #note: pointsNeeded is the collection of points needed by sampler,
    #      while neededPoints is just the reference point that needs running
    #if there's a point that THIS sampler needs, prioritize it
    self.inputInfo['ProbabilityWeight'] = 1.0
    if len(self.neededPoints)>0:
      pt = self.neededPoints.pop()
    #otherwise, take from the highest-impact sampler's needed points
    else:
      #pointsNeeded is in order from least to most impactful, so list reverse of keys.
      subsets = list(self.pointsNeeded.keys())
      subsets.reverse()
      #now they're in order of impact.  Look for the next point to run.
      found = False
      for sub in subsets:
        for p in self.pointsNeeded[sub]:
          pt = self._expandCutPoint(sub,p)
          if pt not in self.submittedNotCollected:
            self.submittedNotCollected.append(pt)
            found = True
            break
        if found:
          break
      if not found:
        #this should not occur, but is a good sign something went wrong in developing.
        self.raiseAnError(RuntimeError,'No point was found to generate!  This should not be possible...')
    #add the number of necessary distinct points to a set (so no duplicates).
    self.distinctPoints.add(pt)
    for v,varName in enumerate(self.features):
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
            if key in self.features:
              location = self.features.index(key)
              break
          if location > -1:
            ndCoordinates[positionList.index(position)] = pt[location]
          else:
            self.raiseAnError(IOError,'The variables ' + var + ' listed in adaptive sobol sampler, but not used in the ROM!' )
          for key in var.strip().split(','):
            self.values[key] = pt[location]
        self.inputInfo['SampledVarsPb'][varName] = self.distDict[varName].pdf(ndCoordinates)
        self.inputInfo['ProbabilityWeight-'+dist] = self.inputInfo['SampledVarsPb'][varName]
        self.inputInfo['ProbabilityWeight']*=self.inputInfo['ProbabilityWeight-'+dist]
    self.inputInfo['PointProbability'] = reduce(mul,self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['SamplerType'] = 'Adaptive Sparse Grids for Sobol'
  def _addPointToDataObject(self,subset,point):
    """
      Adds a cut point to the data object for the subset sampler.
      @ In, subset, tuple(string), the cut point
      @ In, point, tuple(int), the cut point to add
      @ Out, None
    """
    pointSet = self.samplers[subset].solns
    #first, check if the output is in the subset's existing solution set already
    _,inExisting = self.solns.realization(matchDict=self._tupleToDict(self._expandCutPoint(subset,point)))
    #add the point to the data set.
    rlz = dict((var,np.atleast_1d(inExisting[var])) for var in pointSet.getVars())
    pointSet.addRealization(rlz)

  def _calcActualImpact(self,subset,target):
    """
      Calculates the total impact of the current set.
      @ In, subset, tuple(str), new subset for which impact is considered
      @ Out, _calcActualImpact, float, the "error" reduced by acquiring the new point
    """
    #add the new term to the use set
    if subset not in self.useSet.keys():
      self.useSet[subset] = None
    self.useSet[subset] = self.ROMs[subset]
    #compute the impact as the contribution to the variance
    ### SAVING for FUTURE: attempt at improving Adaptive Sobol algorithm.
    # The problem is, until the (1,1,1,...,1) point is in the index set, it
    # claims a sensitivity of 0 for that subset (rightfully so), but that leads
    # into a problem searching unless you start with tensor [0,1] indices.
    #copyShell = copy.deepcopy(self.ROM)
    #copyShell.supervisedContainer = {}
    #for targ,rom in self.ROM.supervisedContainer.items():
    #  copyShell.supervisedContainer[targ] = copy.deepcopy(rom)
    #copyROM =  copyShell.supervisedContainer[target]
    #self._finalizeROM(rom=copyShell,include=[subset])
    #copyShell.train(self.solns)
    #sens,partVar = copyROM.getSensitivities()
    #if subset not in sens.keys(): return 0 #TODO will this result in a bad search?
    #TODO FIXME why is this not working, since I'm using "full" as the index set starter?
    #return sens[subset]
    ### END SAVING
    totvar = 0
    for s in self.useSet.keys():
      totvar += self.ROMs[s].__variance__(target)
    #avoid div by 0 error
    if totvar > 0:
      return self.ROMs[subset].__variance__(target)/totvar
    else:
      return self.ROMs[subset].__variance__(target)

  def _calcExpImpact(self,subset,target):
    """
      Estimates the importance (impact) of the subset, based on its predecessors
      @ In, subset, tuple(str), the subset spanning the cut plane of interest
      @ In, target, str, target to estimate impact for
      @ Out, impact, float, the expected impact
    """
    #estimate impact as the product of predecessors
    #TODO this could be improved for higher dimensions, ie s(a,b,c) = s(a)*s(b)*s(c) or s(a,b)*s(c) or ?
    #for now, using product of all of the immediate predecessors, which seems to be an okay guess
    impact = 1
    for sub in self.useSet.keys():
      #only use immediate predecessor
      if len(sub)<len(subset)-1:
        continue
      #use builtin set mechanics to figure out if "sub" is a subset of "subset"
      if set(sub).issubset(set(subset)):
        #confusing naming!  if sub is a predecessor of subset...
        impact*=self.subsetImpact[target][sub]
    return impact

  def _checkCutPoint(self,subset,pt):
    """
      Determines if a point is in the cut set for the features in the subset.
      @ In, subset, tuple(str), desired subset features
      @ In, pt, tuple(float), the full point
      @ Out, _checkCutPoint, bool, True if pt only varies from reference in dimensions within the subset
    """
    for v,var in enumerate(self.features):
      if var in subset:
        continue #it's okay to vary if you're in the subset
      if pt[v] != self.references[var]:
        #we're outside the cut plane.
        return False
    return True #only if nothing outside the cut plane

  def _expandCutPoint(self,subset,pt):
    """
      Takes a trimmed point from the cut plane and expands it to include the reference values.
      @ In, subset, tuple(str), the subset describing this cut plane
      @ In, pt, tuple(float), the trimmed cutpoint to expand
      @ Out, full, tuple(float), full expanded points
    """
    #initialize full point
    full = np.zeros(len(self.features))
    for v,var in enumerate(self.features):
      #if it's a varying point (spanned by the subset), keep its value
      if var in subset:
        full[v] = pt[subset.index(var)]
      #else, use the reference value
      else:
        full[v] = self.references[var]
    return tuple(full)

  def _extractCutPoint(self,subset,pt):
    """
      Trims the dimensionality of pt to the cut plane spanning subset
      @ In, subset, tuple(str), the cut plane to trim to
      @ In, pt, tuple(float), the point to extract
      @ Out, cutInp, tuple, tuple(pt,vals) -> extracted point with cardinality equal to the subset cardinality
    """
    #slightly faster all in one line.
    cutInp = tuple(pt[self.features.index(var)] for var in subset)
    return cutInp

  def _earlyExit(self):
    """
      In the event the sampler has to terminate before normal completion, this helps to assure
      a usable set of ROMs make it to the HDMR ROM.
      @ In, None
      @ Out, None
    """
    #remove unfinished subsets
    toRemove = []
    for subset in self.ROMs.keys():
      if subset not in self.useSet.keys():
        toRemove.append(subset)
    for subset in toRemove:
      del self.ROMs[subset]
    #finalize subsets
    for sub in self.useSet.keys():
      self._finalizeSubset(sub)
    #set completion trigger
    self.done = True
    #note any missing statepoints if doing convergence study
    if self.doingStudy and len(self.studyPoints)>0:
      self.raiseAWarning('In the convergence study, the following numbers of runs were not reached:',self.studyPoints)
    #set up HDMRRom for training
    self._finalizeROM()

  def _finalizeROM(self, rom=None, include=[]):
    """
      Delivers necessary structures to the HDMRRom object
      @ In, rom, HDMRRom object, optional, rom to finalize before training, defaults to target rom
      @ In, include, list[str], optional, subsets to optionally exclude from trimming
      @ Out, None
    """
    if rom == None:
      rom = self.ROM
    initDict = {'ROMs':None, # multitarget requires setting individually, below
                'SG':self.SQs,
                'dists':self.dists,
                'quads':self.quadDict,
                'polys':self.polyDict,
                'refs':self.references,
                'numRuns':len(self.distinctPoints)}
    #initialize each HDMRRom object in the ROM
    initDict['ROMs'] = copy.deepcopy(self.ROMs)
    #remove unfinished subsets
    for subset in self.ROMs.keys():
      if subset not in self.useSet.keys() and subset not in include:
        del initDict['ROMs'][subset]
    rom.supervisedContainer[0].initialize(initDict)

  def _finalizeSubset(self, subset):
    """
      On completion, finalizes the subset by initializing the associated ROM.
      @ In, subset, tuple(str), subset to finalize
      @ Out, None
    """
    sampler = self.samplers[subset]
    #add collected points to sampler's data object, just in case one's missing.  Could be optimized.
    for pt in self.pointsCollected[subset]:
      self._addPointToDataObject(subset,pt)
    #finalize the subset ROM
    sampler._finalizeROM()
    #train the ROM
    self.romShell[subset].train(sampler.solns)
    #store rom in dedicated use set
    self.useSet[subset] = self.romShell[subset].supervisedContainer[0]

  def _generateSubsets(self, subset):
    """
      Returns a list of the possible subset combinations available, and estimates their impact
      @ In, subset, tuple(str), the leading subset to add more subsets from
      @ Out, None
    """
    #get length of subset
    l = len(subset)
    #we want all combinations of subsets using subset and adding only one more
    #first, get all possible combinations of that length
    #TODO this is wasteful, but I don't know a better way.
    potential = itertools.combinations(self.features,l+1)
    #if all the subset dimensions are in the potential, then it could possibly be used
    #but don't include if it's already there, or if it's in training.
    use = []
    self.raiseADebug('Generating subsets on',subset,'...')
    for p in potential:
      if all(i in p for i in subset):
        if p not in self.useSet.keys():
          if p not in list(s[1] for s in self.inTraining):
            use.append(p)
    if len(use)<1:
      self.raiseADebug('    no new potentials found.')
      return
    #now, look for ones that have all necessary subsets in the use set.
    for p in use:
      if len(p)>self.maxSobolOrder:
        self.raiseADebug('        Discarded',p,'for too large subset cardinality.')
        continue
      #to be included, p needs all of its precedents of lower cardinality to be in the useSet already.
      neededPrecedents = list(itertools.combinations(p,len(p)-1))
      if all(c in self.useSet.keys() for c in neededPrecedents):
        self.raiseADebug('  Adding subset:',p)
        self._makeSubsetRom(p)
        #get expected impact - the max impact among from the targets
        self.subsetExpImpact[p] = max(abs(self._calcExpImpact(p,t)) for t in self.targets)
    #now order the expected impacts so that lowest is first (-1 is highest)
    toSort = list(zip(self.subsetExpImpact.keys(),self.subsetExpImpact.values()))
    toSort.sort(key=itemgetter(1))
    #restore them to the ordered dict.
    self.subsetExpImpact = OrderedDict()
    for key, impact in toSort:
      self.subsetExpImpact[key] = impact

  def _getLargestImpact(self):
    """
      Looks through potential subsets and existing subsets for the most effective polynomial to add
      @ In, None
      @ Out, _getLargestImpact, (str, tuple(str), item ), either 'poly' or 'subset' along with the corresponding subset and either the poly or ''
    """
    #track the total error while we do this
    self.error = 0
    #storage for most impactful polynomial: its impact, the subset it belongs to, and the polynomial index
    maxPolyImpact = 0
    maxPolySubset = None
    poly = None
    #find most effective polynomial among existing subsets
    for subset in self.useSet.keys():
      #if it's already in training, move along
      if any(subset == s[1] for s in self.inTraining):
        continue
      pt,imp =  self.samplers[subset]._findHighestImpactIndex(returnValue = True)
      #apply tweaking parameter for favoring either polys or subsets
      imp = imp**self.tweakParam * (sum(self.subsetImpact[t][subset] for t in self.targets)/len(self.targets))**(2.-self.tweakParam)
      #update global estimated error
      self.error+=imp
      #update max if necessary
      if maxPolyImpact < imp:
        maxPolyImpact = imp
        maxPolySubset = subset
        poly = pt
    #storage for the most impactful subset: its impact, and the subset
    maxSubsetImpact = 0
    maxSubset = None
    #find the expected most effective subset among potential subsets
    for subset,expImp in self.subsetExpImpact.items():
      #if it's already in training, move along
      if any(subset == s[1] for s in self.inTraining):
        continue
      #apply favoring tweaking parameter - take abs() to assure fair comparison
      expImp = abs(expImp)**(2.-self.tweakParam)
      #update global expected error remaining
      self.error+=expImp
      #update max if necessary
      if maxSubsetImpact < expImp:
        maxSubsetImpact = expImp
        maxSubset = subset
    #which champion (poly or subset) is more significant? Slightly favour polynomials as a tiebreaker
    if maxPolySubset is None and maxSubset is None:
      self.raiseAnError(RuntimeError,'No polynomials or subsets found to consider!')
    if maxPolyImpact >= maxSubsetImpact:
      self.raiseADebug('Most impactful is resolving subset',maxPolySubset)
      return 'poly',maxPolySubset,poly
    else:
      self.raiseADebug('Most impactful is adding subset',maxSubset)
      return 'subset',maxSubset,''

  def _havePointsToRun(self):
    """
      Determines if there are points to submit to the jobHandler.
      @ In, None
      @ Out, _havePointsToRun, bool, true if there are points to run
    """
    #check if there's any subsets in the useSet that need points run, that haven't been queued
    for subset in self.useSet.keys():
      for pt in self.pointsNeeded[subset]:
        if self._expandCutPoint(subset,pt) not in self.submittedNotCollected:
          return True
    #check if there's anything in training that needs points run, that haven't been queued
    for item in self.inTraining:
      subset = item[1]
      for pt in self.pointsNeeded[subset]:
        if self._expandCutPoint(subset,pt) not in self.submittedNotCollected:
          return True
    #if not, we have nothing to run.
    return False

  def _makeCutDataObject(self, subset):
    """
      Creates a new PointSet dataobject for a cut subset
      @ In, subset, tuple(str), the subset to make the object for
      @ Out, dataObject, DataObject object, data object with cut points
    """
    #create a new data ojbect
    dataObject = DataObjects.factory.returnInstance('PointSet')
    dataObject.type ='PointSet'
    #write xml to set up data object
    #  -> name it the amalgamation of the subset parts
    node = ET.Element('PointSet',{'name':'-'.join(subset),'verbosity':self.subVerbosity})
    inp = ET.Element('Input')
    inp.text = ','.join(s for s in subset)
    node.append(inp)
    out = ET.Element('Output')
    out.text = ','.join(self.targets)
    node.append(out)
    #initialize the data object
    dataObject.readXML(node)
    return dataObject

  def _makeSubsetRom(self, subset):
    """
      Constructs a ROM for the given subset (but doesn't train it!).
      @ In, subset, tuple(string), subset for cut plane
      @ Out, None
    """
    from .Factory import factory
    verbosity = self.subVerbosity #sets verbosity of created RAVEN objects
    SVL = self.ROM.supervisedContainer[0] #an example SVL for most parameters
    #replicate "normal" construction of the ROM
    distDict={}
    quadDict={}
    polyDict={}
    imptDict={}
    limit=0
    dists = {}
    #make use of the keys to get the distributions, quadratures, polynomials, importances we want
    for c in subset:
      distDict[c] = self.dists[c]
      dists[c] = self.dists[c]
      quadDict[c] = self.quadDict[c]
      polyDict[c] = self.polyDict[c]
      imptDict[c] = self.importanceDict[c]
    #instantiate an adaptive index set for this ROM
    iSet = IndexSets.factory.returnInstance('AdaptiveSet')
    iSet.initialize(subset,imptDict,self.maxPolyOrder,full=True)
    iSet.verbosity=verbosity
    #instantiate a sparse grid quadrature
    self.SQs[subset] = Quadratures.factory.returnInstance(self.sparseGridType)
    self.SQs[subset].initialize(subset,iSet,distDict,quadDict,self.jobHandler)
    #instantiate the SVLs.  Note that we need to call both __init__ and initialize with dictionaries.
    #for target in self.targets:
    initDict = {'IndexSet'       : iSet.type,
                'PolynomialOrder': SVL.maxPolyOrder,
                'Interpolation'  : SVL.itpDict,
                'Features'       : list(subset),
                'Target'         : self.targets}
    self.ROMs[subset] = SupervisedLearning.factory.returnInstance('GaussPolynomialRom')
    self.ROMs[subset].initializeFromDict(initDict)
    initializeDict = {'SG'       : self.SQs[subset],
                      'dists'    : distDict,
                      'quads'    : quadDict,
                      'polys'    : polyDict,
                      'iSet'     : iSet}
    self.ROMs[subset].initialize(initializeDict)
    self.ROMs[subset].verbosity = verbosity
    #instantiate the shell ROM that contains the SVLs
    #   NOTE: the shell is only needed so we can call the train method with a data object.
    self.romShell[subset] = Models.factory.returnInstance('ROM')
    self.romShell[subset].subType = 'GaussPolynomialRom'
    self.romShell[subset].verbosity = verbosity
    self.romShell[subset]._interfaceROM = self.ROMs[subset]
    self.romShell[subset].canHandleDynamicData = self.romShell[subset]._interfaceROM.isDynamic()
    self.romShell[subset].supervisedContainer = [self.romShell[subset]._interfaceROM]
    #coordinate SVLs
    #instantiate the adaptive sparse grid sampler for this rom
    samp = factory.returnInstance('AdaptiveSparseGrid')
    samp.verbosity      = verbosity
    samp.doInParallel   = self.doInParallel #TODO can't be set by user.
    samp.jobHandler     = self.jobHandler
    samp.convType       = 'variance'
    samp.maxPolyOrder   = self.maxPolyOrder
    samp.distDict       = distDict
    samp.dists          = dists
    samp.assemblerDict['ROM']              = [['','','',self.romShell[subset]]]
    soln = self._makeCutDataObject(subset)
    samp.assemblerDict['TargetEvaluation'] = [['','','',soln]]
    for var in subset:
      samp.axisName.append(var)
    samp.localInitialize()
    samp.printTag = 'ASG:('+','.join(subset)+')'
    #propogate sparse grid back from sampler #TODO self.SQs might not really be necessary.
    self.SQs[subset] = samp.sparseGrid
    self.ROMs[subset].sparseGrid  = samp.sparseGrid
    self.samplers[subset] = samp
    #initialize pointsNeeded and pointsCollected databases
    self.pointsNeeded[subset] = []
    self.pointsCollected[subset] = []
    #sort already-solved points
    for inp in self.sorted:
      if self._checkCutPoint(subset,inp):
        #get the solution
        _,inExisting = self.solns.realization(matchDict=self._tupleToDict(inp))
        soln = self._dictToTuple(inExisting,output=True)
        #get the cut point
        cinp = self._extractCutPoint(subset,inp)
        self._addPointToDataObject(subset,cinp)
        self.pointsCollected[subset].append(cinp)
    #get the points needed by the subset samplers and store them locally
    self._retrieveNeededPoints(subset)
    #advance the subset forward if it doesn't have needed points
    if len(self.pointsNeeded[subset])<1:
      self._updateSubset(subset)

  def _printState(self,which,toDoSub,poly):
    """
      Debugging tool.  Prints status of adaptive steps. Togglable in input by specifying logFile.
      @ In, which, string, the type of the next addition to make by the adaptive sampler: poly, or subset
      @ In, toDoSub, tuple(str), the next subset that will be resolved as part of the adaptive sampling
      @ In, poly, tuple(int), the polynomial within the next subset that will be added to resolve it
      @ Out, None
    """
    #print status, including error; next step to make; and existing, training, and expected values
    self.stateCounter+=1
    self.statesFile.writelines('==================== STEP %s ====================\n' %self.stateCounter)
    #write error, next adaptive move to make in this step
    self.statesFile.writelines('\n\nError: %1.9e\n' %self.error)
    self.statesFile.writelines('Next: %6s %8s %12s\n' %(which,','.join(toDoSub),str(poly)))
    #write a summary of the state of each subset sampler: existing points, training points, yet-to-try points, and their impacts on each target
    for sub in self.useSet.keys():
      self.statesFile.writelines('-'*50)
      self.statesFile.writelines('\nsubset %8s with impacts' %','.join(sub))
      for t in self.targets:
        self.statesFile.writelines(    ' [ %4s:%1.6e ] ' %(t,self.subsetImpact[t][sub]))
      self.statesFile.writelines('\n')
      #existing polynomials
      self.statesFile.writelines('ESTABLISHED:\n')
      self.statesFile.writelines('    %12s' %'polynomial')
      for t in self.targets:
        self.statesFile.writelines('  %12s' %t)
      self.statesFile.writelines('\n')
      for coeff in utils.first(self.romShell[sub].supervisedContainer[0].polyCoeffDict.values()).keys():
        self.statesFile.writelines('    %12s' %','.join(str(c) for c in coeff))
        for t in self.targets:
          self.statesFile.writelines('  %1.6e' %self.romShell[sub].supervisedContainer[0].polyCoeffDict[t][coeff])
        self.statesFile.writelines('\n')
      #polynomials in training
      if any(sub==item[1] for item in self.inTraining):
        self.statesFile.writelines('TRAINING:\n')
      for item in self.inTraining:
        if sub == item[1]:
          self.statesFile.writelines('    %12s %12s\n' %(sub,item[2]))
      #polynomials on the fringe that aren't being trained
      self.statesFile.writelines('EXPECTED:\n')
      for poly in utils.first(self.samplers[sub].expImpact.values()).keys():
        self.statesFile.writelines('    %12s' %','.join(str(c) for c in poly))
        self.statesFile.writelines('  %1.6e' %self.samplers[sub].expImpact[t][poly])
        self.statesFile.writelines('\n')
    self.statesFile.writelines('-'*50+'\n')
    #other subsets that haven't been started yet
    self.statesFile.writelines('EXPECTED SUBSETS\n')
    for sub,val in self.subsetExpImpact.items():
      self.statesFile.writelines('    %8s: %1.6e\n' %(','.join(sub),val))
    self.statesFile.writelines('\n==================== END STEP ====================\n')

  def _retrieveNeededPoints(self,subset):
    """
      Get the batch of points needed by the subset sampler and transfer them to local variables
      @ In, subset, tuple(str), cut plane dimensions
      @ Out, None
    """
    sampler = self.samplers[subset]
    #collect all the points and store them locally, so we don't have to inquire the subset sampler
    while len(sampler.neededPoints)>0:
      cutpt = sampler.neededPoints.pop()
      fullPoint = self._expandCutPoint(subset,cutpt)
      #if this point already in local existing, put it straight into collected and sampler existing
      _,inExisting = self.solns.realization(matchDict=self._tupleToDict(fullPoint))
      if inExisting is not None:
        self.pointsCollected[subset].append(cutpt)
        self._addPointToDataObject(subset,cutpt)
      #otherwise, this is a point that needs to be run!
      else:
        self.pointsNeeded[subset].append(cutpt)

  def _sortNewPoints(self):
    """
      Allocates points on cut planes to their respective adaptive sampling data objects.
      @ In, None
      @ Out, None
    """
    #if there's no solutions in the set, no work to do
    if len(self.solns) == 0:
      return
    #update self.exisitng for adaptive sobol sampler (this class)
    for i in range(len(self.solns)):
      existing = self.solns.realization(index=i)
      inp = self._dictToTuple(existing)
      soln = self._dictToTuple(existing,output=True)
      #if point already sorted, don't re-do work
      if inp not in self.submittedNotCollected:
        continue
      #check through neededPoints to find subset that needed this point
      self.raiseADebug('sorting:',inp,soln)
      for subset,needs in self.pointsNeeded.items():
        #check if point in cut for subset
        if self._checkCutPoint(subset,inp):
          self.raiseADebug('...sorting into',subset)
          cutInp = self._extractCutPoint(subset,inp)
          self._addPointToDataObject(subset,cutInp)
          sampler = self.samplers[subset]
          #check if it was requested
          if cutInp in needs:
            #if so, remove the point from Needed ...
            self.pointsNeeded[subset].remove(cutInp)
          # ... and into Collected
          if subset not in self.pointsCollected.keys():
            self.pointsCollected[subset] = []
          self.pointsCollected[subset].append(cutInp)
      self.sorted.append(inp)
      self.submittedNotCollected.remove(inp)

  def _updateSubset(self,subset):
    """
      Updates the index set for the subset, and updates estimated impacts
      @ In, subset, tuple(str), the subset to advance
      @ Out, None
    """
    if len(self.pointsNeeded[subset])<1:
      sampler = self.samplers[subset]
      #update the ROM with the new polynomial point
      sampler._updateQoI()
      #refresh the list of potential points in the index set
      #XXX below line was:
      #sampler.indexSet.forward(sampler.indexSet.points[-1])
      #but forward takes a single integer not a tuple like points[-1] is.
      sampler.indexSet.forward()
      #update estimated impacts
      for pidx in sampler.indexSet.active:
        sampler._estimateImpact(pidx)

  def _writeConvergencePoint(self,runPoint):
    """
      Writes XML out for this ROM at this point in the run
      @ In, runPoint, int, the target runs for this statepoint
      @ Out, None
    """
    for sub in self.useSet.keys():
      self._finalizeSubset(sub)
    AdaptiveSparseGrid._writeConvergencePoint(self,runPoint)
