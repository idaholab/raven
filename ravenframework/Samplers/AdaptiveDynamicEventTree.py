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
  This module contains the Adaptive Dynamic Event Tree and
  the Adaptive Hybrid Dynamic Event Tree sampling strategies

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import numpy as np
from operator import mul
from functools import reduce
import xml.etree.ElementTree as ET
import itertools
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .DynamicEventTree import DynamicEventTree
from .LimitSurfaceSearch import LimitSurfaceSearch
from ..utils import utils
from ..utils import TreeStructure as ETS
#Internal Modules End--------------------------------------------------------------------------------

class AdaptiveDynamicEventTree(DynamicEventTree, LimitSurfaceSearch):
  """
    This class is aimed to perform a supervised Adaptive Dynamic Event Tree sampling strategy
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
    inputSpecification = super(AdaptiveDynamicEventTree, cls).getInputSpecification()

    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    DynamicEventTree.__init__(self)     # init DET
    LimitSurfaceSearch.__init__(self)   # init Adaptive
    self.detAdaptMode         = 1       # Adaptive Dynamic Event Tree method (=1 -> DynamicEventTree as hybridsampler and subsequent LimitSurfaceSearch,=2 -> DynamicEventTree online adaptive)
    self.noTransitionStrategy = 1       # Strategy in case no transitions have been found by DET (1 = 'Probability MC', 2 = Increase the grid exploration)
    self.insertAdaptBPb       = True    # Add Probabability THs requested by adaptive in the initial grid (default = False)
    self.startAdaptive        = False   # Flag to trigger the begin of the adaptive limit surface search
    self.adaptiveReady        = False   # Flag to store the response of the LimitSurfaceSearch.localStillReady method
    self.investigatedPoints   = []      # List containing the points that have been already investigated
    self.completedHistCnt   = 1         # Counter of the completed histories
    self.hybridDETstrategy  = None      # Integer flag to turn the hybrid strategy on:
                                        # None -> No hybrid approach,
                                        # 1    -> the epistemic variables are going to be part of the limit surface search
                                        # 2    -> the epistemic variables are going to be treated by a normal hybrid DET approach and the LimitSurface search
                                        #         will be performed on each epistemic tree (n LimitSurfaces)
    self.foundEpistemicTree = False     # flag that testifies if an epistemic tree has been found (Adaptive Hybrid DET)
    self.actualHybridTree   = ''        # name of the root tree used in self.hybridDETstrategy=2 to check which Tree needs to be used for the current LS search
    self.sortedListOfHists  = []        # sorted list of histories

  @staticmethod
  def _checkIfRunning(treeValues):
    """
      Static method (no self) that checks if a job is running
      @ In, treeValues, TreeStructure.Node, the node in which the running info are stored
      @ Out, _checkIfRunning, bool, is it running?
    """
    return not treeValues['runEnded']

  @staticmethod
  def _checkEnded(treeValues):
    """
      Static method (no self) that checks if a job finished to run
      @ In, treeValues, TreeStructure.Node, the node in which the running info are stored
      @ Out, _checkEnded, bool, is it finished?
    """
    return treeValues['runEnded']

  @staticmethod
  def _checkCompleteHistory(treeValues):
    """
      Static method (no self) that checks if a 'branch' represents a completed history
      @ In, treeValues, TreeStructure.Node, the node in which the running info are stored
      @ Out, _checkCompleteHistory, bool, is it a completed history (hit the last threshold?)
    """
    return treeValues['completedHistory']

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, needDict, dict, dictionary listing needed objects
    """
    #adaptNeedInst = self.limitSurfaceInstances.values()[-1]._localWhatDoINeed()
    needDict = dict(itertools.chain(LimitSurfaceSearch._localWhatDoINeed(self).items(),DynamicEventTree._localWhatDoINeed(self).items()))
    return needDict

  def _checkIfStartAdaptive(self):
    """
      Function that checks if the adaptive needs to be started (mode 1)
      @ In, None
      @ Out, None
    """
    if not self.startAdaptive:
      self.startAdaptive = True
      if len(self.lastOutput) == 0:
        self.startAdaptive = False
        return
      for treer in self.TreeInfo.values():
        for _ in treer.iterProvidedFunction(self._checkIfRunning):
          self.startAdaptive = False
          break
        if not self.startAdaptive:
          break

  def _checkClosestBranch(self):
    """
      Function that checks the closest branch already evaluated
      @ In, None
      @ Out, returnTuple, tuple, closest branch info:
        - if self.hybridDETstrategy and branch found         -> returnTuple = (valBranch,cdfValues,treer)
        - if self.hybridDETstrategy and branch not found     -> returnTuple = (None,cdfValues,treer)
        - if not self.hybridDETstrategy and branch found     -> returnTuple = (valBranch,cdfValues)
        - if not self.hybridDETstrategy and branch not found -> returnTuple = (None,cdfValues)
    """
    from sklearn import neighbors

    # compute cdf of sampled vars
    lowerCdfValues = {}
    cdfValues         = {}
    self.raiseADebug("Check for closest branch:")
    self.raiseADebug("_"*50)
    for key,value in self.values.items():
      self.raiseADebug("Variable name   : "+str(key))
      self.raiseADebug("Distribution name: "+str(self.toBeSampled[key]))
      if key not in self.epistemicVariables.keys():
        cdfValues[key] = self.distDict[key].cdf(value)
        try:
          index = utils.first(np.atleast_1d(np.asarray(self.branchProbabilities[key]) <= cdfValues[key]).nonzero())[-1]
          val = self.branchProbabilities[key][index]
        except (ValueError, IndexError):
          val = None
        lowerCdfValues[key] = val
        self.raiseADebug("CDF value       : "+str(cdfValues[key]))
        self.raiseADebug("Lower CDF found : "+str(lowerCdfValues[key]))
      self.raiseADebug("_"*50)
    #if hybrid DET, we need to find the correct tree that matches the values of the epistemic
    if self.hybridDETstrategy is not None:
      self.foundEpistemicTree, treer, compareDict = False, None, dict.fromkeys(self.epistemicVariables.keys(),False)
      for tree in self.TreeInfo.values():
        epistemicVars = tree.getrootnode().get("hybridsamplerCoordinate")[0]['SampledVars']
        for key in self.epistemicVariables.keys():
          compareDict[key] = utils.compare(epistemicVars[key],self.values[key])
        if all(compareDict.values()):
          # we found the right epistemic tree
          self.foundEpistemicTree, treer = True, tree
          break
    else:
      treer = utils.first(self.TreeInfo.values())

    # check if in the adaptive points already explored (if not push into the grid)
    if not self.insertAdaptBPb:
      candidatesBranch = []
      # check if adaptive point is better choice -> TODO: improve efficiency
      for invPoint in self.investigatedPoints:
        pbth = [invPoint[self.toBeSampled[key]] for key in cdfValues.keys()]
        if all(i <= pbth[cnt] for cnt,i in enumerate(cdfValues.values())):
          candidatesBranch.append(invPoint)
      if len(candidatesBranch) > 0:
        if None in lowerCdfValues.values():
          lowerCdfValues = candidatesBranch[0]
        for invPoint in candidatesBranch:
          pbth = [invPoint[self.toBeSampled[key]] for key in cdfValues.keys()]
          if all(i >= pbth[cnt] for cnt,i in enumerate(lowerCdfValues.values())):
            lowerCdfValues = invPoint
    # Check if The adaptive point requested is outside the so far run grid; in case return None
    # In addition, if Adaptive Hybrid DET, if treer is None, we did not find any tree
    #              in the epistemic space => we need to create another one
    if None in lowerCdfValues.values() or treer is None:
      if self.hybridDETstrategy is not None:
        returnTuple = None, cdfValues, treer
      else:
        returnTuple = None, cdfValues
      return returnTuple

    nntrain, mapping = None, {}
    for ending in treer.iterProvidedFunction(self._checkEnded):
      #already ended branches, create training set for nearest algorithm (take coordinates <= of cdfValues) -> TODO: improve efficiency
      pbth = [ending.get('SampledVarsPb')[key] for key in lowerCdfValues.keys()]
      if all(pbth[cnt] <= i for cnt,i in enumerate(lowerCdfValues.values())):
        if nntrain is None:
          nntrain = np.zeros((1,len(cdfValues.keys())))
          nntrain[0,:] = np.array(copy.copy(pbth))
        else:
          nntrain = np.concatenate((nntrain,np.atleast_2d(np.array(copy.copy(pbth)))),axis=0)
        mapping[nntrain.shape[0]] = ending
    if nntrain is not None:
      neigh = neighbors.NearestNeighbors(n_neighbors=len(mapping.keys()))
      neigh.fit(nntrain)
      valBranch = self._checkValidityOfBranch(neigh.kneighbors([list(lowerCdfValues.values())]),mapping)
      if self.hybridDETstrategy is not None:
        returnTuple = valBranch,cdfValues,treer
      else:
        returnTuple = valBranch,cdfValues
      return returnTuple
    else:
      returnTuple = (None,cdfValues,treer) if self.hybridDETstrategy is not None else (None,cdfValues)
      return returnTuple

  def _checkValidityOfBranch(self,branchSet,mapping):
    """
      Function that checks if the nearest branches found by method _checkClosestBranch are valid
      @ In, branchSet, tuple, tuple of branches
      @ In, mapping, dict, dictionary of candidate branches
      @ Out, validBranch, TreeStructure.Node, most valid branch (if not found, return None)
    """
    validBranch   = None
    idOfBranches  = branchSet[1][-1]
    for closestBranch in idOfBranches:
      if not mapping[closestBranch+1].get('completedHistory') and not mapping[closestBranch+1].get('happenedEvent'):
        validBranch = mapping[closestBranch+1]
        break
    return validBranch

  def _retrieveBranchInfo(self,branch):
    """
       Function that retrieves the key information from a branch to start a newer calculation
       @ In, branch, TreeStructure.Node, the branch to inquire
       @ Out, info, dict, the dictionary with information on the inputted branch
    """
    info = branch.getValues()
    info['actualBranchOnLevel'] = branch.numberBranches()
    info['parentNode']         = branch
    return info

  def _constructEndInfoFromBranch(self,model, myInput, info, cdfValues):
    """
      Method to construct the end information from the 'info' inputted
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ In, info, dict, dictionary of information at the end of a branch (information collected by the method _retrieveBranchInfo)
      @ In, cdfValues, dict, dictionary of CDF thresholds reached by the branch that just ended.
      @ Out, None
    """
    endInfo = info['parentNode'].get('endInfo')
    #del self.inputInfo
    self.counter           += 1
    self.branchCountOnLevel = info['actualBranchOnLevel']+1
    # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
    rname = info['parentNode'].get('name') + '-' + str(self.branchCountOnLevel)
    info['parentNode'].add('completedHistory', False)
    self.raiseADebug(str(rname))
    bcnt = self.branchCountOnLevel
    while info['parentNode'].isAnActualBranch(rname):
      bcnt += 1
      rname = info['parentNode'].get('name') + '-' + str(bcnt)
    # create a subgroup that will be appended to the parent element in the xml tree structure
    subGroup = ETS.HierarchicalNode(rname)
    subGroup.add('parent', info['parentNode'].get('name'))
    subGroup.add('name', rname)
    self.raiseADebug('cond pb = '+str(info['parentNode'].get('conditionalPb')))
    condPbC  = float(info['parentNode'].get('conditionalPb'))

    # Loop over  branchChangedParams (events) and start storing information,
    # such as conditional pb, variable values, into the xml tree object
    branchChangedParamValue = []
    branchChangedParamPb    = []
    branchParams            = []
    if endInfo:
      for key in endInfo['branchChangedParams'].keys():
        branchParams.append(key)
        branchChangedParamPb.append(endInfo['branchChangedParams'][key]['associatedProbability'][0])
        branchChangedParamValue.append(endInfo['branchChangedParams'][key]['oldValue'][0])
      subGroup.add('branchChangedParam',branchParams)
      subGroup.add('branchChangedParamValue',branchChangedParamValue)
      subGroup.add('branchChangedParamPb',branchChangedParamPb)
    # add conditional probability
    subGroup.add('conditionalPb',condPbC)
    # add initiator distribution info, start time, etc.
    subGroup.add('startTime', info['parentNode'].get('endTime'))
    # initialize the endTime to be equal to the start one... It will modified at the end of this branch
    subGroup.add('endTime', info['parentNode'].get('endTime'))
    # add the branchedLevel dictionary to the subgroup
    # branch calculation info... running, queue, etc are set here
    subGroup.add('runEnded',False)
    subGroup.add('running',False)
    subGroup.add('queue',True)
    subGroup.add('completedHistory', False)
    subGroup.add('happenedEvent', True)
    subGroup.add('triggeredVariable',info['parentNode'].get('triggeredVariable'))
    # Append the new branch (subgroup) info to the parentNode in the tree object
    info['parentNode'].appendBranch(subGroup)
    # Fill the values dictionary that will be passed into the model in order to create an input
    # In this dictionary the info for changing the original input is stored
    self.inputInfo.update({'prefix':rname,'endTimeStep':info['parentNode'].get('actualEndTimeStep'),
              'branchChangedParam':subGroup.get('branchChangedParam'),
              'branchChangedParamValue':subGroup.get('branchChangedParamValue'),
              'conditionalPb':subGroup.get('conditionalPb'),
              'startTime':info['parentNode'].get('endTime'),
              'happenedEvent':subGroup.get('happenedEvent'),
              'triggeredVariable':subGroup.get('triggeredVariable'),
              'RAVEN_parentID':subGroup.get('parent'),
              'RAVEN_isEnding':True})
    # add the newer branch name to the map
    self.rootToJob[rname] = self.rootToJob[subGroup.get('parent')]
    # check if it is a preconditioned DET sampling, if so add the relative information
    # it exists only in case an hybridDET strategy is activated
    precSampled = info['parentNode'].get('hybridsamplerCoordinate')
    if precSampled:
      self.inputInfo['hybridsamplerCoordinate'  ] = copy.deepcopy(precSampled)
      subGroup.add('hybridsamplerCoordinate', copy.copy(precSampled))
    # The probability Thresholds are stored here in the cdfValues dictionary... We are sure that they are whitin the ones defined in the grid
    # check is not needed
    self.inputInfo['initiatorDistribution' ] = [self.toBeSampled[key] for key in cdfValues.keys()]
    self.inputInfo['PbThreshold'           ] = list(cdfValues.values())
    self.inputInfo['ValueThreshold'        ] = [self.distDict[key].ppf(value) for key,value in cdfValues.items()]
    self.inputInfo['SampledVars'           ] = {}
    self.inputInfo['SampledVarsPb'         ] = {}
    for varname in self.standardDETvariables:
      self.inputInfo['SampledVars'  ][varname] = self.distDict[varname].ppf(cdfValues[varname])
      self.inputInfo['SampledVarsPb'][varname] = cdfValues[varname]
    # constant variables
    self._constantVariables()
    if precSampled:
      for precSample in precSampled:
        self.inputInfo['SampledVars'  ].update(precSample['SampledVars'])
        self.inputInfo['SampledVarsPb'].update(precSample['SampledVarsPb'])
    pointPb = reduce(mul,[it for sub in [pre['SampledVarsPb'].values() for pre in precSampled ] for it in sub] if precSampled else [1.0])
    self.inputInfo['PointProbability' ] = pointPb*subGroup.get('conditionalPb')
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
    self.inputInfo.update({'ProbabilityWeight-'+key.strip():value for key,value in self.inputInfo['SampledVarsPb'].items()})
    # add additional edits if needed
    model.getAdditionalInputEdits(self.inputInfo)
    # Add the new input path into the RunQueue system
    newInputs = {'args':[str(self.type)], 'kwargs': dict(self.inputInfo)}
    self.RunQueue['queue'].append(newInputs)
    self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
    for key,value in self.inputInfo.items():
      subGroup.add(key,copy.copy(value))
    if endInfo:
      subGroup.add('endInfo',copy.deepcopy(endInfo))

  def localStillReady(self,ready):
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    if self.counter == 0:
      return True
    if len(self.RunQueue['queue']) != 0:
      detReady = True
    else:
      detReady = False
    # since the RunQueue is empty, let's check if there are still branches running => if not => start the adaptive search
    self._checkIfStartAdaptive()
    if self.startAdaptive:
      data = self.lastOutput.asDataset()
      endingData = data.where(data['RAVEN_isEnding']==True,drop=True)
      numCompletedHistories = len(endingData['RAVEN_isEnding'])
      if numCompletedHistories > self.completedHistCnt:
        lastOutDict = {key:endingData[key].values for key in endingData.keys()}
      if numCompletedHistories > self.completedHistCnt:
        actualLastOutput      = self.lastOutput
        self.lastOutput       = copy.deepcopy(lastOutDict)
        ready                 = LimitSurfaceSearch.localStillReady(self,ready)
        self.lastOutput       = actualLastOutput
        self.completedHistCnt = numCompletedHistories
        self.raiseAMessage("Completed full histories are "+str(self.completedHistCnt))
      else:
        ready = False
      self.adaptiveReady = ready
      if ready or detReady:
        return True
      else:
        return False
    return detReady

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    if self.startAdaptive == True and self.adaptiveReady == True:
      LimitSurfaceSearch.localGenerateInput(self,model,myInput)
      #the adaptive sampler created the next point sampled vars
      #find the closest branch
      if self.hybridDETstrategy is not None:
        closestBranch, cdfValues, treer = self._checkClosestBranch()
      else:
        closestBranch, cdfValues = self._checkClosestBranch()
      if closestBranch is None:
        self.raiseADebug('An usable branch for next candidate has not been found => create a parallel branch!')
      # add pbthresholds in the grid
      investigatedPoint = {}
      for key,value in cdfValues.items():
        try:
          ind = utils.first(np.atleast_1d(np.asarray(self.branchProbabilities[key]) <= value).nonzero())[-1]
        except (IndexError, ValueError):
          ind = 0
        if value not in self.branchProbabilities[key]:
          self.branchProbabilities[key].insert(ind,value)
          self.branchValues[key].insert(ind,self.distDict[key].ppf(value))
        investigatedPoint[key] = value
      # collect investigated point
      self.investigatedPoints.append(investigatedPoint)
      if closestBranch:
        info = self._retrieveBranchInfo(closestBranch)
        self._constructEndInfoFromBranch(model, myInput, info, cdfValues)
      else:
        # create a new tree, since there are no branches that are close enough to the adaptive request
        elm = ETS.HierarchicalNode(self.name + '_' + str(len(self.TreeInfo.keys())+1))
        elm.add('name', self.name + '_'+ str(len(self.TreeInfo.keys())+1))
        elm.add('startTime', 0.0)
        # Initialize the endTime to be equal to the start one...
        # It will modified at the end of each branch
        elm.add('endTime', 0.0)
        elm.add('runEnded',False)
        elm.add('running',True)
        elm.add('queue',False)
        elm.add('completedHistory', False)
        branchedLevel = {}
        for key,value in cdfValues.items():
          branchedLevel[key] = utils.first(np.atleast_1d(np.asarray(self.branchProbabilities[key]) == value).nonzero())[-1]
        # The dictionary branchedLevel is stored in the xml tree too. That's because
        # the advancement of the thresholds must follow the tree structure
        elm.add('branchedLevel', branchedLevel)
        if self.hybridDETstrategy is not None and not self.foundEpistemicTree:
          # adaptive hybrid DET and not found a tree in the epistemic space
          # take the first tree and modify the hybridsamplerCoordinate
          hybridSampled = copy.deepcopy(utils.first(self.TreeInfo.values()).getrootnode().get('hybridsamplerCoordinate'))
          for hybridStrategy in hybridSampled:
            for key in self.epistemicVariables.keys():
              if key in hybridStrategy['SampledVars'].keys():
                self.raiseADebug("epistemic var " + str(key)+" value = "+str(self.values[key]))
                hybridStrategy['SampledVars'][key]   = copy.copy(self.values[key])
                hybridStrategy['SampledVarsPb'][key] = self.distDict[key].pdf(self.values[key])
                hybridStrategy['prefix'] = len(self.TreeInfo.values())+1
            # TODO: find a strategy to recompute the probability weight here (for now == PointProbability)
            hybridStrategy['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
            hybridStrategy['ProbabilityWeight'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
          elm.add('hybridsamplerCoordinate', hybridSampled)
        self.inputInfo.update({'ProbabilityWeight-'+key.strip():value for key,value in self.inputInfo['SampledVarsPb'].items()})
        # Here it is stored all the info regarding the DET => we create the info for all the branchings and we store them
        self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys())+1)] = ETS.HierarchicalTree(elm)
        self._createRunningQueueBeginOne(self.TreeInfo[self.name + '_' + str(len(self.TreeInfo.keys()))],branchedLevel, model,myInput)
    return DynamicEventTree.localGenerateInput(self,model,myInput)

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    #check if the hybrid DET has been activated, in case remove the nodes and treat them separaterly
    hybridNodes = xmlNode.findall("HybridSampler")
    if len(hybridNodes) != 0:
      # check the type of hybrid that needs to be performed
      limitSurfaceHybrid = False
      for elm in hybridNodes:
        samplType = elm.attrib['type'] if 'type' in elm.attrib.keys() else None
        if samplType == 'LimitSurface':
          if len(hybridNodes) != 1:
            self.raiseAnError(IOError,'if one of the HybridSampler is of type "LimitSurface", it can not be combined with other strategies. Only one HybridSampler node can be inputted!')
          limitSurfaceHybrid = True
      if limitSurfaceHybrid == True:
        #remove the elements from original xmlNode and check if the types are compatible
        for elm in hybridNodes:
          xmlNode.remove(elm)
        self.hybridDETstrategy = 1
      else:
        self.hybridDETstrategy = 2
      if self.hybridDETstrategy == 2:
        self.raiseAnError(IOError, 'The sheaf of LSs for the Adaptive Hybrid DET is not yet available. Use type "LimitSurface"!')

    DynamicEventTree.localInputAndChecks(self,xmlNode, paramInput)
    # now we put back the nodes into the xmlNode to initialize the LimitSurfaceSearch with those variables as well
    for elm in hybridNodes:
      for child in elm:
        if limitSurfaceHybrid == True:
          xmlNode.append(child)
        if child.tag in ['variable','Distribution']:
          self.epistemicVariables[child.attrib['name']] = None
    LimitSurfaceSearch._readMoreXMLbase(self,xmlNode)
    LimitSurfaceSearch.localInputAndChecks(self,xmlNode, paramInput)
    if 'mode' in xmlNode.attrib.keys():
      if   xmlNode.attrib['mode'].lower() == 'online':
        self.detAdaptMode = 2
      elif xmlNode.attrib['mode'].lower() == 'post':
        self.detAdaptMode = 1
      else:
        self.raiseAnError(IOError,'unknown mode ' + xmlNode.attrib['mode'] + '. Available are "online" and "post"!')
    if 'noTransitionStrategy' in xmlNode.attrib.keys():
      if xmlNode.attrib['noTransitionStrategy'].lower() == 'mc':
        self.noTransitionStrategy = 1
      elif xmlNode.attrib['noTransitionStrategy'].lower() == 'grid':
        self.noTransitionStrategy = 2
      else:
        self.raiseAnError(IOError,'unknown noTransitionStrategy '+xmlNode.attrib['noTransitionStrategy']+'. Available are "mc" and "grid"!')
    if 'updateGrid' in xmlNode.attrib.keys():
      if utils.stringIsTrue(xmlNode.attrib['updateGrid']):
        self.insertAdaptBPb = True
    # we add an artificial threshold because I need to find a way to prepend a rootbranch into a Tree object
    for  val in self.branchProbabilities.values():
      if min(val) != 1e-3:
        val.insert(0, 1e-3)

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distributions and functions.
      @ In, availDist, dict, dict of distributions
      @ In, availableFunc, dict, dict of functions
      @ Out, None
    """
    DynamicEventTree._generateDistributions(self,availableDist,availableFunc)

  def localInitialize(self,solutionExport = None):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, solutionExport, DataObjects, optional, a PointSet to hold the solution (a list of limit surface points)
      @ Out, None
    """
    if self.detAdaptMode == 2:
      self.startAdaptive = True
    # we first initialize the LimitSurfaceSearch sampler
    LimitSurfaceSearch.localInitialize(self,solutionExport=solutionExport)
    if self.hybridDETstrategy is not None:
      # we are running an adaptive hybrid DET and not only an adaptive DET
      if self.hybridDETstrategy == 1:
        gridVector = self.limitSurfacePP.gridEntity.returnParameter("gridVectors")
        # construct an hybrid DET through an XML node
        distDict, xmlNode = {}, ET.fromstring('<InitNode> <HybridSampler type="Grid" name="none"/> </InitNode>')
        for varName, dist in self.distDict.items():
          if varName.replace('<distribution>','') in self.epistemicVariables.keys():
            # found an epistemic
            varNode  = ET.Element('Distribution' if varName.startswith('<distribution>') else 'variable',{'name':varName.replace('<distribution>','')})
            varNode.append(ET.fromstring("<distribution>"+dist.name.strip()+"</distribution>"))
            distDict[dist.name.strip()] = self.distDict[varName]
            varNode.append(ET.fromstring('<grid construction="custom" type="value">'+' '.join([str(elm) for elm in utils.first(gridVector.values())[varName.replace('<distribution>','')]])+'</grid>'))
            xmlNode.find("HybridSampler").append(varNode)
        #TODO, need to pass real paramInput
        self._localInputAndChecksHybrid(xmlNode, paramInput=None)
        for hybridsampler in self.hybridStrategyToApply.values():
          hybridsampler._generateDistributions(distDict, {})
    DynamicEventTree.localInitialize(self)
    if self.hybridDETstrategy == 2:
      self.actualHybridTree = utils.first(self.TreeInfo.keys())
    self._endJobRunnable    = sys.maxsize

  def generateInput(self,model,oldInput):
    """
      This method has to be overwritten to provide the specialization for the specific sampler
      The model instance in might be needed since, especially for external codes,
      only the code interface possesses the dictionary for reading the variable definition syntax
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, generateInput, tuple(0,list), list containing the new inputs -in reality it is the model that returns this; the Sampler generates the value to be placed in the input of the model.
    """
    return DynamicEventTree.generateInput(self, model, oldInput)

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case (DET), The function reads the
      information from the ended calculation, updates the working variables, and
      creates the new inputs for the next branches
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    returncode = DynamicEventTree.localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=False)
    forceEvent = True if self.startAdaptive else False
    if returncode:
      self._createRunningQueue(model,myInput, forceEvent)
