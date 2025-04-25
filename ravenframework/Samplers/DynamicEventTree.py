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
  This module contains the Dynamic Event Tree and
  the Hybrid Dynamic Event Tree sampling strategies

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import numpy as np
from operator import mul
from functools import reduce
import xml.etree.ElementTree as ET
import itertools
from collections import Counter
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Grid import Grid
from .MonteCarlo import MonteCarlo
from .Stratified import Stratified
from .Sampler import Sampler
from ..utils import utils
from ..utils import InputData, InputTypes
from ..utils import TreeStructure as ETS
from ..Realizations import RealizationBatch, Realization
#Internal Modules End-------------------------------------------------------------------------------

class DynamicEventTree(Grid):
  """
    DYNAMIC EVENT TREE Sampler (DET)
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
    inputSpecification = super(DynamicEventTree, cls).getInputSpecification()

    inputSpecification.addParam("printEndXmlSummary", InputTypes.StringType)
    inputSpecification.addParam("maxSimulationType", InputTypes.FloatType)
    inputSpecification.addParam("removeXmlBranchInfo", InputTypes.StringType)

    oldSub = inputSpecification.popSub("Distribution")
    newDistributionInput = InputData.parameterInputFactory("Distribution", baseNode=oldSub)
    gridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.StringType)
    gridInput.addParam("type", InputTypes.StringType)
    gridInput.addParam("construction", InputTypes.StringType)
    gridInput.addParam("steps", InputTypes.IntegerType)

    newDistributionInput.addSub(gridInput)
    inputSpecification.addSub(newDistributionInput)

    #Strict mode off because basically this allows things to be passed to
    # sub Samplers, which will be checked later.
    hybridSamplerInput = InputData.parameterInputFactory("HybridSampler", strictMode=False)
    hybridSamplerInput.addParam("type", InputTypes.StringType)

    for nodeName in ['variable','Distribution']:
      nodeInput = InputData.parameterInputFactory(nodeName, strictMode=False)
      nodeInput.addParam("name", InputTypes.StringType)
      hybridSamplerInput.addSub(nodeInput)
    inputSpecification.addSub(hybridSamplerInput)

    return inputSpecification

  def __init__(self):
    """
    Default Constructor that will initialize member variables with reasonable
    defaults or empty lists/dictionaries where applicable.
    @ In, None
    @ Out, None
    """
    Grid.__init__(self)
    self.onlySampleAfterCollecting = True # see note in Steps.MultiRun about the not-point-sampler loop
    # Working directory (Path of the directory in which all the outputs,etc. are stored)
    self.workingDir = ""
    # (optional) if not present, the sampler will not change the relative keyword in the input file
    self.maxSimulTime = None
    # print the xml tree representation of the dynamic event tree calculation
    # see variable 'self.TreeInfo'
    self.printEndXmlSummary = False
    # flag to control if the branch info xml file needs to be removed after reading
    self.removeXmlBranchInfo = True
    # Dictionary of the probability bins for each distribution that have been
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    self.branchProbabilities = {}
    # Dictionary of the Values' bins for each distribution that have been
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    # these are the invCDFs of the PBs inputted in branchProbabilities (if ProbabilityThresholds have been inputted)
    self.branchValues = {}
    # List of Dictionaries of the last probability bin level (position in the array) reached for each distribution ('distName':IntegerValue)
    # This container is a working dictionary. The branchedLevels are stored in the xml tree "self.TreeInfo" since they must track
    # the evolution of the dynamic event tree
    self.branchedLevel = []
    # Counter for the branch needs to be run after a calculation branched (it is a working variable)
    self.branchCountOnLevel = 0
    # Dictionary tha contains the actual branching info
    # (i.e. distribution that triggered, values of the variables that need to be changed, etc)
    self.actualBranchInfo = {}
    # Parent Branch end time (It's a working variable used to set up the new branches need to be run.
    #   The new branches' start time will be the end time of the parent branch )
    self.actualEndTime = 0.0
    # Parent Branch end time step (It's a working variable used to set up the new branches need to be run.
    #  The end time step is used to construct the filename of the restart files needed for restart the new branch calculations)
    self.actualEndTs = 0
    # Xml tree object. It stored all the info regarding the DET. It is in continue evolution during a DET calculation
    self.TreeInfo = None
    # List of Dictionaries. It is a working variable used to store the information needed to create branches from a Parent Branch
    self.endInfo = []
    # Queue system. The inputs are waiting to be run are stored in this queue dictionary
    self.RunQueue = {}
    # identifiers of the inputs in queue (name of the history... for example DET_1,1,1)
    self.RunQueue['identifiers'] = []
    # Corresponding inputs
    self.RunQueue['queue'] = []
    # mapping from jobID to rootname in TreeInfo {jobID:rootName}
    self.rootToJob = {}
    # dictionary of Hybrid Samplers available
    self.hybridSamplersAvail = {'MonteCarlo':MonteCarlo,'Stratified':Stratified,'Grid':Grid}
    # dictionary of inputted hybridsamplers need to be applied
    self.hybridStrategyToApply = {}
    # total number of hybridsampler samples (combination of all different hybridsampler strategy)
    self.hybridNumberSamplers = 0
    # List of variables that represent the aleatory space
    self.standardDETvariables = []
    # Dictionary of variables that represent the epistemic space (hybrid det).
    # Format => {'epistemicVarName':{'HybridTree name':value}}
    self.epistemicVariables = {}
    # Dictionary (mapping) between the fully correlated variables and the "epistemic"
    # variable representation (variable name) in the input file. For example,
    # {'var1':'var1,var2','var2':'var1,var2'}
    self.fullyCorrelatedEpistemicToVar = {}
    # Dictionary to store constants, e.g., {constantName: constantValue} from hybrid sampler
    self.hybridConstants = {}
    # Dictionary to store dependent variables, e.g., {dependentVariableName:value} from hybrid sampler
    self.hybridDependentSample = {}

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implmented here because this Sampler requests special objects
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed
    """
    needDict = Sampler._localWhatDoINeed(self)
    for hybridsampler in self.hybridStrategyToApply.values():
      preNeedDict = hybridsampler.whatDoINeed()
      for key,value in preNeedDict.items():
        if key not in needDict.keys():
          needDict[key] = []
        needDict[key] = needDict[key] + value
    return needDict

  def localStillReady(self, ready):
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    self._endJobRunnable = max((len(self.RunQueue['queue']), 1))
    if(len(self.RunQueue['queue']) != 0 or self.counters['samples'] == 0):
      ready = True
    else:
      if self.printEndXmlSummary:
        name = os.path.join(self.workingDir, f'{self.name}_outputSummary.xml')
        with open(name, 'w', encoding='utf-8') as myFile:
          for treeNode in self.TreeInfo.values():
            treeNode.writeNodeTree(myFile)
      ready = False
    return ready

  def _retrieveParentNode(self,idj):
    """
      Grants access to the parent node of a particular job
      @ In, idj, string, the identifier of a job object
      @ Out, parentNode, TreeStructure.Node, the parent node of the job linked to idj
    """
    if(idj == self.TreeInfo[self.rootToJob[idj]].getrootnode().name):
      parentNode = self.TreeInfo[self.rootToJob[idj]].getrootnode()
    else:
      parentNode = list(self.TreeInfo[self.rootToJob[idj]].getrootnode().iter(idj))[0]
    return parentNode

  def localFinalizeActualSampling(self,jobObject,model,myInput,genRunQueue=True):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case (DET), The function reads the
      information from the ended calculation, updates the working variables, and
      creates the new inputs for the next branches
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ In, genRunQueue, bool, optional, True if the RunQueue needs to be updated
      @ Out, None
    """
    # FIXME it doesn't appear that genRunQueue can ever be false ... Sampler calls this method without arguments
    self.workingDir = model.workingDir
    # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    parentNode = self._retrieveParentNode(jobObject.identifier)
    # set runEnded and running to true and false respectively
    parentNode.add('runEnded',True)
    parentNode.add('running',False)


    # Read the branch info from the parent calculation (just ended calculation)
    # This function stores the information in the dictionary 'self.actualBranchInfo'
    # If no branch info, this history is concluded => return

    ## There are two ways to get at the working directory from the job instance
    ## and both feel a bit hacky and fragile to changes in the Runner classes.
    ## They are both listed below and the second inevitably stems from the first.
    ## I am wondering if there is a reason we cannot use the model.workingDir
    ## from above here? Granted the job instance should have a snapshot of
    ## whatever the model's current working directory was for that evaluation,
    ## and it could have changed in the meantime, so I will keep this as is for
    ## now, but note this should be re-evaluated in the future. -- DPM 4/12/17
    # codeModel = jobObject.args[0]
    # jobWorkingDir = codeModel.workingDir
    rlz = jobObject.args[3]
    info = rlz.inputInfo
    stepWorkingDir = info['WORKING_DIR']
    jobWorkingDir = os.path.join(stepWorkingDir, info['prefix'] if 'prefix' in info else '1')

    ## This appears to be the same, so I am switching to the model's workingDir
    ## since it is more directly available and less change to how data is stored
    ## in the args of a job instance. -- DPM 4/12/17
    if not self.__readBranchInfo(jobObject.getMetadata()['outfile'], jobWorkingDir):
      parentNode.add('completedHistory', True)
      parentNode.add('endTime',self.actualEndTime)
      return False
    # Collect the branch info in a multi-level dictionary
    endInfo = {
        'endTime': self.actualEndTime,
        'endTimeStep': self.actualEndTs,
        'branchDist': list(self.actualBranchInfo.keys())[0]
    }
    endInfo['branchChangedParams'] = self.actualBranchInfo[endInfo['branchDist']]
    # check if RELAP7 mode is activated, in case prepend the "<distribution>" string
    if any("<distribution>" in s for s in self.branchProbabilities):
      endInfo['branchDist'] = list(self.toBeSampled.keys())[list(self.toBeSampled.values()).index(endInfo['branchDist'])]
      #endInfo['branchDist'] = "<distribution>"+endInfo['branchDist']
    parentNode.add('actualEndTimeStep',self.actualEndTs)
    # # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    endInfo['parentNode'] = parentNode
    # get the branchedLevel dictionary
    branchedLevel = {}
    self.raiseMarker(f'branch {rlz.inputInfo["prefix"]}')
    self.raiseWhatsThis('parent', endInfo['parentNode'])
    self.raiseWhatsThis('branchProbs', self.branchProbabilities)
    self.raiseWhatsThis('epistemics', self.epistemicVariables)
    for distk, distpb in endInfo['parentNode'].get('SampledVarsPb').items(): #zip(endInfo['parentNode'].get('SampledVarsPb'), endInfo['parentNode'].get('SampledVarsPb').values()):
      self.raiseMarker(f'disk {distk} distpb {distpb}')
      if all(distk not in checklist for checklist in (self.epistemicVariables, self.constants, self.hybridConstants)):
        branchedLevel[distk] = utils.first(np.atleast_1d(np.asarray(self.branchProbabilities[distk]) == distpb).nonzero())[-1]
        self.raiseMarker('... found aleatoric var')
      else:
        self.raiseMarker('... not an aleatoric var')
    if not branchedLevel:
      self.raiseAnError(RuntimeError, f'branchedLevel of node {jobObject.identifier} not found!')
    if endInfo['branchDist'] not in branchedLevel:
      self.raiseAnError(RuntimeError,'<Distribution_trigger> (aka variable) "{}"  in job "{}" not found in the DET set of variables. Available variables are "{}". Problem with the alias?'.format(endInfo['branchDist'],jobObject.identifier, ','.join(branchedLevel.keys())))

    # Loop of the parameters that have been changed after a trigger gets activated
    if 'None' not in endInfo['branchChangedParams']:
      for key in endInfo['branchChangedParams']:
        endInfo['n_branches'] = 1 + int(len(endInfo['branchChangedParams'][key]['actualValue']))
        if(len(endInfo['branchChangedParams'][key]['actualValue']) > 1):
          #  Multi-Branch mode => the resulting branches from this parent calculation (just ended)
          # will be more then 2
          # unchangedPb = probability (not conditional probability yet) that the event does not occur
          unchangedPb = 0.0
          try:
            # changed_pb = probability (not conditional probability yet) that the event A occurs and the final state is 'alpha' """
            for pb in range(len(endInfo['branchChangedParams'][key]['associatedProbability'])):
              unchangedPb = unchangedPb + endInfo['branchChangedParams'][key]['associatedProbability'][pb]
          except KeyError:
            self.raiseAWarning("KeyError:"+str(key))
          if(unchangedPb <= 1):
            endInfo['branchChangedParams'][key]['unchangedPb'] = 1.0-unchangedPb
          else:
            self.raiseAWarning("unchangedPb > 1:"+str(unchangedPb))
        else:
          # Two-Way mode => the resulting branches from this parent calculation (just ended) = 2
          if branchedLevel[endInfo['branchDist']] > len(self.branchProbabilities[endInfo['branchDist']])-1:
            pb = 1.0
          else:
            pb = self.branchProbabilities[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]
          endInfo['branchChangedParams'][key]['unchangedPb'] = 1.0 - pb
          endInfo['branchChangedParams'][key]['associatedProbability'] = [pb]
    else:
      endInfo['n_branches'] = 2
      if branchedLevel[endInfo['branchDist']] > len(self.branchProbabilities[endInfo['branchDist']])-1:
        pb = 1.0
      else:
        pb = self.branchProbabilities[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]
      endInfo['branchChangedParams']['None']['unchangedPb'] = 1.0 - pb
      endInfo['branchChangedParams']['None']['associatedProbability'] = [pb]
    self.branchCountOnLevel = 0
    # # set runEnded and running to true and false respectively
    # The branchedLevel counter is updated
    if branchedLevel[endInfo['branchDist']] < len(self.branchProbabilities[endInfo['branchDist']]):
      branchedLevel[endInfo['branchDist']] += 1

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
    if genRunQueue:
      self._createRunningQueue(rlz, model, myInput)
      self._endJobRunnable = len(self.RunQueue['identifiers'])
    return True

  def computeConditionalProbability(self,index=None):
    """
      Function to compute Conditional probability of the branches that are going to be run.
      The conditional probabilities are stored in the self.endInfo object
      @ In, index, int, optional, position in the self.endInfo list (optional). Default = 0
      @ Out, None
    """
    if not index:
      index = len(self.endInfo)-1
    try:
      parentCondPb = self.endInfo[index]['parentNode'].get('conditionalPb')
      if parentCondPb is None:
        self.raiseAnError(Exception, "parent node conditional pb not found. It should not happen!!!")
    except KeyError:
      parentCondPb = 1.0
    # for all the branches the conditional pb is computed
    # unchangedConditionalPb = Conditional Probability of the branches in which the event has not occurred
    # changedConditionalPb   = Conditional Probability of the branches in which the event has occurred
    for key in self.endInfo[index]['branchChangedParams']:
      self.endInfo[index]['branchChangedParams'][key]['changedConditionalPb'] = []
      self.endInfo[index]['branchChangedParams'][key]['unchangedConditionalPb'] = parentCondPb*float(self.endInfo[
        index]['branchChangedParams'][key]['unchangedPb'])
      for pb in range(len(self.endInfo[index]['branchChangedParams'][key]['associatedProbability'])):
        self.endInfo[index]['branchChangedParams'][key]['changedConditionalPb'].append(parentCondPb*float(self.endInfo[
          index]['branchChangedParams'][key]['associatedProbability'][pb]))

  def __readBranchInfo(self,outBase=None,currentWorkingDir=None):
    """
      Function to read the Branching info that comes from a Model
      The branching info (for example, distribution that triggered, parameters must be changed, etc)
      are supposed to be in a xml format
      @ In, outBase, string, optional, it is the output root that, if present, is used to construct the file name the function is going to try reading.
      @ In, currentWorkingDir, string, optional, it is the current working directory. If not present, the branch info are going to be looked in the self.workingDir
      @ Out, branchPresent, bool, true if the info are present (a set of new branches need to be run), false if the actual parent calculation reached an end point
    """
    # Remove all the elements from the info container
    del self.actualBranchInfo
    branchPresent = False
    self.actualBranchInfo = {}
    # Construct the file name adding the outBase root if present
    filename   = outBase + "_actual_branch_info.xml" if outBase else "actual_branch_info.xml"
    workingDir = currentWorkingDir if currentWorkingDir is not None else self.workingDir

    if not os.path.isabs(filename):
      filename = os.path.join(workingDir,filename)

    if not os.path.exists(filename):
      self.raiseADebug('branch info file ' + os.path.basename(filename) +' has not been found. => No Branching.')
      return branchPresent
    # Parse the file and create the xml element tree object
    #try:
    branchInfoTree = ET.parse(filename)
    self.raiseADebug('Done parsing '+filename)
    root = branchInfoTree.getroot()
    # Check if endTime and endTimeStep (time step)  are present... In case store them in the relative working vars
    #try: #Branch info written out by program, so should always exist.
    self.actualEndTime = float(root.attrib['end_time'])
    self.actualEndTs   = int(root.attrib['end_ts']) if  'end_ts' in root.attrib.keys() else -1
    #except? pass
    # Store the information in a dictionary that has as keywords the distributions that triggered
    for node in root:
      distTrigger = root.findall(".//Distribution_trigger")

      if not len(distTrigger):
        self.raiseAnError(Exception, '"Distribution_trigger" node has not been found in file: '+str(filename))
      elif len(distTrigger) > 1:
        self.raiseAWarning ( 'More then one "Distribution_trigger" node have been found in file: '+str(filename)+'. Grepping the first one only!')
      node = distTrigger[0]
      distName = node.attrib['name'].strip()
      self.actualBranchInfo[distName] = {}
      variables = node.findall(".//Variable")
      if len(variables) > 0:
        for child in variables:
          varName = child.text.strip()
          varType = child.attrib.get('type','auxiliar').strip()
          newValue = child.attrib.get('actual_value',None)
          oldValue = child.attrib.get('old_value',None)
          multiBranchPb = child.attrib.get('probability',None)
          if newValue is None:
            self.raiseAnError('"actual_value" is not present in the branch info file: '+str(filename)+'!')
          if oldValue is None:
            self.raiseAnError('"old_value" is not present in the branch info file: '+str(filename)+'!')
          newValue = [elm.strip() for elm in newValue.split()]
          if len(newValue) > 1:
            if multiBranchPb is None:
              self.raiseAnError(Exception, 'Multiple entries have been provided for "actual_value" (space separated value) but no "probability" attribute has been found!')
            multiBranchPb = [elm.strip() for elm in multiBranchPb.split()]
            if len(multiBranchPb) != len(newValue):
              self.raiseAnError(Exception, 'Multiple entries have been provided for "actual_value" (space separated value) but the number of entries in "probability" attribute does not match!')
            multiBranchPb = [utils.floatConversion(elm) for elm in multiBranchPb]
            if None in multiBranchPb:
              self.raiseAnError(ValueError, 'One of the entries in "probability" attribute can not be converted in float!')
          if multiBranchPb is not None and len(newValue) == 1:
            self.raiseAnError(Exception, 'Attribute "probability" has been inputted but no multi-branch detected.')
          # store
          self.actualBranchInfo[distName][varName] = {'varType':varType,'actualValue':newValue,'oldValue':oldValue.strip()}
          if multiBranchPb is not None:
            self.actualBranchInfo[distName][varName]['associatedProbability'] = multiBranchPb
      else:
        # not provided information regardind
        self.actualBranchInfo[distName]['None'] = {'varType':None,'actualValue':[None],'oldValue':None}
    # remove the file
    if self.removeXmlBranchInfo:
      os.remove(filename)
    branchPresent = True
    return branchPresent

  def _createRunningQueueBeginOne(self, rlz, rootTree, branchedLevel, model, myInput):
    """
      Method to generate the running internal queue for one point in the epistemic
      space. It generates the initial information to instantiate the root of a
      Deterministic Dynamic Event Tree.
      @ In, rlz, Realization, dict-like object to fill with sample
      @ In, rootTree, Node object, the rootTree of the single coordinate in the epistemic space.
      @ In, branchedLevel, dict, dictionary of the levels reached by the rootTree mapped in the internal grid dictionary (self.branchProbabilities)
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ Out, None
    """
    # add additional edits if needed
    model.getAdditionalInputEdits(rlz.inputInfo)
    precSampled = rootTree.getrootnode().get('hybridsamplerCoordinate')
    rootnode = rootTree.getrootnode()
    rname = rootnode.name
    rootnode.add('completedHistory', False)
    # Fill th values dictionary in
    if precSampled:
      rlz.inputInfo['hybridsamplerCoordinate'] = copy.deepcopy(precSampled)
    rlz.inputInfo['prefix'] = rname
    rlz.inputInfo['standardDETvariables'] = self.standardDETvariables
    rlz.inputInfo['initiatorDistribution'] = []
    rlz.inputInfo['triggeredVariable'] = 'None'
    rlz.inputInfo['happenedEventVarHistory'] = []
    rlz.inputInfo['PbThreshold'] = []
    rlz.inputInfo['ValueThreshold'] = []
    rlz.inputInfo['branchChangedParam'] = ['None']
    rlz.inputInfo['branchChangedParamValue'] = ['None']
    rlz.inputInfo['startTime'] = -sys.float_info.max
    rlz.inputInfo['endTimeStep'] = 0
    rlz.inputInfo['RAVEN_parentID'] = "None"
    rlz.inputInfo['RAVEN_isEnding'] = True
    rlz.inputInfo['conditionalPb'] = 1.0
    rlz.inputInfo['happenedEvent'] = False
    for branch, probs in self.branchProbabilities.items():
      rlz.inputInfo['initiatorDistribution'].append(self.toBeSampled[branch])
      rlz.inputInfo['PbThreshold'].append(probs[branchedLevel[branch]])
      rlz.inputInfo['ValueThreshold'].append(self.branchValues[branch][branchedLevel[branch]])
    for varname in self.standardDETvariables:
      rlz[varname] = self.branchValues[varname][branchedLevel[varname]]
      rlz.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[varname][branchedLevel[varname]]
    # constant variables
    # need to wrap in a batch # TODO is there a better way?
    batch = RealizationBatch(1)
    batch[0] = rlz
    self._constantVariables(batch)

    if precSampled:
      for precRlz in precSampled:
        precInfo = precRlz.inputInfo
        rlz.update(precRlz)
        rlz.inputInfo['SampledVarsPb'].update(precInfo['SampledVarsPb'])
    pointPb = reduce(mul,[it for sub in [pre.inputInfo['SampledVarsPb'].values() for pre in precSampled ] for it in sub] if precSampled else [1.0])
    rlz.inputInfo['PointProbability' ] = pointPb
    rlz.inputInfo['ProbabilityWeight'] = pointPb
    for name, pb in rlz.inputInfo['SampledVarsPb'].items():
      rlz.inputInfo[f'ProbabilityWeight-{name.strip()}'] = pb
    # rlz.inputInfo.update({'ProbabilityWeight-'+key.strip():value for key,value in rlz.inputInfo['SampledVarsPb'].items()})

    ##### REDUNDANT FUNCTIONALS #####
    self._functionalVariables(batch) # TODO batch or single?

    if(self.maxSimulTime):
      rlz.inputInfo['endTime'] = self.maxSimulTime

    # Add some useful variable naming in the input
    rlz.inputInfo.update(self.__createVariablesInfoForKwargs(model, rlz))

    # Add the new input path into the RunQueue system
    newInputs = {'args':[str(self.type)], 'rlz':rlz}
    for key,value in rlz.inputInfo.items():
      rootnode.add(key,copy.copy(value))
    self.RunQueue['queue'].append(newInputs)
    self.RunQueue['identifiers'].append(rlz.inputInfo['prefix'])
    self.rootToJob[rlz.inputInfo['prefix']] = rname
    del newInputs # TODO is this necessary? should go out of scope naturally.
    self.counters['samples'] += 1

  def _createRunningQueueBegin(self, rlz, model, myInput):
    """
      Method to generate the running internal queue for all the points in
      the epistemic space. It generates the initial information to
      instantiate the roots of all the N-D coordinates to construct multiple
      Deterministic Dynamic Event Trees.
      @ In, rlz, Realization, dict-like object to fill with sample
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ Out, None
    """
    # We construct the input for the first DET branch calculation'
    # Increase the counter
    # The root name of the xml element tree is the starting name for all the branches
    # (this root name = the user defined sampler name)
    # Get the initial branchedLevel dictionary (=> the list gets empty)
    branchedLevel = self.branchedLevel.pop(0)
    for rootTree in self.TreeInfo.values():
      # we actually need a separate rlz for each entry here, so make copies
      branchRlz = copy.deepcopy(rlz)
      self._createRunningQueueBeginOne(branchRlz,rootTree,branchedLevel, model,myInput)

  def _createRunningQueueBranch(self, baseRlz, model, myInput, forceEvent=False):
    """
      Method to generate the running internal queue right after a branch occurred
      It generates the the information to insatiate the branches' continuation of the Deterministic Dynamic Event Tree
      @ In, baseRlz, Realization, dict-like object to fill with sample
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ In, forceEvent, bool, if True the events are forced to happen (basically, the "unchanged event" is not created at all)
      @ Out, None
    """
    # The first DET calculation branch has already been run'
    # Start the manipulation:

    #  Pop out the last endInfo information and the branchedLevel
    branchedLevelParent = self.branchedLevel.pop(0)
    endInfo = self.endInfo.pop(0)
    self.branchCountOnLevel = 0
    # n_branches = number of branches need to be run
    nBranches = endInfo['n_branches']
    # Check if the distribution that just triggered hitted the last probability threshold .
    # In case we create a number of branches = endInfo['n_branches'] - 1 => the branch in
    # which the event did not occur is not going to be tracked
    self.raiseMarker('top crqb')
    if branchedLevelParent[endInfo['branchDist']] >= len(self.branchProbabilities[endInfo['branchDist']]):
      self.raiseADebug(f'Branch {endInfo["parentNode"].get("name")} hit last Threshold for distribution {endInfo["branchDist"]}')
      self.raiseADebug(f'Branch {endInfo["parentNode"].get("name")} is dead end.')
      self.branchCountOnLevel = 1
      nBranches -= 1
    else:
      if forceEvent:
        self.branchCountOnLevel = 1
        nBranches -= 1
    # Loop over the branches for which the inputs must be created
    for _ in range(nBranches):
      # since each branch will become a new input, each needs to be a fresh rlz
      rlz = copy.deepcopy(baseRlz)
      self.counters['samples'] += 1
      self.branchCountOnLevel += 1
      branchedLevel = copy.deepcopy(branchedLevelParent)
      # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
      rname = endInfo['parentNode'].get('name') + '-' + str(self.branchCountOnLevel)
      self.raiseMarker(f'rname "{rname}"')
      # create a subgroup that will be appended to the parent element in the xml tree structure
      subGroup = ETS.HierarchicalNode(rname)
      subGroup.add('parent', endInfo['parentNode'].get('name'))
      subGroup.add('name', rname)
      subGroup.add('completedHistory', False)
      # condPbUn = conditional probability event not occur
      # condPbC  = conditional probability event/s occur/s
      condPbUn = 0.0
      condPbC  = 0.0
      # Loop over  branchChangedParams (events) and start storing information,
      # such as conditional pb, variable values, into the xml tree object
      branchChangedParamValue = []
      branchChangedParamPb    = []
      branchParams            = []
      #subGroup.add('branchChangedParam',endInfo['branchChangedParams'].keys())

      for key in endInfo['branchChangedParams'].keys():
        branchParams.append(key)
        if self.branchCountOnLevel != 1:
          branchChangedParamValue.append(endInfo['branchChangedParams'][key]['actualValue'][self.branchCountOnLevel-2])
          branchChangedParamPb.append(endInfo['branchChangedParams'][key]['associatedProbability'][self.branchCountOnLevel-2])
          condPbC = endInfo['branchChangedParams'][key]['changedConditionalPb'][self.branchCountOnLevel-2]
          subGroup.add('happenedEvent',True)
          hist = endInfo['parentNode'].get('happenedEventVarHistory') + [endInfo['branchDist']]
          subGroup.add('happenedEventVarHistory', hist)
        else:
          subGroup.add('happenedEvent',False)
          subGroup.add('happenedEventVarHistory', endInfo['parentNode'].get('happenedEventVarHistory'))
          branchChangedParamValue.append(endInfo['branchChangedParams'][key]['oldValue'])
          branchChangedParamPb.append(endInfo['branchChangedParams'][key]['unchangedPb'])
          condPbUn =  endInfo['branchChangedParams'][key]['unchangedConditionalPb']

      subGroup.add('branchChangedParam',branchParams)
      # add conditional probability
      if self.branchCountOnLevel != 1:
        subGroup.add('conditionalPb',condPbC)
        subGroup.add('branchChangedParamValue',branchChangedParamValue)
        subGroup.add('branchChangedParamPb',branchChangedParamPb)
      else:
        subGroup.add('conditionalPb',condPbUn)
        subGroup.add('branchChangedParamValue',branchChangedParamValue)
        subGroup.add('branchChangedParamPb',branchChangedParamPb)
      # add initiator distribution info, start time, etc.

      subGroup.add('initiatorDistribution',self.toBeSampled[endInfo['branchDist']])
      subGroup.add('triggeredVariable',endInfo['branchDist'])
      subGroup.add('startTime', endInfo['parentNode'].get('endTime'))
      # initialize the endTime to be equal to the start one... It will modified at the end of this branch
      subGroup.add('endTime', endInfo['parentNode'].get('endTime'))
      # add the branchedLevel dictionary to the subgroup
      if self.branchCountOnLevel != 1:
        branchedLevel[endInfo['branchDist']] = branchedLevel[endInfo['branchDist']] - 1
      # branch calculation info... running, queue, etc are set here
      subGroup.add('runEnded',False)
      subGroup.add('running',False)
      subGroup.add('queue',True)
      #  subGroup.set('restartFileRoot',endInfo['restartRoot'])
      # Append the new branch (subgroup) info to the parentNode in the tree object
      endInfo['parentNode'].appendBranch(subGroup)
      # Fill the values dictionary that will be passed into the model in order to create an input
      # In this dictionary the info for changing the original input is stored
      rlz.inputInfo['prefix'] = rname
      rlz.inputInfo['standardDETvariables'] = self.standardDETvariables
      rlz.inputInfo['endTimeStep'] = endInfo['endTimeStep']
      rlz.inputInfo['branchChangedParam'] = subGroup.get('branchChangedParam')
      rlz.inputInfo['branchChangedParamValue'] = subGroup.get('branchChangedParamValue')
      rlz.inputInfo['conditionalPb'] = subGroup.get('conditionalPb')
      rlz.inputInfo['startTime'] = endInfo['parentNode'].get('endTime')
      rlz.inputInfo['RAVEN_parentID'] = subGroup.get('parent')
      rlz.inputInfo['RAVEN_isEnding'] = True

      #'RAVEN_parentID','RAVEN_isEnding'
      rlz.inputInfo['happenedEvent'] = subGroup.get('happenedEvent')
      rlz.inputInfo['happenedEventVarHistory'] = subGroup.get('happenedEventVarHistory')
      # add additional edits if needed
      model.getAdditionalInputEdits(rlz.inputInfo)
      # add the newer branch name to the map
      self.rootToJob[rname] = self.rootToJob[subGroup.get('parent')]
      # check if it is a preconditioned DET sampling, if so add the relative information
      precSampled = endInfo['parentNode'].get('hybridsamplerCoordinate')
      if precSampled:
        # TODO deepcopy necessary?
        rlz.inputInfo['hybridsamplerCoordinate'] = copy.deepcopy(precSampled)
        subGroup.add('hybridsamplerCoordinate', precSampled)
      # Check if the distribution that just triggered hitted the last probability threshold .
      #  In this case there is not a probability threshold that needs to be added in the input
      #  for this particular distribution
      if not (branchedLevel[endInfo['branchDist']] >= len(self.branchProbabilities[endInfo['branchDist']])):
        rlz.inputInfo['initiatorDistribution'] = [self.toBeSampled[endInfo['branchDist']]]
        rlz.inputInfo['triggeredVariable'] = endInfo['branchDist']
        rlz.inputInfo['PbThreshold'] = [self.branchProbabilities[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]]
        rlz.inputInfo['ValueThreshold'] = [self.branchValues[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]]
      #  For the other distributions, we put the unbranched thresholds
      #  Before adding these thresholds, check if the keyword 'initiatorDistribution' is present...
      #  (In the case the previous if statement is true, this keyword is not present yet
      #  Add it otherwise
      if not ('initiatorDistribution' in rlz.inputInfo):
        rlz.inputInfo['initiatorDistribution'] = []
        rlz.inputInfo['PbThreshold'] = []
        rlz.inputInfo['ValueThreshold'] = []
        rlz.inputInfo['triggeredVariable'] = 'None'
      # Add the unbranched thresholds
      for branch, probs in self.branchProbabilities.items():
        if not (branch in self.toBeSampled[endInfo['branchDist']]) and (branchedLevel[branch] < len(probs)):
          rlz.inputInfo['initiatorDistribution'].append(self.toBeSampled[branch])
        # duplicate?? TODO remove if not (branch in self.toBeSampled[endInfo['branchDist']]) and (branchedLevel[branch] < len(probs)):
          rlz.inputInfo['PbThreshold'].append(probs[branchedLevel[branch]])
          rlz.inputInfo['ValueThreshold'].append(self.branchValues[branch][branchedLevel[branch]])
      rlz.clear(what=['values'])
      self.raiseMarker(f'stdDETvars, pref {rlz.inputInfo["prefix"]}')
      self.raiseWhatsThis('stdDETvars', self.standardDETvariables)
      for varname in self.standardDETvariables:
        rlz[varname] = self.branchValues[varname][branchedLevel[varname]]
        rlz.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[varname][branchedLevel[varname]]
      self._constantVariables(rlz)
      if precSampled:
        for precSample in precSampled:
          # precSample is a realization object, so pull information to rlz from it
          rlz.update(precSample)
          rlz.inputInfo.update(precSample.inputInfo)
      pointPb = reduce(mul,
            [it for sub in
               [pre.inputInfo['SampledVarsPb'].values() for pre in precSampled ]
            for it in sub]
            if precSampled else [1.0])
      rlz.inputInfo['PointProbability'] = pointPb * subGroup.get('conditionalPb')
      rlz.inputInfo['ProbabilityWeight'] = rlz.inputInfo['PointProbability']
      for varName, pb in rlz.inputInfo['SampledVarsPb'].items():
        rlz.inputInfo[f'ProbabilityWeight-{varName.strip()}'] = pb
      # TODO REMOVE rlz.inputInfo.update({f'ProbabilityWeight-{key.strip()}':value for key,value in rlz.inputInfo['SampledVarsPb'].items()})
      ##### REDUNDANT FUNCTIONALS #####
      self._functionalVariables(rlz)
      # Add some useful variable naming in the input
      rlz.inputInfo.update(self.__createVariablesInfoForKwargs(model, rlz))
      # Add the new input path into the RunQueue system
      newInputs = {'args': [str(self.type)], 'rlz': rlz}
      self.RunQueue['queue'].append(newInputs)
      self.RunQueue['identifiers'].append(rlz.inputInfo['prefix'])
      for key,value in rlz.inputInfo.items():
        subGroup.add(key,copy.copy(value))
      popped = endInfo.pop('parentNode')
      subGroup.add('endInfo',copy.deepcopy(endInfo))
      endInfo['parentNode'] = popped
      del branchedLevel

  def _createRunningQueue(self, rlz, model, myInput, forceEvent=False):
    """
      Function to create and append new inputs to the queue. It uses all the containers have been updated by the previous functions
      @ In, model, Model instance, model instance that can be a Code type, ROM, etc.
      @ In, myInput, list, List of the original inputs
      @ In, forceEvent, bool, True if a branching needs to be forced
      @ Out, None
    """
    if self.counters['samples'] >= 1:
      # The first DET calculation branch has already been run
      # Start the manipulation:
      #  Pop out the last endInfo information and the branchedLevel
      self._createRunningQueueBranch(rlz, model, myInput, forceEvent=forceEvent)
    else:
      # We construct the input for the first DET branch calculation'
      self._createRunningQueueBegin(rlz, model, myInput)
    return

  def __createVariablesInfoForKwargs(self, model, rlz):
    """
      This utility method is to create a variable info block
      useful to couple with DET external codes
      @ In, model, Model instance, model instance that can be a Code type, ROM, etc.
      @ In, rlz, Realization, dict-like object to fill with sample
      @ Out, varInfo, dict, the dictionary containing the variable names
             ({'DETVariables':[...], 'HDETVariables':[...],'FunctionVariables':{Var:DepdendentVar},'ConstantVariables':[...]})
    """
    varInfo = {}
    # We collect some useful information for the DET handling (DET variables, contants, functions)
    standardDet = copy.deepcopy(self.standardDETvariables)
    depVars = copy.deepcopy(self.dependentSample)
    consts = copy.deepcopy(self.constants)
    for var in depVars:
      depVars[var] = self.funcDict[var].instance.parameterNames()
    model._replaceVariablesNamesWithAliasSystem(depVars)
    model._replaceVariablesNamesWithAliasSystem(standardDet)
    model._replaceVariablesNamesWithAliasSystem(consts)
    varInfo['DETVariables'] = list(standardDet)
    hvars = {}
    if 'hybridsamplerCoordinate' in rlz.inputInfo:
      for precSample in rlz.inputInfo['hybridsamplerCoordinate']:
        hvars.update(precSample)
        # TODO also need inputInfo?
      model._replaceVariablesNamesWithAliasSystem(hvars)
      varInfo['HDETVariables'] = list(hvars.keys())
    varInfo['FunctionVariables'] = depVars
    varInfo['ConstantVariables'] = list(consts.keys())
    # if len(depVars):
    #  # create graph structure
    #  graphDict = dict.fromkeys(standardDet, [])
    #  graphDict.update(depVars)
    #  varInfo['dependencyGraph'] = graph(graphDict)
    return varInfo

  def __getQueueElement(self):
    """
      Function to get an input from the internal queue system
      @ In, None
      @ Out, jobInput, list, the list of inout (First input in the queue)
    """
    # Pop out the first input in queue
    jobInput = self.RunQueue['queue'].pop(0)
    jobId = self.RunQueue['identifiers'].pop(0)
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

  def generateInput(self, model, modelInput):
    """
      This method has to be overwritten to provide the specialization for the specific sampler
      The model instance in might be needed since, especially for external codes,
      only the code interface possesses the dictionary for reading the variable definition syntax
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, generateInput, (0,list), list containing the new inputs -in reality it is the model that returns this; the Sampler generates the value to be placed in the input of the model.
    """
    rlzBatch = RealizationBatch(self.getBatchSize())
    # for now, we take one realization at a time, until such time as DET is reworked to
    # provide batches of realizations.
    rlz = rlzBatch[0]
    modelInput = self.localGenerateInput(rlz, model, modelInput)
    rlzBatch.ID = rlz.inputInfo['prefix'] # TODO fix when moving to batches
    return rlzBatch, modelInput

  def localGenerateInput(self, rlz, model, modelInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the rlz.inputInfo should be ready to be sent
      to the model
      @ In, rlz, Realization, dict-like object to fill with sample
      @ In, model, model instance, an instance of a model
      @ In, modelInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, newerInput, list, list of new inputs
    """
    if self.counters['samples'] <= 1:
      # If first branch input, create the queue
      self._createRunningQueue(rlz, model, modelInput)
    # retrieve the input from the queue
    newerInput = self.__getQueueElement()
    # If no inputs are present in the queue => a branch is finished
    if not newerInput:
      self.raiseADebug('A Branch ended!')

    ## It turns out the "newerInput" contains all of the information that should
    ## be in inputInfo -- DPM 4/26/17
    newRlz = newerInput['rlz']
    rlz.copyFrom(newRlz)
    return modelInput

  def _generateDistributions(self,availableDist,availableFunc):
    """
      Generates the distrbutions and functions.
      It is overloaded here because we need to manage the sub-sampling
      strategies for the Hybrid DET approach
      @ In, availDist, dict, dict of distributions
      @ In, availableFunc, dict, dict of functions
      @ Out, None
    """
    Grid._generateDistributions(self,availableDist,availableFunc)
    for hybridsampler in self.hybridStrategyToApply.values():
      hybridsampler._generateDistributions(availableDist,availableFunc)

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    #TODO remove using xmlNode
    self._localInputAndChecksDET(xmlNode, paramInput)
    self._localInputAndChecksHybrid(xmlNode, paramInput)

  def _localInputAndChecksDET(self,xmlNode, paramInput):
    """
      Class specific inputs will be read here and checked for validity.
      This method reads the standard DET portion only (no hybrid)
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    Grid.localInputAndChecks(self,xmlNode, paramInput)
    self.printEndXmlSummary = utils.stringIsTrue(xmlNode.attrib.get('printEndXmlSummary', None))
    self.removeXmlBranchInfo = utils.stringIsTrue(xmlNode.attrib.get('removeXmlBranchInfo', None))
    if 'maxSimulationTime'   in xmlNode.attrib.keys():
      try:
        self.maxSimulTime = float(xmlNode.attrib['maxSimulationTime'])
      except (KeyError,NameError):
        self.raiseAnError(IOError,'Can not convert maxSimulationTime in float number!!!')
    branchedLevel = {}
    gridInfo   = self.gridEntity.returnParameter("gridInfo")
    errorFound = False
    errorMsgs  = ''

    for keyk in self.axisName:
      branchedLevel[keyk] = 0
      #branchedLevel[self.toBeSampled[keyk]] = 0
      self.standardDETvariables.append(keyk)
      if self.gridInfo[keyk] == 'CDF':
        self.branchProbabilities[keyk] = gridInfo[keyk][2]
        self.branchProbabilities[keyk].sort()
        if max(self.branchProbabilities[keyk]) > 1:
          errorMsgs += "One of the Thresholds for distribution " + str(gridInfo[keyk][2]) + " is > 1 \n"
          errorFound = True
        probMultiplicities = Counter(self.branchProbabilities[keyk])
        multiples = [prob for prob,mult in probMultiplicities.items() if mult > 1]
        ## Only the multiple variables remain
        for prob in multiples:
          errorMsgs += "In variable " + str(keyk) + " the Threshold " + str(prob)+" appears multiple times!!\n"
          errorFound = True
      else:
        self.branchValues[keyk] = gridInfo[keyk][2]
        self.branchValues[keyk].sort()
        valueMultiplicities = Counter(self.branchValues[keyk])
        multiples = [value for value,mult in valueMultiplicities.items() if mult > 1]
        ## Only the multiple variables remain
        for value in multiples:
          errorMsgs += "In variable " + str(keyk) + " the Threshold " + str(value)+" appears multiple times!!\n"
          errorFound = True
    # check if RELAP7 mode is activated, in case check that a <distribution> variable is unique in the input
    if any("<distribution>" in s for s in self.branchProbabilities.keys()):
      associatedDists = self.toBeSampled.values()
      if len(list(set(associatedDists))) != len(associatedDists):
        errorMsgs += "Distribution-mode sampling activated in " + self.name+". In this case every <distribution> needs to be assocaited with one single <Distribution> block!\n"
        errorFound = True
    if errorFound:
      self.raiseAnError(IOError,"In sampler named " + self.name+' the following errors have been found: \n'+errorMsgs )
    # Append the branchedLevel dictionary in the proper list
    self.branchedLevel.append(branchedLevel)

  def _localInputAndChecksHybrid(self,xmlNode, paramInput):
    """
      Class specific inputs will be read here and checked for validity.
      This method reads the hybrid det portion only
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'HybridSampler':
        if not 'type' in child.attrib.keys():
          self.raiseAnError(IOError,'Not found attribute type in hybridsamplerSampler block!')
        if child.attrib['type'] in self.hybridStrategyToApply:
          self.raiseAnError(IOError,'Hybrid Sampler type '+child.attrib['type'] + ' already inputted!')
        if child.attrib['type'] not in self.hybridSamplersAvail:
          self.raiseAnError(IOError,'Hybrid Sampler type ' +child.attrib['type'] + ' unknown. Available are '+ ','.join(self.hybridSamplersAvail.keys()) + '!')
        self.hybridNumberSamplers = 1
        # the user can decided how to sample the epistemic
        self.hybridStrategyToApply[child.attrib['type']] = self.hybridSamplersAvail[child.attrib['type']]()
        # make the hybridsampler sampler read  its own xml block
        childCopy = copy.deepcopy(child)
        childCopy.tag = child.attrib['type']
        childCopy.attrib['name']='none'
        childCopy.attrib.pop('type')
        self.hybridStrategyToApply[child.attrib['type']]._readMoreXML(childCopy)
        # store the variables that represent the epistemic space
        self.epistemicVariables.update(dict.fromkeys(self.hybridStrategyToApply[child.attrib['type']].toBeSampled.keys(),{}))
        for epVar in self.epistemicVariables:
          if len(epVar.split(",")) > 1:
            for el in epVar.split(","):
              self.fullyCorrelatedEpistemicToVar[el.strip()] = epVar

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for branch, probs in self.branchProbabilities.items():
      paramDict[f'Probability Thresholds for var {branch} are: '] = [str(x) for x in probs]
    for branch, vals in self.branchValues.items()       :
      paramDict[f'Values Thresholds for var {branch} are: '] = [str(x) for x in vals]
    return paramDict

  def localGetCurrentSetting(self):
    """
      Appends a given dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['actual threshold levels are '] = self.branchedLevel[0]
    return paramDict

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    # if using the "hybrid" approach, then the aleatoric uncertainty is handled by DET,
    # but the epistemic uncertainty is handled by other samplers (called "hybrid" samplers)
    # Thus, the hybrid samples serve as starting branch points for the DET sampling, I think.
    # See 2015 PSA ALFONSI et al. - talbpw 2025
    if self.hybridStrategyToApply:
      hybridRlzs = []
      for cnt, samplerName  in enumerate(self.hybridStrategyToApply):
        hybridSampler = self.hybridStrategyToApply[samplerName]
        hybridRlzs.append([])
        hybridSampler.initialize()
        self.hybridNumberSamplers *= hybridSampler.limits['samples']
        while hybridSampler.amIreadyToProvideAnInput():
          # TODO make this a full batch rather than one-at-a-time?
          batch = RealizationBatch(1)
          rlz = batch[0]
          hybridSampler.counters['samples'] +=1
          if hybridSampler.batch > 0:
            hybridSampler.localGenerateInput(batch, None, None)
          else:
            hybridSampler.localGenerateInput(rlz, None, None)
          hybridSampler._constantVariables(batch)
          hybridSampler._functionalVariables(batch)
          # TODO is this redundant with calling _constantVariables?
          self.hybridConstants.update(hybridSampler.constants)
          self.hybridDependentSample.update(hybridSampler.dependentSample)
          # TODO adding this crashes the run ... why? Who knows.
          # rlz.inputInfo['prefix'] = hybridSampler.counters['samples']
          # only store the Realization (not the Batch), since we're going to recombine these later.
          hybridRlzs[cnt].append(rlz)
    # samples covering epistemic space are then the full combination of hybrid samples
    if self.hybridNumberSamplers > 0:
      self.raiseAMessage('Number of Hybrid Samples are ' + str(self.hybridNumberSamplers) + '!')
      hybridNumber = self.hybridNumberSamplers
      combinations = list(itertools.product(*hybridRlzs))
    else:
      hybridNumber = 1
    # build tree mapping to store each branch (I think?)
    self.TreeInfo = {}
    for precSample in range(hybridNumber):
      branchName = f'{self.name}_{precSample+1}'
      elm = ETS.HierarchicalNode(branchName)
      elm.add('name', branchName)
      self.raiseMarker(f'branch "{branchName}"')
      elm.add('startTime', str(0.0))
      # Initialize the endTime to be equal to the start one...
      # It will modified at the end of each branch
      elm.add('endTime', str(0.0))
      elm.add('runEnded', False)
      elm.add('running', True)
      elm.add('queue', False)
      # if preconditioned DET, add the sampled from hybridsampler samplers
      if self.hybridNumberSamplers > 0: # isn't this always true? We are already looping over hybridNumber
        elm.add('hybridsamplerCoordinate', combinations[precSample])
        for rlz in combinations[precSample]:
          for epistVar, val in rlz.items():
            # skip constants or dependent variables, as these will be added separately
            if epistVar in hybridSampler.constants or epistVar in hybridSampler.dependentSample:
              continue
            # correlated variables need slight special handling
            if epistVar in self.fullyCorrelatedEpistemicToVar:
              self.epistemicVariables[self.fullyCorrelatedEpistemicToVar[epistVar]][elm.get('name')] = val
            # add realization values to the epistemic variables; epistemicVariables become effectively
            #   rebuilt samples combined from the hybrid samples
            # TODO should they be realizations?
            else:
              self.epistemicVariables[epistVar][elm.get('name')] = val
      # The dictionary branchedLevel is stored in the xml tree too. That's because
      # the advancement of the thresholds must follow the tree structure
      elm.add('branchedLevel', self.branchedLevel[0])
      # Here it is stored all the info regarding the DET => we create the info for all the
      # branchings and we store them
      self.TreeInfo[branchName] = ETS.HierarchicalTree(elm)

    initBranchProbabilities = copy.copy(self.branchProbabilities)
    initBranchValues = copy.copy(self.branchValues)
    for name, probs in self.branchProbabilities.items():
      if ("<distribution>" in name) or (self.variables2distributionsMapping[name]['totDim']==1):
        # 1Dimensional Distributions (inverse CDF)
        initBranchValues[name] = [self.distDict[name].ppf(float(probs[index])) for index in range(len(probs))]
      else:
        # NDimensional Distrubutions (inverse Marginal CDF)
        initBranchValues[name] = [
            self.distDict[name].inverseMarginalDistribution(
                float(probs[index]),
                self.variables2distributionsMapping[name]['dim']-1)
            for index in range(len(probs))
        ]

    for name, value in self.branchValues.items():
      if ("<distribution>" in name) or (self.variables2distributionsMapping[name]['totDim']==1):
        # 1Dimensional Distributions (CDF)
        initBranchProbabilities[name] = [self.distDict[name].cdf(float(value[index])) for index in range(len(value))]
      else:
        # NDimensional Distrubutions (Marginal CDF)
        initBranchProbabilities[name] = [
            self.distDict[name].marginalDistribution(
                float(value[index]),
                self.variables2distributionsMapping[name]['dim']-1)
            for index in range(len(value))
        ]

    self.branchValues.update(initBranchValues)
    self.branchProbabilities.update(initBranchProbabilities)
    for name, value in self.branchValues.items():
      # add the last forced branch (CDF=1)
      if 1.0 not in self.branchProbabilities[name]:
        self.branchProbabilities[name] = np.append(self.branchProbabilities[name],[1.0])
        #self.branchProbabilities[key].append( 1.0 )
        if ("<distribution>" in name) or (self.variables2distributionsMapping[name]['totDim']==1):
          self.branchValues[name] = np.append(value,[self.distDict[name].ppf(1.0)])
          #self.branchValues[key].append( self.distDict[key].ppf(1.0) )
        else:
          self.branchValues[name] = np.append(
              value,
              self.distDict[name].inverseMarginalDistribution(
                  1.0, self.variables2distributionsMapping[name]['dim']-1)
          )
          #self.branchValues[key].append(self.distDict[key].inverseMarginalDistribution(1.0,self.variables2distributionsMapping[key]['dim']-1) )
    self.limits['samples'] = sys.maxsize # ??? why?
    # add expected metadata
    self.addMetaKeys(['RAVEN_parentID','RAVEN_isEnding','conditionalPb','triggeredVariable','happenedEvent'])
