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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
from operator import mul
from functools import reduce
import xml.etree.ElementTree as ET
import itertools
from collections import Counter
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Grid import Grid
from .MonteCarlo import MonteCarlo
from .Stratified import Stratified
from .Sampler import Sampler
from utils import utils
from utils import InputData
import utils.TreeStructure as ETS
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

    inputSpecification.addParam("printEndXmlSummary", InputData.StringType)
    inputSpecification.addParam("maxSimulationType", InputData.FloatType)
    inputSpecification.addParam("removeXmlBranchInfo", InputData.StringType)

    oldSub = inputSpecification.popSub("Distribution")
    newDistributionInput = InputData.parameterInputFactory("Distribution", baseNode=oldSub)
    gridInput = InputData.parameterInputFactory("grid", contentType=InputData.StringType)
    gridInput.addParam("type", InputData.StringType)
    gridInput.addParam("construction", InputData.StringType)
    gridInput.addParam("steps", InputData.IntegerType)

    newDistributionInput.addSub(gridInput)
    inputSpecification.addSub(newDistributionInput)

    #Strict mode off because basically this allows things to be passed to
    # sub Samplers, which will be checked later.
    hybridSamplerInput = InputData.parameterInputFactory("HybridSampler", strictMode=False)
    hybridSamplerInput.addParam("type", InputData.StringType)

    for nodeName in ['variable','Distribution']:
      nodeInput = InputData.parameterInputFactory(nodeName, strictMode=False)
      nodeInput.addParam("name", InputData.StringType)
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
    # Working directory (Path of the directory in which all the outputs,etc. are stored)
    self.workingDir                        = ""
    # (optional) if not present, the sampler will not change the relative keyword in the input file
    self.maxSimulTime                      = None
    # print the xml tree representation of the dynamic event tree calculation
    # see variable 'self.TreeInfo'
    self.printEndXmlSummary                = False
    # flag to control if the branch info xml file needs to be removed after reading
    self.removeXmlBranchInfo               = True
    # Dictionary of the probability bins for each distribution that have been
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    self.branchProbabilities               = {}
    # Dictionary of the Values' bins for each distribution that have been
    #  inputted by the user ('distName':[Pb_Threshold_1, Pb_Threshold_2, ..., Pb_Threshold_n])
    # these are the invCDFs of the PBs inputted in branchProbabilities (if ProbabilityThresholds have been inputted)
    self.branchValues                      = {}
    # List of Dictionaries of the last probability bin level (position in the array) reached for each distribution ('distName':IntegerValue)
    # This container is a working dictionary. The branchedLevels are stored in the xml tree "self.TreeInfo" since they must track
    # the evolution of the dynamic event tree
    self.branchedLevel                     = []
    # Counter for the branch needs to be run after a calculation branched (it is a working variable)
    self.branchCountOnLevel                = 0
    # Dictionary tha contains the actual branching info
    # (i.e. distribution that triggered, values of the variables that need to be changed, etc)
    self.actualBranchInfo                  = {}
    # Parent Branch end time (It's a working variable used to set up the new branches need to be run.
    #   The new branches' start time will be the end time of the parent branch )
    self.actualEndTime                     = 0.0
    # Parent Branch end time step (It's a working variable used to set up the new branches need to be run.
    #  The end time step is used to construct the filename of the restart files needed for restart the new branch calculations)
    self.actualEndTs                       = 0
    # Xml tree object. It stored all the info regarding the DET. It is in continue evolution during a DET calculation
    self.TreeInfo                          = None
    # List of Dictionaries. It is a working variable used to store the information needed to create branches from a Parent Branch
    self.endInfo                           = []
    # Queue system. The inputs are waiting to be run are stored in this queue dictionary
    self.RunQueue                          = {}
    # identifiers of the inputs in queue (name of the history... for example DET_1,1,1)
    self.RunQueue['identifiers']           = []
    # Corresponding inputs
    self.RunQueue['queue']                 = []
    # mapping from jobID to rootname in TreeInfo {jobID:rootName}
    self.rootToJob                         = {}
    # dictionary of Hybrid Samplers available
    self.hybridSamplersAvail               = {'MonteCarlo':MonteCarlo,'Stratified':Stratified,'Grid':Grid}
    # dictionary of inputted hybridsamplers need to be applied
    self.hybridStrategyToApply             = {}
    # total number of hybridsampler samples (combination of all different hybridsampler strategy)
    self.hybridNumberSamplers              = 0
    # List of variables that represent the aleatory space
    self.standardDETvariables              = []
    # Dictionary of variables that represent the epistemic space (hybrid det). Format => {'epistemicVarName':{'HybridTree name':value}}
    self.epistemicVariables                = {}

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
    self._endJobRunnable = max((len(self.RunQueue['queue']),1))
    if(len(self.RunQueue['queue']) != 0 or self.counter == 0):
      ready = True
    else:
      if self.printEndXmlSummary:
        myFile = open(os.path.join(self.workingDir,self.name + "_outputSummary.xml"),'w')
        for treeNode in self.TreeInfo.values():
          treeNode.writeNodeTree(myFile)
        myFile.close()
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
    self.workingDir = model.workingDir
    # returnBranchInfo = self.__readBranchInfo(jobObject.output)
    # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    parentNode = self._retrieveParentNode(jobObject.identifier)
    # set runEnded and running to true and false respectively
    parentNode.add('runEnded',True)
    parentNode.add('running',False)
    parentNode.add('endTime',self.actualEndTime)
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
    kwargs = jobObject.args[3]
    stepWorkingDir = kwargs['WORKING_DIR']
    jobWorkingDir = os.path.join(stepWorkingDir,kwargs['prefix'] if 'prefix' in kwargs.keys() else '1')

    ## This appears to be the same, so I am switching to the model's workingDir
    ## since it is more directly available and less change to how data is stored
    ## in the args of a job instance. -- DPM 4/12/17
    # jobWorkingDir = self.workingDir

    if not self.__readBranchInfo(jobObject.getMetadata()['outfile'], jobWorkingDir):
      parentNode.add('completedHistory', True)
      return False
    # Collect the branch info in a multi-level dictionary
    endInfo = {'endTime':self.actualEndTime,'endTimeStep':self.actualEndTs,'branchDist':list(self.actualBranchInfo.keys())[0]}
    endInfo['branchChangedParams'] = self.actualBranchInfo[endInfo['branchDist']]
    # check if RELAP7 mode is activated, in case prepend the "<distribution>" string
    if any("<distribution>" in s for s in self.branchProbabilities.keys()):
      endInfo['branchDist'] = list(self.toBeSampled.keys())[list(self.toBeSampled.values()).index(endInfo['branchDist'])]
      #endInfo['branchDist'] = "<distribution>"+endInfo['branchDist']
    parentNode.add('actualEndTimeStep',self.actualEndTs)
    # # Get the parent element tree (xml object) to retrieve the information needed to create the new inputs
    # if(jobObject.identifier == self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().name): endInfo['parentNode'] = self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode()
    # else: endInfo['parentNode'] = list(self.TreeInfo[self.rootToJob[jobObject.identifier]].getrootnode().iter(jobObject.identifier))[0]
    endInfo['parentNode'] = parentNode
    # get the branchedLevel dictionary
    branchedLevel = {}
    for distk, distpb in zip(endInfo['parentNode'].get('SampledVarsPb').keys(),endInfo['parentNode'].get('SampledVarsPb').values()):
      if distk not in self.epistemicVariables.keys():
        branchedLevel[distk] = utils.index(self.branchProbabilities[distk],distpb)
    if not branchedLevel:
      self.raiseAnError(RuntimeError,'branchedLevel of node '+jobObject.identifier+'not found!')
    # Loop of the parameters that have been changed after a trigger gets activated
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

    self.branchCountOnLevel = 0
    # # set runEnded and running to true and false respectively
    # endInfo['parentNode'].add('runEnded',True)
    # endInfo['parentNode'].add('running',False)
    # endInfo['parentNode'].add('endTime',self.actualEndTime)
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
      self._createRunningQueue(model,myInput)
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
      parentCondPb = self.endInfo[index]['parentNode'].get('conditionalPbr')
      if not parentCondPb:
        parentCondPb = 1.0
    except KeyError:
      parentCondPb = 1.0
    # for all the branches the conditional pb is computed
    # unchangedConditionalPb = Conditional Probability of the branches in which the event has not occurred
    # changedConditionalPb   = Conditional Probability of the branches in which the event has occurred
    for key in self.endInfo[index]['branchChangedParams']:
      self.endInfo[index]['branchChangedParams'][key]['changedConditionalPb'] = []
      self.endInfo[index]['branchChangedParams'][key]['unchangedConditionalPb'] = parentCondPb*float(self.endInfo[index]['branchChangedParams'][key]['unchangedPb'])
      for pb in range(len(self.endInfo[index]['branchChangedParams'][key]['associatedProbability'])):
        self.endInfo[index]['branchChangedParams'][key]['changedConditionalPb'].append(parentCondPb*float(self.endInfo[index]['branchChangedParams'][key]['associatedProbability'][pb]))

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
      if node.tag == "Distribution_trigger":
        distName = node.attrib['name'].strip()
        self.actualBranchInfo[distName] = {}
        for child in node:
          self.actualBranchInfo[distName][child.text.strip()] = {'varType':child.attrib['type'].strip(),'actualValue':child.attrib['actual_value'].strip().split(),'oldValue':child.attrib['old_value'].strip()}
          if 'probability' in child.attrib:
            asPb = child.attrib['probability'].strip().split()
            self.actualBranchInfo[distName][child.text.strip()]['associatedProbability'] = []
            #self.actualBranchInfo[distName][child.text.strip()]['associatedProbability'].append(float(asPb))
            for index in range(len(asPb)):
              self.actualBranchInfo[distName][child.text.strip()]['associatedProbability'].append(float(asPb[index]))
      # we exit the loop here, because only one trigger at the time can be handled  right now
      break
    # remove the file
    if self.removeXmlBranchInfo:
      os.remove(filename)
    branchPresent = True
    return branchPresent

  def _createRunningQueueBeginOne(self,rootTree,branchedLevel, model,myInput):
    """
      Method to generate the running internal queue for one point in the epistemic
      space. It generates the initial information to instantiate the root of a
      Deterministic Dynamic Event Tree.
      @ In, rootTree, Node object, the rootTree of the single coordinate in the epistemic space.
      @ In, branchedLevel, dict, dictionary of the levels reached by the rootTree mapped in the internal grid dictionary (self.branchProbabilities)
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ Out, None
    """
    # add additional edits if needed
    model.getAdditionalInputEdits(self.inputInfo)

    precSampled = rootTree.getrootnode().get('hybridsamplerCoordinate')
    rootnode    =  rootTree.getrootnode()
    rname       = rootnode.name
    rootnode.add('completedHistory', False)
    # Fill th values dictionary in
    if precSampled:
      self.inputInfo['hybridsamplerCoordinate'  ] = copy.deepcopy(precSampled)
    self.inputInfo['prefix'                    ] = rname
    self.inputInfo['initiatorDistribution'     ] = []
    self.inputInfo['PbThreshold'               ] = []
    self.inputInfo['ValueThreshold'            ] = []
    self.inputInfo['branchChangedParam'        ] = [b'None']
    self.inputInfo['branchChangedParamValue'   ] = [b'None']
    self.inputInfo['startTime'                 ] = -sys.float_info.max
    self.inputInfo['endTimeStep'               ] = 0
    self.inputInfo['RAVEN_parentID'            ] = "None"
    self.inputInfo['RAVEN_isEnding'            ] = True
    self.inputInfo['conditionalPb'             ] = [1.0]
    self.inputInfo['conditionalPbr'            ] = 1.0
    self.inputInfo['happenedEvent'             ] = False
    for key in self.branchProbabilities.keys():
      self.inputInfo['initiatorDistribution'].append(self.toBeSampled[key])
    #for key in self.branchProbabilities.keys():self.inputInfo['initiatorDistribution'].append(key.encode())
    for key in self.branchProbabilities.keys():
      self.inputInfo['PbThreshold'].append(self.branchProbabilities[key][branchedLevel[key]])
    #for key in self.branchProbabilities.keys():self.inputInfo['PbThreshold'].append(self.branchProbabilities[key][branchedLevel[key]])
    for key in self.branchProbabilities.keys():
      self.inputInfo['ValueThreshold'].append(self.branchValues[key][branchedLevel[key]])
    #for key in self.branchProbabilities.keys():self.inputInfo['ValueThreshold'].append(self.branchValues[key][branchedLevel[key]])
    for varname in self.standardDETvariables:
      self.inputInfo['SampledVars'  ][varname] = self.branchValues[varname][branchedLevel[varname]]
      #self.inputInfo['SampledVars'  ][varname] = self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
      self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[varname][branchedLevel[varname] ]
      #self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
    # constant variables
    self._constantVariables()

    if precSampled:
      for precSample in precSampled:
        self.inputInfo['SampledVars'  ].update(precSample['SampledVars'])
        self.inputInfo['SampledVarsPb'].update(precSample['SampledVarsPb'])
    self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
    self.inputInfo.update({'ProbabilityWeight-'+key.strip():value for key,value in self.inputInfo['SampledVarsPb'].items()})

    if(self.maxSimulTime):
      self.inputInfo['endTime'] = self.maxSimulTime
    # Add the new input path into the RunQueue system
    newInputs = {'args':[str(self.type)], 'kwargs':dict(self.inputInfo)}
    for key,value in self.inputInfo.items():
      rootnode.add(key,copy.copy(value))
    self.RunQueue['queue'].append(newInputs)
    print(self.inputInfo['prefix'])
    self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
    self.rootToJob[self.inputInfo['prefix']] = rname
    del newInputs
    self.counter += 1

  def _createRunningQueueBegin(self,model,myInput):
    """
      Method to generate the running internal queue for all the points in
      the epistemic space. It generates the initial information to
      instantiate the roots of all the N-D coordinates to construct multiple
      Deterministic Dynamic Event Trees.
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
      self._createRunningQueueBeginOne(rootTree,branchedLevel, model,myInput)

  def _createRunningQueueBranch(self,model,myInput,forceEvent=False):
    """
      Method to generate the running internal queue right after a branch occurred
      It generates the the information to insatiate the branches' continuation of the Deterministic Dynamic Event Tree
      @ In, model, Models object, the model that is used to explore the input space (e.g. a code, like RELAP-7)
      @ In, myInput, list, list of inputs for the Models object (passed through the Steps XML block)
      @ In, forceEvent, bool, if True the events are forced to happen (basically, the "unchanged event" is not created at all)
      @ Out, None
    """
    # The first DET calculation branch has already been run'
    # Start the manipulation:

    #  Pop out the last endInfo information and the branchedLevel
    branchedLevelParent     = self.branchedLevel.pop(0)
    endInfo                 = self.endInfo.pop(0)
    self.branchCountOnLevel = 0 #?
    # n_branches = number of branches need to be run
    nBranches = endInfo['n_branches']
    # Check if the distribution that just triggered hitted the last probability threshold .
    # In case we create a number of branches = endInfo['n_branches'] - 1 => the branch in
    # which the event did not occur is not going to be tracked
    if branchedLevelParent[endInfo['branchDist']] >= len(self.branchProbabilities[endInfo['branchDist']]):
      self.raiseADebug('Branch ' + endInfo['parentNode'].get('name') + ' hit last Threshold for distribution ' + endInfo['branchDist'])
      self.raiseADebug('Branch ' + endInfo['parentNode'].get('name') + ' is dead end.')
      self.branchCountOnLevel = 1
      nBranches -= 1
    else:
      if forceEvent == True:
        self.branchCountOnLevel = 1
        nBranches -= 1
    # Loop over the branches for which the inputs must be created
    for _ in range(nBranches):
      del self.inputInfo
      self.counter += 1
      self.branchCountOnLevel += 1
      branchedLevel = copy.deepcopy(branchedLevelParent)
      # Get Parent node name => the branch name is creating appending to this name  a comma and self.branchCountOnLevel counter
      rname = endInfo['parentNode'].get('name') + '-' + str(self.branchCountOnLevel)
      # create a subgroup that will be appended to the parent element in the xml tree structure
      subGroup = ETS.HierarchicalNode(self.messageHandler,rname)
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
        #subGroup.add('branchChangedParam',key)
        branchParams.append(key)
        if self.branchCountOnLevel != 1:
          branchChangedParamValue.append(endInfo['branchChangedParams'][key]['actualValue'][self.branchCountOnLevel-2])
          branchChangedParamPb.append(endInfo['branchChangedParams'][key]['associatedProbability'][self.branchCountOnLevel-2])
          #subGroup.add('branchChangedParamValue',endInfo['branchChangedParams'][key]['actualValue'][self.branchCountOnLevel-2])
          #subGroup.add('branchChangedParamPb',endInfo['branchChangedParams'][key]['associatedProbability'][self.branchCountOnLevel-2])
          #condPbC.append(endInfo['branchChangedParams'][key]['changedConditionalPb'][self.branchCountOnLevel-2])
          condPbC = condPbC + endInfo['branchChangedParams'][key]['changedConditionalPb'][self.branchCountOnLevel-2]
          subGroup.add('happenedEvent',True)
        else:
          subGroup.add('happenedEvent',endInfo['parentNode'].get('happenedEvent'))
          branchChangedParamValue.append(endInfo['branchChangedParams'][key]['oldValue'])
          branchChangedParamPb.append(endInfo['branchChangedParams'][key]['unchangedPb'])
          #subGroup.add('branchChangedParamValue',endInfo['branchChangedParams'][key]['oldValue'])
          #subGroup.add('branchChangedParamPb',endInfo['branchChangedParams'][key]['unchangedPb'])
          #condPbUn.append(endInfo['branchChangedParams'][key]['unchangedConditionalPb'])
          condPbUn =  condPbUn + endInfo['branchChangedParams'][key]['unchangedConditionalPb']
      subGroup.add('branchChangedParam',branchParams)
      # add conditional probability
      if self.branchCountOnLevel != 1:
        subGroup.add('conditionalPbr',condPbC)
        subGroup.add('branchChangedParamValue',branchChangedParamValue)
        subGroup.add('branchChangedParamPb',branchChangedParamPb)
      else:
        subGroup.add('conditionalPbr',condPbUn)
        subGroup.add('branchChangedParamValue',branchChangedParamValue)
        subGroup.add('branchChangedParamPb',branchChangedParamPb)
      # add initiator distribution info, start time, etc.

      subGroup.add('initiatorDistribution',self.toBeSampled[endInfo['branchDist']])
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
      self.inputInfo = {'prefix':rname,'endTimeStep':endInfo['endTimeStep'],
                'branchChangedParam':subGroup.get('branchChangedParam'),
                'branchChangedParamValue':subGroup.get('branchChangedParamValue'),
                'conditionalPb':subGroup.get('conditionalPbr'),
                'startTime':endInfo['parentNode'].get('endTime'),
                'RAVEN_parentID':subGroup.get('parent'),
                'RAVEN_isEnding':True}

      #'RAVEN_parentID','RAVEN_isEnding'
      self.inputInfo['happenedEvent'] = subGroup.get('happenedEvent')
      # add additional edits if needed
      model.getAdditionalInputEdits(self.inputInfo)
      # add the newer branch name to the map
      self.rootToJob[rname] = self.rootToJob[subGroup.get('parent')]
      # check if it is a preconditioned DET sampling, if so add the relative information
      precSampled = endInfo['parentNode'].get('hybridsamplerCoordinate')
      if precSampled:
        self.inputInfo['hybridsamplerCoordinate'] = copy.deepcopy(precSampled)
        subGroup.add('hybridsamplerCoordinate', precSampled)
      # Check if the distribution that just triggered hitted the last probability threshold .
      #  In this case there is not a probability threshold that needs to be added in the input
      #  for this particular distribution
      if not (branchedLevel[endInfo['branchDist']] >= len(self.branchProbabilities[endInfo['branchDist']])):
        self.inputInfo['initiatorDistribution'] = [self.toBeSampled[endInfo['branchDist']]]
        self.inputInfo['PbThreshold'           ] = [self.branchProbabilities[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]]
        self.inputInfo['ValueThreshold'        ] = [self.branchValues[endInfo['branchDist']][branchedLevel[endInfo['branchDist']]]]
      #  For the other distributions, we put the unbranched thresholds
      #  Before adding these thresholds, check if the keyword 'initiatorDistribution' is present...
      #  (In the case the previous if statement is true, this keyword is not present yet
      #  Add it otherwise
      if not ('initiatorDistribution' in self.inputInfo.keys()):
        self.inputInfo['initiatorDistribution' ] = []
        self.inputInfo['PbThreshold'           ] = []
        self.inputInfo['ValueThreshold'        ] = []
      # Add the unbranched thresholds
      for key in self.branchProbabilities.keys():
        if not (key in self.toBeSampled[endInfo['branchDist']]) and (branchedLevel[key] < len(self.branchProbabilities[key])):
          self.inputInfo['initiatorDistribution'].append(self.toBeSampled[key])
      for key in self.branchProbabilities.keys():
        if not (key in self.toBeSampled[endInfo['branchDist']]) and (branchedLevel[key] < len(self.branchProbabilities[key])):
          self.inputInfo['PbThreshold'   ].append(self.branchProbabilities[key][branchedLevel[key]])
          self.inputInfo['ValueThreshold'].append(self.branchValues[key][branchedLevel[key]])
      self.inputInfo['SampledVars']   = {}
      self.inputInfo['SampledVarsPb'] = {}
      for varname in self.standardDETvariables:
        self.inputInfo['SampledVars'][varname]   = self.branchValues[varname][branchedLevel[varname]]
        self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[varname][branchedLevel[varname]]
        #self.inputInfo['SampledVars'][varname]   = self.branchValues[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
        #self.inputInfo['SampledVarsPb'][varname] = self.branchProbabilities[self.toBeSampled[varname]][branchedLevel[self.toBeSampled[varname]]]
      self._constantVariables()
      if precSampled:
        for precSample in precSampled:
          self.inputInfo['SampledVars'  ].update(precSample['SampledVars'])
          self.inputInfo['SampledVarsPb'].update(precSample['SampledVarsPb'])
      self.inputInfo['PointProbability' ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())*subGroup.get('conditionalPbr')
      self.inputInfo['ProbabilityWeight'] = self.inputInfo['PointProbability' ]
      self.inputInfo.update({'ProbabilityWeight-'+key.strip():value for key,value in self.inputInfo['SampledVarsPb'].items()})
      # Add the new input path into the RunQueue system
      newInputs = {'args': [str(self.type)], 'kwargs':dict(self.inputInfo)}
      self.RunQueue['queue'].append(newInputs)
      self.RunQueue['identifiers'].append(self.inputInfo['prefix'])
      for key,value in self.inputInfo.items():
        subGroup.add(key,copy.copy(value))
      popped = endInfo.pop('parentNode')
      subGroup.add('endInfo',copy.deepcopy(endInfo))
      endInfo['parentNode'] = popped
      del branchedLevel

  def _createRunningQueue(self, model, myInput, forceEvent=False):
    """
      Function to create and append new inputs to the queue. It uses all the containers have been updated by the previous functions
      @ In, model, Model instance, model instance that can be a Code type, ROM, etc.
      @ In, myInput, list, List of the original inputs
      @ In, forceEvent, bool, True if a branching needs to be forced
      @ Out, None
    """
    if self.counter >= 1:
      # The first DET calculation branch has already been run
      # Start the manipulation:
      #  Pop out the last endInfo information and the branchedLevel
      self._createRunningQueueBranch(model, myInput, forceEvent)
    else:
      # We construct the input for the first DET branch calculation'
      self._createRunningQueueBegin(model, myInput)
    return

  def __getQueueElement(self):
    """
      Function to get an input from the internal queue system
      @ In, None
      @ Out, jobInput, list, the list of inout (First input in the queue)
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
      This method has to be overwritten to provide the specialization for the specific sampler
      The model instance in might be needed since, especially for external codes,
      only the code interface possesses the dictionary for reading the variable definition syntax
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, generateInput, (0,list), list containing the new inputs -in reality it is the model that returns this; the Sampler generates the value to be placed in the input of the model.
    """
    #NB: if someday the DET handles restarts as other samplers do in generateInput, the return code 1 indicates the result
    #  is stored in a restart data object, while 0 indicates a new run has been found.
    #model.getAdditionalInputEdits(self.inputInfo)
    return 0, self.localGenerateInput(model, oldInput)

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, newerInput, list, list of new inputs
    """
    #self._endJobRunnable = max([len(self.RunQueue['queue']),1])
    if self.counter <= 1:
      # If first branch input, create the queue
      self._createRunningQueue(model, myInput)
    # retrieve the input from the queue
    newerInput = self.__getQueueElement()
    # If no inputs are present in the queue => a branch is finished
    if not newerInput:
      self.raiseADebug('A Branch ended!')

    ## It turns out the "newerInput" contains all of the information that should
    ## be in inputInfo (which should actually be returned and not stored in the
    ## sampler object, but all samplers do this for now) -- DPM 4/26/17
    self.inputInfo = newerInput['kwargs']
    return myInput

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
    if 'printEndXmlSummary'  in xmlNode.attrib.keys():
      self.printEndXmlSummary  = xmlNode.attrib['printEndXmlSummary'].lower()  in utils.stringsThatMeanTrue()
    if 'removeXmlBranchInfo' in xmlNode.attrib.keys():
      self.removeXmlBranchInfo = xmlNode.attrib['removeXmlBranchInfo'].lower() in utils.stringsThatMeanTrue()
    if 'maxSimulationTime'   in xmlNode.attrib.keys():
      try:
        self.maxSimulTime = float(xmlNode.attrib['maxSimulationTime'])
      except (KeyError,NameError):
        self.raiseAnError(IOError,'Can not convert maxSimulationTime in float number!!!')
    branchedLevel, error_found = {}, False
    gridInfo   = self.gridEntity.returnParameter("gridInfo")
    errorFound = False
    errorMsgs  = ''

    for keyk in self.axisName:
      branchedLevel[keyk] = 0
      #branchedLevel[self.toBeSampled[keyk]] = 0
      self.standardDETvariables.append(keyk)
      if self.gridInfo[keyk] == 'CDF':
        self.branchProbabilities[keyk] = gridInfo[keyk][2]
        self.branchProbabilities[keyk].sort(key=float)
        if max(self.branchProbabilities[keyk]) > 1:
          errorMsgs += "One of the Thresholds for distribution " + str(gridInfo[keyk][2]) + " is > 1 \n"
          errorFound = True
        probMultiplicities = Counter(self.branchProbabilities[keyk])
        multiples = [prob for prob,mult in probMultiplicities.items() if mult > 1]
        ## Only the multiple variables remain
        for prob in multiples:
          errorMsgs += "In variable " + str(keyk) + " the Threshold " + str(prob)+" appears multiple times!!\n"
          errorFound = True
#         self.branchProbabilities[self.toBeSampled[keyk]] = gridInfo[keyk][2]
#         self.branchProbabilities[self.toBeSampled[keyk]].sort(key=float)
#         if max(self.branchProbabilities[self.toBeSampled[keyk]]) > 1:
#           self.raiseAWarning("One of the Thresholds for distribution " + str(gridInfo[keyk][2]) + " is > 1")
#           errorFound = True
#           for index in range(len(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float))):
#             if sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float).count(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index]) > 1:
#               self.raiseAWarning("In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchProbabilities[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
#               errorFound = True
      else:
        self.branchValues[keyk] = gridInfo[keyk][2]
        self.branchValues[keyk].sort(key=float)
        valueMultiplicities = Counter(self.branchValues[keyk])
        multiples = [value for value,mult in valueMultiplicities.items() if mult > 1]
        ## Only the multiple variables remain
        for value in multiples:
          errorMsgs += "In variable " + str(keyk) + " the Threshold " + str(value)+" appears multiple times!!\n"
          errorFound = True
#         self.branchValues[self.toBeSampled[keyk]] = gridInfo[keyk][2]
#         self.branchValues[self.toBeSampled[keyk]].sort(key=float)
#         for index in range(len(sorted(self.branchValues[self.toBeSampled[keyk]], key=float))):
#           if sorted(self.branchValues[self.toBeSampled[keyk]], key=float).count(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index]) > 1:
#             self.raiseAWarning("In distribution " + str(self.toBeSampled[keyk]) + " the Threshold " + str(sorted(self.branchValues[self.toBeSampled[keyk]], key=float)[index])+" appears multiple times!!")
#             errorFound = True

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
        if child.attrib['type'] in self.hybridStrategyToApply.keys():
          self.raiseAnError(IOError,'Hybrid Sampler type '+child.attrib['type'] + ' already inputted!')
        if child.attrib['type'] not in self.hybridSamplersAvail.keys():
          self.raiseAnError(IOError,'Hybrid Sampler type ' +child.attrib['type'] + ' unknown. Available are '+ ','.join(self.hybridSamplersAvail.keys()) + '!')
        self.hybridNumberSamplers = 1
        # the user can decided how to sample the epistemic
        self.hybridStrategyToApply[child.attrib['type']] = self.hybridSamplersAvail[child.attrib['type']]()
        # give the hybridsampler sampler the message handler
        self.hybridStrategyToApply[child.attrib['type']].setMessageHandler(self.messageHandler)
        # make the hybridsampler sampler read  its own xml block
        childCopy = copy.deepcopy(child)
        childCopy.tag = child.attrib['type']
        childCopy.attrib.pop('type')
        self.hybridStrategyToApply[child.attrib['type']]._readMoreXML(childCopy)
        # store the variables that represent the epistemic space
        self.epistemicVariables.update(dict.fromkeys(self.hybridStrategyToApply[child.attrib['type']].toBeSampled.keys(),{}))

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for key in self.branchProbabilities.keys():
      paramDict['Probability Thresholds for var ' + str(key) + ' are: '] = [str(x) for x in self.branchProbabilities[key]]
    for key in self.branchValues.keys()       :
      paramDict['Values Thresholds for var ' + str(key) + ' are: '] = [str(x) for x in self.branchValues[key]]
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
    if len(self.hybridStrategyToApply.keys()) > 0:
      hybridlistoflist = []
    for cnt, preckey  in enumerate(self.hybridStrategyToApply.keys()):
      hybridsampler =  self.hybridStrategyToApply[preckey]
      hybridlistoflist.append([])
      hybridsampler.initialize()
      self.hybridNumberSamplers *= hybridsampler.limit
      while hybridsampler.amIreadyToProvideAnInput():
        hybridsampler.counter +=1
        hybridsampler.localGenerateInput(None,None)
        hybridsampler.inputInfo['prefix'] = hybridsampler.counter
        hybridlistoflist[cnt].append(copy.deepcopy(hybridsampler.inputInfo))
    if self.hybridNumberSamplers > 0:
      self.raiseAMessage('Number of Hybrid Samples are ' + str(self.hybridNumberSamplers) + '!')
      hybridNumber = self.hybridNumberSamplers
      combinations = list(itertools.product(*hybridlistoflist))
    else:
      hybridNumber = 1
    self.TreeInfo = {}
    for precSample in range(hybridNumber):
      elm = ETS.HierarchicalNode(self.messageHandler,self.name + '_' + str(precSample+1))
      elm.add('name', self.name + '_'+ str(precSample+1))
      elm.add('startTime', str(0.0))
      # Initialize the endTime to be equal to the start one...
      # It will modified at the end of each branch
      elm.add('endTime', str(0.0))
      elm.add('runEnded',False)
      elm.add('running',True)
      elm.add('queue',False)
      # if preconditioned DET, add the sampled from hybridsampler samplers
      if self.hybridNumberSamplers > 0:
        elm.add('hybridsamplerCoordinate', combinations[precSample])
        for point in combinations[precSample]:
          for epistVar, val in point['SampledVars'].items():
            self.epistemicVariables[epistVar][elm.get('name')] = val
      # The dictionary branchedLevel is stored in the xml tree too. That's because
      # the advancement of the thresholds must follow the tree structure
      elm.add('branchedLevel', self.branchedLevel[0])
      # Here it is stored all the info regarding the DET => we create the info for all the
      # branchings and we store them
      self.TreeInfo[self.name + '_' + str(precSample+1)] = ETS.HierarchicalTree(self.messageHandler,elm)

    for key in self.branchProbabilities.keys():
      #kk = self.toBeSampled.values().index(key)
      #self.branchValues[key] = [self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].ppf(float(self.branchProbabilities[key][index])) for index in range(len(self.branchProbabilities[key]))]
      self.branchValues[key] = [self.distDict[key].ppf(float(self.branchProbabilities[key][index])) for index in range(len(self.branchProbabilities[key]))]
    for key in self.branchValues.keys():
      #kk = self.toBeSampled.values().index(key)
      #self.branchProbabilities[key] = [self.distDict[self.toBeSampled.keys()[self.toBeSampled.values().index(key)]].cdf(float(self.branchValues[key][index])) for index in range(len(self.branchValues[key]))]
      self.branchProbabilities[key] = [self.distDict[key].cdf(float(self.branchValues[key][index])) for index in range(len(self.branchValues[key]))]
    self.limit = sys.maxsize
    # add expected metadata
    self.addMetaKeys(['RAVEN_parentID','RAVEN_isEnding'])
