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
  Created on January 10, 2019

  @author: talbpaul
  Container to handle ROMs that are made of many sub-roms
"""
# standard libraries
from __future__ import division, print_function, absolute_import
import copy
import warnings
from collections import defaultdict
# external libraries
import abc
import numpy as np
# internal libraries
from utils import mathUtils, xmlUtils
from .SupervisedLearning import supervisedLearning

warnings.simplefilter('default', DeprecationWarning)


#
#
#
#
class Collection(supervisedLearning):
  """
    A container that handles collections of ROMs in a particular way.
  """
  def __init__(self, messageHandler, **kwargs):
    """
      Constructor.
      @ In, messageHandler, MesageHandler.MessageHandler, message tracker
      @ In, kwargs, dict, options and initialization settings (from XML)
      @ Out, None
    """
    supervisedLearning.__init__(self, messageHandler, **kwargs)
    self.printTag = 'ROM Collection'              # message printing appearance
    self._romName = kwargs.get('name', 'unnamed') # name of the requested ROM
    self._templateROM = kwargs['modelInstance']   # example of a ROM that will be used in this grouping, set by setTemplateROM
    self._roms = []                               # ROMs that belong to this grouping.
    # TODO dynamicHandling? depends on sub-ROMs

  def __getstate__(self):
    nope = ['_divisionClassifier', '_assembledObjects']
    for k in self.__dict__:
      if k not in nope:
        print('. ', k)
    d = dict((key, val) for key, val in self.__dict__.items() if key not in nope) # deepcopy needed
    return d

  @abc.abstractmethod
  def train(self, tdict):
    """
      Trains the SVL and its supporting SVLs. Overwrites base class behavior due to special clustering needs.
      @ In, trainDict, dict, dicitonary with training data
      @ Out, None
    """
    # TODO can we get away without overwriting method?

  @abc.abstractmethod
  def evaluate(self, edict):
    """
      Method to evaluate a point or set of points via surrogate.
      Overwritten for special needs in this ROM
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, np.array, evaluated points
    """
    # TODO can we get away without overwriting method?

  def reseed(self, seed):
    pass # XXX

  # dummy methods that are required by SVL and not generally used
  def __confidenceLocal__(self, featureVals):
    pass
  def __resetLocal__(self):
    pass
  def __returnCurrentSettingLocal__(self):
    return {}
  def __returnInitialParametersLocal__(self):
    return {}
  # TODO find a way to use these like other SVL?
  def __evaluateLocal__(self, featureVals):
    pass
  def __trainLocal__(self, featureVals, targetVals):
    pass

#
#
#
#
class Segments(Collection):
  """
    A container that handles ROMs that are segmented along some set of indices
  """
  ########################
  # CONSTRUCTION METHODS #
  ########################
  def __init__(self, messageHandler, **kwargs):
    """
      Constructor.
      @ In, messageHandler, MesageHandler.MessageHandler, message tracker
      @ In, kwargs, dict, options and initialization settings (from XML)
      @ Out, None
    """
    Collection.__init__(self, messageHandler, **kwargs)
    self.printTag = 'Segmented ROM'
    self._divisionInstructions = {}    # which parameters are clustered, and how, and how much?
    self._divisionMetrics = None       # requested metrics to apply; if None, implies everything we know about
    self._divisionInfo = {}            # data that should persist across methods
    self._divisionPivotShift = {}      # whether and how to normalize/shift subspaces
    self._indexValues = {}             # original index values, by index
    self._originalTrainingData = {}    # stores the data on which this ROM was originally trained; TODO redundance with indexValues?
    # allow some ROM training to happen globally, seperate from individual segment training
    ## see design note for Clusters
    self._romGlobalAdjustments = None  # global ROM settings, provided by the templateROM before clustering

    # set up segmentation
    # get input specifications from inputParams
    inputSpecs = kwargs['paramInput'].findFirst('Segment')
    # initialize settings
    divisionMode = {}
    for node in inputSpecs.subparts:
      if node.name == 'subspace':
        subspace = node.value
        # check for duplicate definition
        if subspace in divisionMode.keys():
          self.raiseAWarning('Subspace was defined multiple times for "{}"! Using the first.'.format(subspace))
          continue
        # check correct arguments are given
        if 'divisions' in node.parameterValues and 'pivotLength' in node.parameterValues:
          self.raiseAnError(IOError, 'Cannot provide both \'pivotLength\' and \'divisions\' for subspace "{}"!'.format(subspace))
        if 'divisions' not in node.parameterValues and 'pivotLength' not in node.parameterValues:
          self.raiseAnError(IOError, 'Must provide either \'pivotLength\' or \'divisions\' for subspace "{}"!'.format(subspace))
        # determine segmentation type
        if 'divisions' in node.parameterValues:
          # splitting a particular number of times (or "divisions")
          mode = 'split'
          key = 'divisions'
        elif 'pivotLength' in node.parameterValues:
          # splitting by pivot parameter values
          mode = 'value'
          key = 'pivotLength'
        divisionMode[subspace] = (mode, node.parameterValues[key])
        # standardize pivot param?
        if 'shift' in node.parameterValues:
          shift = node.parameterValues['shift'].strip().lower()
          # check value given makes sense (either "zero" or "first")
          acceptable = ['zero', 'first']
          if shift not in [None] + acceptable:
            self.raiseAnError(IOError, 'If <subspace> "shift" is specificed, it must be one of {}; got "{}".'.format(acceptable, shift))
          self._divisionPivotShift[subspace] = shift
        else:
          self._divisionPivotShift[subspace] = None
    self._divisionInstructions = divisionMode
    if len(self._divisionInstructions) > 1:
      self.raiseAnError(NotImplementedError, 'Segmented ROMs do not yet handle multiple subspaces!')


  ###############
  # RUN METHODS #
  ###############
  def train(self, tdict): #featureVals, targetVals):
    """
      Trains the SVL and its supporting SVLs. Overwrites base class behavior due to special clustering needs.
      @ In, trainDict, dict, dicitonary with training data
      @ Out, None
    """
    self.readAssembledObjects()
    # subdivide space
    divisions = self._subdivideDomain(self._divisionInstructions, tdict)
    self._divisionInfo['delimiters'] = divisions[0] + divisions[1]
    # DEBUGG only
    self._originalTrainingData = copy.deepcopy(tdict)
    # allow ROM to handle some global features
    self._romGlobalAdjustments, newTrainingDict = self._templateROM.getGlobalRomSegmentSettings(tdict, divisions)
    # train segments
    self._trainBySegments(divisions, newTrainingDict)
    self.amITrained = True
    self._templateROM.amITrained = True # TODO is this a safe thing to do??

  def evaluate(self, edict):
    """
      Method to evaluate a point or set of points via surrogate.
      Overwritten for special needs in this ROM
      @ In, edict, dict, evaluation dictionary
      @ Out, result, np.array, evaluated points
    """
    result = self._evaluateBySegments(edict)
    # allow each segment ROM to modify signal based on global training settings
    for s, segment in enumerate(self._getSequentialRoms()):
      delim = self._divisionInfo['delimiters'][s]
      picker = slice(delim[0], delim[-1] + 1)
      result = segment.finalizeLocalRomSegmentEvaluation(self._romGlobalAdjustments, result, picker)
    result = self._templateROM.finalizeGlobalRomSegmentEvaluation(self._romGlobalAdjustments, result)
    return result

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    # TODO write global information
    newNode = xmlUtils.StaticXmlElement('GlobalROM', attrib={'segment':'all'})
    self._templateROM.writeXML(newNode, targets, skip)
    writeTo.getRoot().append(newNode.getRoot())
    # write subrom information
    for i, rom in enumerate(self._roms):
      newNode = xmlUtils.StaticXmlElement('SegmentROM', attrib={'segment':str(i)})
      rom.writeXML(newNode, targets, skip)
      writeTo.getRoot().append(newNode.getRoot())


  ###################
  # UTILITY METHODS #
  ###################
  def _evaluateBySegments(self, evaluationDict):
    """
      Evaluate ROM by evaluating its segments
      @ In, evaluationDict, dict, realization to evaluate
      @ Out, result, dict, dictionary of results
    """
    # TODO assuming only subspace is pivot param
    pivotID = self._templateROM.pivotParameterID
    lastEntry = self._divisionInfo['historyLength']
    result = None  # we don't know the targets yet, so wait until we get the first evaluation to set this up
    nextEntry = 0  # index to fill next data set into
    self.raiseADebug('Sampling from {} segments ...'.format(len(self._roms)))
    roms = self._getSequentialRoms()
    for r, rom in enumerate(roms):
      self.raiseADebug('Evaluating ROM segment', r)
      subResults = rom.evaluate(evaluationDict)
      # NOTE the pivot values for subResults will be wrong (shifted) if shifting is used in training
      ## however, we will set the pivotID values all at once after all results are gathered, so it's okay.
      # build "results" structure if not already done -> easier to do once we gather the first sample
      if result is None:
        # TODO would this be better stored as a numpy array instead?
        result = dict((target, np.zeros(lastEntry)) for target in subResults.keys())
      # place subresult into overall result # TODO this assumes consistent history length! True for ARMA at least.
      entries = len(list(subResults.values())[0])
      # There's a problem here, if using Clustering; the residual shorter-length element at the end might be represented
      #   by a ROM that expects to deliver the full signal.  TODO this should be handled in a better way,
      #   but for now we can truncate the signal to the length needed
      for target, values in subResults.items():
        # skip the pivotID
        if target == pivotID:
          continue
        if len(result[target][nextEntry:]) < len(values):
          result[target][nextEntry:] = values[:len(result[target][nextEntry:])]
        else:
          result[target][nextEntry:nextEntry + entries] = values
      # update next subdomain storage location
      nextEntry += entries
    # place pivot values
    result[pivotID] = self._indexValues[pivotID]
    return result

  def _getSequentialRoms(self):
    """
      Returns ROMs in sequential order. Trivial for Segmented.
      @ In, None
      @ Out, list, list of ROMs in order (pointer, not copy)
    """
    return self._roms

  def _subdivideDomain(self, divisionInstructions, trainingSet):
    """
      Creates markers for subdividing the pivot parameter domain, either based on number of subdivisions
      or on requested pivotValue lengths.
      @ In, divisionInstructions, dict, dictionary of inputs/indices to cluster on mapped to either
               number of subdivisions to make or length of the pivot value segments to include
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ Out, counter, list(tuple), indices that belong to each division; at minimum (first index, last index)
      @ Out, unclustered, list(tuple), as "counter" but for segments that will not be clustered
    """
    unclustered = []
    # division instructions are as {subspace: (mode, value)}
    ## where "value" is the number of segments in "split" mode
    ## or the length of pivot values per segment in "value" mode
    self.raiseADebug('Training segmented subspaces for "{}" ...'.format(self._romName))
    for subspace, (mode, value) in divisionInstructions.items():
      dataLen = len(trainingSet[subspace][0]) # TODO assumes syncronized histories, or single history
      self._divisionInfo['historyLength'] = dataLen # TODO assumes single pivotParameter
      if mode == 'split':
        numSegments = value # renamed for clarity
        # divide the subspace into equally-sized segments, store the indexes for each segment
        counter = np.array_split(np.arange(dataLen), numSegments)
        # TODO don't over-store data; only store the first and last index in "counter"!
        # Note that "segmented" doesn't have "unclustered" since chunks are evenly sized
      elif mode == 'value':
        segmentValue = value # renamed for clarity
        # divide the subspace into segments with roughly the same pivot length (e.g. time length)
        pivot = trainingSet[subspace][0]
        # find where the data passes the requested length, and make dividers
        floor = 0                # where does this pivot segment start?
        nextOne = segmentValue   # how high should this pivot segment go?
        counter = []
        # FIXME can we do this without looping?
        while pivot[floor] < pivot[-1]:
          cross = np.searchsorted(pivot, nextOne)
          # if the next crossing point is past the end, put the remainder piece
          ## into the "unclustered" grouping, since it might be very oddly sized
          ## and throw off segmentation (specifically for clustering)
          if cross == len(pivot):
            unclustered.append((floor, cross - 1))
            break
          # add this segment, only really need to know the first and last index (inclusive)
          counter.append((floor, cross - 1))
          # update search parameters
          floor = cross
          nextOne += segmentValue
      self.raiseADebug('Dividing {:^20s} into {:^5d} divisions for training ...'.format(subspace, len(counter) + len(unclustered)))
    # return the counter indicies as well as any odd-piece-out parts
    return counter, unclustered

  def _trainBySegments(self, divisions, trainingSet):
    """
      Train ROM by training many ROMs depending on the input/index space clustering.
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ Out, None
    """
    # train the subdomain ROMs
    counter, remainder = divisions
    roms = self._trainSubdomainROMs(self._templateROM, counter, trainingSet, self._romGlobalAdjustments)
    # if there were leftover domain segments that didn't go with the rest, train those now
    if remainder:
      unclusteredROMs = self._trainSubdomainROMs(self._templateROM, remainder, trainingSet, self._romGlobalAdjustments)
      roms = np.hstack([roms, unclusteredROMs])
    self._roms = roms

  def _trainSubdomainROMs(self, templateROM, counter, trainingSet, romGlobalAdjustments):
    """
      Trains the ROMs on each clusterable subdomain
      @ In, templateROM, SupervisedLEarning.supervisedLearning instance, template ROM
      @ In, counter, list(tuple), instructions for dividing subspace into subdomains
      @ In, trainingSet, dict, data on which ROMs should be trained
      @ In, romGlobalAdjustments, object, arbitrary container created by ROMs and passed to ROM training
      @ Out, roms, np.array(supervisedLearning), trained ROMs for each subdomain
    """
    targets = templateROM.target[:]
    # clear indices from teh training list, since they're independents
    # TODO assumes pivotParameter is the only subspace being divided
    pivotID = templateROM.pivotParameterID
    targets.remove(pivotID)
    # stash pivot values, since those will break up while training segments
    # TODO assumes only pivot param
    if pivotID not in self._indexValues:
      self._indexValues[pivotID] = trainingSet[pivotID][0]
    # loop over clusters and train data
    roms = []
    for i, subdiv in enumerate(counter):
      # slicer for data selection # TODO extract as a small utility method?
      picker = slice(subdiv[0], subdiv[-1] + 1)
      ## TODO we need to be slicing all the data, not just one realization, once we support non-ARMA segmentation.
      data = dict((var, [copy.deepcopy(trainingSet[var][0][picker])]) for var in trainingSet)
      # renormalize the pivot if requested, e.g. by shifting values
      norm = self._divisionPivotShift[pivotID]
      if norm:
        if norm == 'zero':
          # left-shift pivot so subspace starts at 0 each time
          delta = data[pivotID][0][0]
        elif norm == 'first':
          # left-shift so that first entry is equal to pivot's first value (maybe not zero)
          delta = data[pivotID][0][0] - trainingSet[pivotID][0][0]
        data[pivotID][0] -= delta
      # create a new ROM and train it!
      newROM = copy.deepcopy(templateROM)
      newROM.name = '{}_seg{}'.format(self._romName, i)
      newROM.adjustLocalRomSegment(self._romGlobalAdjustments)
      self.raiseADebug('Training segment', i, picker)
      newROM.train(data)
      roms.append(newROM)
    # format array for future use
    roms = np.array(roms)
    return roms



#
#
#
#
class Clusters(Segments):
  """
    A container that handles ROMs that use clustering to subdivide their space

    Some design notes:
    The individual SupervisedLearning algorithms need to be able to contribute to how a ROM
    collection gets trained, clustered, and evaluated. As such, there's communication between
    three entities: the Collection (this class), the templateROM (a single instance of the SVL), and
    the segment ROMs (the group of SVLs).  

    The templateROM is allowed to perform arbitrary global modifications to the training signal before
    the signal is subdivided and sent to individual ROMs. For instance, some global training may not
    apply at a local level. The method "getGlocalRomSegmentSettings" is called on the templateROM to
    get both the modified training data as well as the "settings", which won't ever be read by the
    Collection but will be provided to the segment ROMs at several times.

    Before training, the "settings" from the global training may need to make modifications in the
    input parameters of the segment ROMs, so these settings are passed to "adjustLocalRomSegment" on
    each segment ROM so it can update its state to reflect the global settings.

    Upon evaluating the collection, first the individual segment ROMs are evaluated, and then first
    local then global finalizing modifications are allowed based on the global training "setings". These
    are managed by calling finalizeLocalRomSegmentEvaluation on each segment ROM, and
    finalizeGlovalRomSegmentEvaluation on the templateROM.
  """
  ## Constructors ##
  def __init__(self, messageHandler, **kwargs):
    """
      Constructor.
      @ In, messageHandler, MesageHandler.MessageHandler, message tracker
      @ In, kwargs, dict, options and initialization settings (from XML)
      @ Out, None
    """
    Segments.__init__(self, messageHandler, **kwargs)
    self.printTag = 'Clustered ROM'
    self._divisionClassifier = None      # Classifier to cluster subdomain ROMs
    self._metricClassifiers = None       # Metrics for clustering subdomain ROMs
    self._clusterInfo = {}               # contains all the useful clustering results
    self._clusterGroupExpansionInfo = {} # dict of global expansions for local clusters
    self._evaluationMode = 'truncated'   # TODO make user option, whether returning full histories or truncated ones
    self._featureTemplate = '{target}|{metric}|{id}' # created feature ID template

    ## check if ROM has methods to cluster on (errors out if not)
    # TODO find a better way
    if not callable(getattr(self._templateROM, "getLocalRomClusterFeatures", None)):
      self.raiseAnError(NotImplementedError, 'Requested ROM "{}" does not yet have methods for clustering!'.format(self._romName))

  def readAssembledObjects(self):
    """
      Collects the entities from the Assembler as needed.
      Clusters need the classifer to cluster by, as well as any additional clustering metrics
      @ In, None
      @ Out, None
    """
    # get the classifier to use, if any, from the Assembler
    ## this is used to cluster the ROM segments
    self._divisionClassifier = self._assembledObjects.get('Classifier', [[None]*4])[0][3]
    self._metricClassifiers = self._assembledObjects.get('Metric', None)

  ## API ##

  def evaluate(self, edict):
    """
      Method to evaluate a point or set of points via surrogate.
      Overwritten for special needs in this ROM
      @ In, edict, dict, evaluation dictionary
      @ Out, result, np.array, evaluated points
    """
    if self._evaluationMode == 'full':
      # TODO there's no input-based way to request this mode right now, so it's not tested as much
      result = Segments.evaluate(self, edict)
    if self._evaluationMode == 'truncated':
      result, weights = self._createTruncatedEvaluation(edict)
      for r, rom in enumerate(self._roms):
        # "r" is the cluster
        clusterIndex = list(self._clusterInfo['map'][r]).index(rom)
        segmentIndex = self._getSegmentIndexFromClusterIndex(r, self._clusterInfo['labels'], clusterIndex=clusterIndex)
        delim = self._divisionInfo['delimiters'][r]
        picker = slice(delim[0], delim[-1] + 1)
        result = rom.finalizeLocalRomSegmentEvaluation(self._romGlobalAdjustments, result, picker)
      result = self._templateROM.finalizeGlobalRomSegmentEvaluation(self._romGlobalAdjustments, result, weights=weights)
    return result

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    Segments.writeXML(self, writeTo, targets, skip)
    main = writeTo.getRoot()
    labels = self._clusterInfo['labels']
    for i, repRom in enumerate(self._roms):
      # find associated node
      modify = xmlUtils.findPath(main, 'SegmentROM[@segment={}]'.format(i))
      modify.tag = 'ClusterROM'
      modify.attrib['cluster'] = modify.attrib.pop('segment')
      modify.append(xmlUtils.newNode('segments_represented', text=', '.join(str(x) for x in np.arange(len(labels))[labels==i])))
    # TODO add clustering information to existing nodes

  ## Utilities ##
  def _calculateBasicMetrics(self, data):
    """
      Evaluates basic statistical data for clustering.
      Someday should probably leverage BasicStatistics!
      @ In, data, dict, data to compute metrics on
      @ Out, metrics, dict, {feature:value} for features like "target_mean" and etc
    """
    metrics = {}
    for target, values in data.items():
      # mean
      feature = self._featureTemplate.format(target=target, metric='Basic', id='mean')
      metrics[feature] = np.average(values)
      # std dev
      feature = self._featureTemplate.format(target=target, metric='Basic', id='std')
      metrics[feature] = np.std(values)
    return metrics

  def _classifyROMs(self, classifier, features, clusterFeatures):
    """
      Classifies the subdomain ROMs.
      @ In, classifier, Models.PostProcessor, classification model to use
      @ In, features, list(str), ordered list of features
      @ In, clusterFeatures, dict, data on which to train classifier
      @ Out, labels, list(int), ordered list of labels corresponding to ROM subdomains
    """
    # the actual classifying algorithms is the unSupervisedEnging of the QDataMining of the PP Model
    ## get the instance
    classifier = classifier.interface.unSupervisedEngine
    # update classifier features
    classifier.updateFeatures(features)
    # make the clustering instance)
    classifier.train(clusterFeatures)
    # label the training data
    labels = classifier.evaluate(clusterFeatures)
    return labels

  def _createTruncatedEvaluation(self, evaluationDict):
    """
      Evaluates truncated representation of ROM
      @ In, evaluationDict, dict, realization to evaluate
      @ Out, result, dict, dictionary of results
      @ Out, sampleWeights, np.array, array of cluster weights (normalized)
    """
    pivotID = self._templateROM.pivotParameterID
    result = None # populate on first sample -> could use defaultdict, but that's more lenient
    sampleWeights = [] # importance of each cluster
    # sample signal, one piece for each segment
    labelMap = self._clusterInfo['labels']
    clusters = sorted(list(set(labelMap)))
    for cluster in clusters:
      # choose a ROM
      chooseRomMode = 'first' # TODO user option? alternative is random
      if chooseRomMode == 'first':
        ## option 1: just take the first one
        segmentIndex, clusterIndex = self._getSegmentIndexFromClusterIndex(cluster, labelMap, clusterIndex=0)
      elif chooseRomMode == 'random':
        ## option 2: choose randomly
        segmentIndex, clusterIndex = self._getSegmentIndexFromClusterIndex(cluster, labelMap, chooseRandom=True)
      # grab the Chosen ROM to represent this cluster
      rom = self._clusterInfo['map'][cluster][clusterIndex] # the Chosen ROM
      # get the slice opject that picks out the history range associated to the Chosen Segment
      delimiter = self._divisionInfo['delimiters'][segmentIndex]
      picker = slice(delimiter[0], delimiter[-1] + 1)
      # evaluate the ROM
      subResults = rom.evaluate(evaluationDict) # TODO assumes only scalar inputs. That's a RAVEN truth right now though.
      if result is None:
        result = dict((target, []) for target in subResults)
      # populate weights
      sampleWeights.append(np.ones(len(subResults[pivotID])) * len(self._clusterInfo['map'][cluster]))
      for target, values in subResults.items():
        result[target].append(values)
    # combine histories (we stored each one as a distinct array during collecting)
    for target, values in result.items():
      result[target] = np.hstack(values)
    result[pivotID] = self._indexValues[pivotID][:len(result[pivotID])]
    sampleWeights = np.hstack(sampleWeights)
    sampleWeights /= sum(sampleWeights)
    return result, sampleWeights

  def _getSegmentIndexFromClusterIndex(self, cluster, labelMap, clusterIndex=None, chooseRandom=False):
    """
      Given the index of a rom WITHIN a cluster, get the index of that rom's segment in the full history
      @ In, cluster, int, label of cluster that ROM is within
      @ In, labelMap, list(int), map of where clusters appear in order of full history
      @ In, clusterIndex, int, optional, index of ROM within the cluster
      @ In, chooseRandom, bool, optional, if True then choose randomly from eligible indices
      @ Out, segmentIndex, int, position of rom's segment in full history
      @ Out, clusterIndex, int, position of rom within cluster (returned in event of random)
    """
    # Need to either provide the index, or let it be random, but not both
    ## TODO modify to use internal RNG from randUtils
    assert not (clusterIndex is None and chooseRandom is False)
    assert not (clusterIndex is not None and chooseRandom is True)
    # indices of all segments
    indices = np.arange(len(labelMap))
    # indices who belong to this cluster
    eligible = indices[labelMap == cluster]
    # if random, choose now
    if chooseRandom:
      clusterIndex = np.random.choice(range(len(eligible)))
    # global index
    segmentIndex = eligible[clusterIndex]
    return segmentIndex, clusterIndex

  def _gatherClusterFeatures(self, roms, counter, trainingSet):
    """
      Collects features of the ROMs for clustering purposes
      @ In, roms, list, list of segmented SVL ROMs
      @ In, counter, list(tuple), instructions for dividing subspace into subdomains
      @ In, trainingSet, dict, data on which ROMs should be trained
      @ Out, clusterFeatures, dict, clusterable parameters as {feature: [rom values]}
    """
    targets = self._templateROM.target[:]
    pivotID = self._templateROM.pivotParameterID
    targets.remove(pivotID)
    clusterFeatures = defaultdict(list)
    for r, rom in enumerate(roms):
      # select pertinent data
      ## NOTE assuming only "leftover" roms are at the end, so the rest are sequential and match "counters"
      picker = slice(counter[r][0], counter[r][-1]+1)
      # OLD data = dict((var, [copy.deepcopy(trainingSet[var][0][picker])]) for var in trainingSet)
      ## DEBUGG temporarily? disable basic metrics ##
      # get general basic metrics (aka cluster features)
      #basicData = self._calculateBasicMetrics(targetData)
      #for feature, val in basicData.items():
      #  clusterFeatures[feature].append(val)
      # get ROM-specific metrics
      romData = rom.getLocalRomClusterFeatures(self._featureTemplate, self._romGlobalAdjustments, picker=picker)
      for feature, val in romData.items():
        clusterFeatures[feature].append(val)
    return clusterFeatures

  def _getSequentialRoms(self):
    """
      Returns ROMs in sequential order.
      @ In, None
      @ Out, _getSequentialRoms, list of ROMs in order (pointer, not copy)
    """
    # TODO always returns the first cluster currently. Should be different?
    return list(self._roms[l] for l in self._clusterInfo['labels'])

  def _trainBySegments(self, divisions, trainingSet):
    """
      Train ROM by training many ROMs depending on the input/index space clustering.
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ Out, None
    """
    # subdivide domain and train subdomain ROMs, as with the segmentation
    ## TODO can we increase the inheritance more here, or is this the minimum cutset?
    counter, remainder = divisions
    # store delimiters
    if len(remainder):
      self.raiseADebug('"{}" division(s) are being excluded from clustering consideration.'.format(len(remainder)))
    ## train ROMs for each segment
    roms = self._trainSubdomainROMs(self._templateROM, counter, trainingSet, self._romGlobalAdjustments)
    # collect ROM features (basic stats, etc)
    clusterFeatures = self._gatherClusterFeatures(roms, counter, trainingSet)
    # future: requested metrics
    ## TODO someday
    # weight and scale data
    ## create hierarchy for cluster params
    features = sorted(clusterFeatures.keys())
    hierarchFeatures = defaultdict(list)
    for feature in features:
      _, metric, ident = feature.split('|', 2)
      # the same identifier might show up for multiple targets
      if ident not in hierarchFeatures[metric]:
        hierarchFeatures[metric].append(ident)
    ## weighting strategy, TODO make optional for the user
    weightingStrategy = 'uniform'
    clusterFeatures = self._weightAndScaleClusters(features, hierarchFeatures, clusterFeatures, weightingStrategy)
    # perform clustering
    labels = self._classifyROMs(self._divisionClassifier, features, clusterFeatures)
    uniqueLabels = sorted(list(set(labels))) # note: keep these ordered!
    self.raiseAMessage('Identified {} clusters while training clustered ROM "{}".'.format(len(uniqueLabels), self._romName))
    # if there were some segments that won't compare well (e.g. leftovers), handle those separately
    if len(remainder):
      unclusteredROMs = self._trainSubdomainROMs(self._templateROM, remainder, trainingSet, self._romGlobalAdjustments)
    else:
      unclusteredROMs = []
    # make cluster information dict
    self._clusterInfo = {'labels':labels}
    ## clustered
    self._clusterInfo['map'] = dict((label, roms[labels == label]) for label in uniqueLabels)
    ## unclustered
    self._clusterInfo['map']['unclustered'] = unclusteredROMs
    #############
    #   DEBUG   #
    #############
    _plotSignalsClustered(labels, clusterFeatures, counter, self._originalTrainingData)
    #############
    # END DEBUG #
    #############
    # TODO what about the unclustered ones? Do we throw them out in truncated representation?
    self._roms = list(self._clusterInfo['map'][label][0] for label in uniqueLabels)

  def _weightAndScaleClusters(self, features, featureGroups, clusterFeatures, weightingStrategy):
    """
      Applies normalization and weighting to cluster training features.
      @ In, features, list(str), ordered list of features
      @ In, featureGroups, dict, hierarchal structure of requested features
      @ In, clusterFeaturs, dict, features mapped to arrays of values (per ROM)
      @ In, weightingStrategy, str, weighting strategy to use in ROM metrics
      @ Out, clusterFeatures, dict, weighted and scaled feature space (destructive on original dict)
    """
    # initialize structure
    weights = np.zeros(len(features))
    for f, feature in enumerate(features):
      # scale the data
      data = np.asarray(clusterFeatures[feature])
      # using Z normalization allows the data that is truly far apart to be streched,
      ## while data that is close together remains clustered.
      ## This does not work well if SMALL relative differences SHOULD make a big difference in clustering,
      ##    or if LARGE relative distances should NOT make a big difference in clustering!
      loc, scale = mathUtils.normalizationFactors(data, mode='z')
      clusterFeatures[feature] = (data - loc)/scale
      # weight the data --> NOTE doesn't really work like we think it does!
      _, metric, ID = feature.split('|', 2)
      if weightingStrategy == 'uniform':
        weight = 1.0
      else:
        # TODO when this gets moved to an input spec, we won't need to check it here.
        ## for now, though, it's the only option.
        self.raiseAnError(RuntimeError, 'Unrecognized weighting strategy: "{}"!'.format(weightingStrategy))
      weights[f] = weight
    # TODO? make sum of the weights unity
    ##weights /= np.sum(weights)
    ## alternative weighting that didn't show much promise so far: by volume
    # vol = np.product(list(np.max(v) for v in clusterFeatures.values()))
    # newVolume = np.product(weights)
    # oldVolume = 1.0
    # scale = (oldVolume / newVolume)**(1.0 / float(len(features)))
    ## END scale by volume
    for f, feature in enumerate(features):
      clusterFeatures[feature] = clusterFeatures[feature] * weights[f]
    return clusterFeatures

#
#
#
#
# DEBUGGING TOOLS
def _plotSignalsClustered(labels, clusterFeatures, slices, trainingSet):
  # dump training parameters
  import pandas as pd
  trainDF = pd.DataFrame(clusterFeatures)
  trainDF['labels'] = labels
  trainDF.to_csv('debug_clustering.csv')
  # plot
  import matplotlib.pyplot as plt
  for target in trainingSet:
    if target in ['Time', 'scaling']:
      continue
    print('')
    print('DEBUGG plotting target "{}"'.format(target))
    fig, ax = plt.subplots(figsize=(12, 10))
    ymin = trainingSet[target][0].min()
    ymax = trainingSet[target][0].max()
    lim = (ymin, ymax)
    cluster_hist = list(np.zeros(len(trainingSet['Time'][0]))*np.nan for _ in range(max(labels)+1))
    for l, label in enumerate(labels):
      picker = slice(slices[l][0], slices[l][-1]+1)
      data = dict((var, [trainingSet[var][0][picker]]) for var in trainingSet)
      end = data['Time'][0][1]
      ax.plot([end, end], lim, 'k:', alpha=0.2)
      if l == 0:
        start = data['Time'][0][0]
        ax.plot([start, start], lim, 'k:', alpha=0.2)
      ax.plot(data['Time'][0], data[target][0], '-', color='C{}'.format(label%10), label='C {}'.format(label))
      cluster_hist[label][picker] = data[target][0]
    ax.set_ylabel(target)
    ax.set_xlabel('Time (s)')
    ax.set_title('{} Clustered Training Data'.format(target))
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.legend(loc=0)
    fig.savefig('{}_all_clusters.png'.format(target))

    # plot cluster histories, by cluster
    for l in range(max(labels)):
      fig, ax = plt.subplots(figsize=(12, 10))
      ax.plot(trainingSet['Time'][0], cluster_hist[l], color='C{}'.format(l%10), label=str(l))
      ax.legend(loc=0)
      ax.set_xlim(xlims)
      ax.set_ylim(ylims)
      ax.set_xlabel('Time (s)')
      ax.set_ylabel(target)
      ax.set_title('{} cluster {}'.format(target, l))
      fig.savefig('{}_cluster_{}.png'.format(target, l))


    #print('DEBUGG showing plot (close to continue) ...')
    #plt.show()




