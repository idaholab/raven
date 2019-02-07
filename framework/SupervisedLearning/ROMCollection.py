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
    # allow some ROM training to happen globally, seperate from individual segment training
    ## we accomplish this by calling getGlobalRomClusterSettings on the templateROM, then passing these
    ## settigns to the subspace segment ROMs before training, and finally again after? XXX evaluating them.
    self._romGlobalAdjustments = None  # global ROM settings, provided by the templateROM before clustering

    self.targetDatas = None            # DEBUGG only!

    # set up segmentation
    ## check if ROM has methods to cluster on (errors out if not)
    self._templateROM.getRomClusterParams()
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
    # DEBUGG only
    self._originalTrainingData = copy.deepcopy(tdict)
    # allow ROM to handle some global features
    self._romGlobalAdjustments, newTrainingDict = self._templateROM.getGlobalRomClusterSettings(tdict, divisions)
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
    result = self._templateROM.finalizeGlobalRomClusterSample(self._romGlobalAdjustments, result)
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
      for target, values in subResults.items():
        # skip the pivotID
        if target == pivotID:
          continue
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
    # DEBUGG
    if self.targetDatas is None: # DEBUGG
      self.targetDatas = []      # DEBUGG
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
      targetData = dict((var, data[var][0]) for var in targets)
      self.targetDatas.append(targetData) # DEBUGG
      # create a new ROM and train it!
      newROM = copy.deepcopy(templateROM)
      newROM.name = '{}_seg{}'.format(self._romName, i)
      newROM.setGlobalRomClusterSettings(self._romGlobalAdjustments)
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
    self._divisionClassifier = None  # Classifier to cluster subdomain ROMs
    self._metricClassifiers = None   # Metrics for clustering subdomain ROMs
    self._clusterInfo = {}            # contains all the useful clustering results
    self._featureTemplate = '{target}|{metric}|{id}' # created feature ID template

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
    uniqueLabels = set(labels)
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
    ## TODO self._roms needs to be populated by something ... is this it?
    self._roms = list(self._clusterInfo['map'][label][0] for label in uniqueLabels)

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
      data = dict((var, [copy.deepcopy(trainingSet[var][0][picker])]) for var in trainingSet)
      targetData = dict((var, data[var][0]) for var in targets)
      ## DEBUGG temporarily disable basic metrics ##
      # get general basic metrics (aka cluster features)
      #basicData = self._calculateBasicMetrics(targetData)
      #for feature, val in basicData.items():
      #  clusterFeatures[feature].append(val)
      # get ROM-specific metrics
      romData = rom.getRomClusterValues(self._featureTemplate)
      for feature, val in romData.items():
        clusterFeatures[feature].append(val)
    return clusterFeatures

  def _getSequentialRoms(self):
    """
      Returns ROMs in sequential order.
      @ In, None
      @ Out, _getSequentialRoms, list of ROMs in order (pointer, not copy)
    """
    return list(self._roms[l] for l in self._clusterInfo['labels'])

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
      loc, scale = mathUtils.normalizationFactors(data, mode='scale')
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
def _plotSignalsSegmented(labels, roms, targetDatas):
  """
    Debug tool. Should be removed or relocated when clustered ROMs are fully implemented.
    Plots the original data, colored by ROM clusters.
    @ In, labels, list(str), cluster labels corresponding to ROM order
    @ In, roms, list(SupervisedLearning), trained subset ROMs in the same order as labels
    @ In, targetDatas, dict, debugging tool
    @ Out, None
  """
  # TODO remove
  ## TODO doesn't work with shifted roms well.
  ## TODO can we make this a tool the user can use?
  targetDatas = np.array(targetDatas)
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  fig, ax = plt.subplots()
  ax.set_title('Clustered (Fourier)')
  legends = []
  for label in set(labels):
    # legend
    clr = ('C'+str(label % 10)) if label >= 0 else 'k'
    legends.append(Line2D([0], [0], color=clr))
    #figS,axS = plt.subplots()
    #axS.set_title('Compared: Cluster {}'.format(label))
    mask = labels == label
    for r in range(sum(mask)):
      rom = roms[mask][r]
      target = targetDatas[mask][r]
      x = rom.pivotParameterValues
      y = target['Signal']
      index = list(roms).index(rom)+1
      ax.plot(x, y, color=clr)
      ax.plot([x[0]]*2, [5000, 20000], 'k:')
      ax.plot([x[-1]]*2, [5000, 20000], 'k:')
      if (index - 1) % 4 == 0:
        ax.plot([x[0]]*2, [5000, 20000], 'k-')
      ax.text(np.average(x), 6000, str(index), ha='center')
      #axS.plot(x - x[0], y, label=str(list(roms).index(rom)+1))
    #axS.legend(loc=0)
  ax.legend(legends, list(set(labels)))
  fig.savefig('clusters.png')
  plt.show()

def _plotSignalsClustered(labels, clusterFeatures, slices, trainingSet):
  # dump training parameters
  import pandas as pd
  trainDF = pd.DataFrame(clusterFeatures)
  trainDF['labels'] = labels
  trainDF.to_csv('debug_clustering.csv')
  # plot
  import matplotlib.pyplot as plt
  _, ax = plt.subplots(figsize=(12, 10))
  for l, label in enumerate(labels):
    picker = slice(slices[l][0], slices[l][-1]+1)
    data = dict((var, [trainingSet[var][0][picker]]) for var in trainingSet)
    ax.plot(data['Time'][0], data['Signal'][0], '-', color='C{}'.format(label), label='C {}'.format(label))
  ax.set_ylabel('Demand')
  ax.set_xlabel('Time')
  ax.set_title('Clustered Training Data')
  ax.legend(loc=0)
  print('')
  print('DEBUGG showing plot (close to continue) ...')
  plt.show()


