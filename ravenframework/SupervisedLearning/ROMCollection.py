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

  @author: talbpaul, wangc
  Container to handle ROMs that are made of many sub-roms
"""
# standard libraries
import copy
import warnings
from collections import defaultdict, OrderedDict
import pprint

# external libraries
import abc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# internal libraries
from ..utils import utils, mathUtils, xmlUtils, randomUtils
from ..utils import InputData, InputTypes
from .SupervisedLearning import SupervisedLearning
# import pickle as pk # TODO remove me!
import os
#
#
#
#
class Collection(SupervisedLearning):
  """
    A container that handles collections of ROMs in a particular way.
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
    spec = super().getInputSpecification()
    spec.description = r"""To be added"""
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'ROM Collection'              # message printing appearance
    self._romName = None # name of the requested ROM
    self._templateROM = None   # example of a ROM that will be used in this grouping, set by setTemplateROM
    self._roms = []                               # ROMs that belong to this grouping.
    self._romInitAdditionalParams = {}            # used for deserialization, empty by default

  def setTemplateROM(self, romInfo):
    """
      Set the ROM that will be used in this grouping
      @ In, romInfo, dict, {'name':romName, 'modelInstance':romInstance}, the information used to set up template ROM
      @ Out, None
    """
    self._templateROM = romInfo.get('modelInstance')
    self._romName = romInfo.get('name', 'unnamed')
    if self._templateROM is None:
      self.raiseAnError(IOError, 'A rom instance is required by', self.name, 'please check your implementation')

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def __getstate__(self):
    """
      Customizes the serialization of this class.
      @ In, None
      @ Out, d, dict, dictionary with class members
    """
    # construct a list of unpicklable entties and exclude them from pickling
    ## previously, divisionClassifier was a problem, but it seems not to be now
    ## save this for future debugging
    ## nope = ['_divisionClassifier', '_assembledObjects']
    nope = ['_assembledObjects']
    # base class
    d = SupervisedLearning.__getstate__(self)
    # additional
    for n in nope:
      d.pop(n, None)
    return d

  @abc.abstractmethod
  def train(self, tdict):
    """
      Trains the SVL and its supporting SVLs. Overwrites base class behavior due to special clustering needs.
      @ In, trainDict, dict, dicitonary with training data
      @ Out, None
    """
    pass

  @abc.abstractmethod
  def evaluate(self, edict):
    """
      Method to evaluate a point or set of points via surrogate.
      Overwritten for special needs in this ROM
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, np.array, evaluated points
    """
    pass

  # dummy methods that are required by SVL and not generally used
  def __confidenceLocal__(self, featureVals):
    """
      This should return an estimation of the quality of the prediction.
      This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
      @ In, featureVals, 2-D numpy array , [n_samples,n_features]
      @ Out, __confidenceLocal__, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    return {}

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    return {}

  # Are private-ish so should not be called directly, so we don't implement them, as they don't fit the collection.
  def __evaluateLocal__(self, featureVals):
    """
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """
    pass

  def _train(self, featureVals, targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
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

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""\xmlNode{Segment} provides an alternative way to build the ROM. When
    this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
    requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
    during the subspace the ROM representation of the signal changes significantly. For example, if the signal
    is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
    segments. By default, no segmentation occurs."""
    # segmenting and clustering
    segment = InputData.parameterInputFactory("Segment", strictMode=True,
                                              descr=r"""provides an alternative way to build the ROM. When
                                                this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
                                                requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
                                                during the subspace the ROM representation of the signal changes significantly. For example, if the signal
                                                is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
                                                segments. By default, no segmentation occurs.""")
    segmentGroups = InputTypes.makeEnumType('segmentGroup', 'sesgmentGroupType', ['segment', 'cluster', 'interpolate'])
    segment.addParam('grouping', segmentGroups, descr=r"""enables the use of ROM subspace clustering in
        addition to segmenting if set to \xmlString{cluster}. If set to \xmlString{segment}, then performs
        segmentation without clustering. If clustering, then an additional node needs to be included in the
        \xmlNode{Segment} node.""", default='segment')
    subspace = InputData.parameterInputFactory('subspace', contentType=InputTypes.StringType, descr=r"""designates the subspace to divide. This
        should be the pivot parameter (often ``time'') for the ROM.""")
    subspace.addParam('divisions', InputTypes.IntegerType, False, descr=r"""as an alternative to
            \xmlAttr{pivotLength}, this attribute can be used to specify how many data points to include in
            each subdivision, rather than use the pivot values. The algorithm will attempt to split the data
            points as equally as possible.""")
    subspace.addParam('pivotLength', InputTypes.FloatType, False, descr=r"""provides the value in the subspace
            that each segment should attempt to represent, independently of how the data is stored. For
            example, if the subspace has hourly resolution, is measured in seconds, and the desired
            segmentation is daily, the \xmlAttr{pivotLength} would be 86400.
            Either this option or \xmlAttr{divisions} must be provided.
            Either this option or \xmlAttr{pivotLength} must be provided.""")
    subspace.addParam('shift', InputTypes.StringType, False, descr=r"""governs the way in which the subspace is
            treated in each segment. By default, the subspace retains its actual values for each segment; for
            example, if each segment is 4 hours long, the first segment starts at time 0, the second at 4
            hours, the third at 8 hours, and so forth. Options to change this behavior are \xmlString{zero}
            and \xmlString{first}. In the case of \xmlString{zero}, each segment restarts the pivot with the
            subspace value as 0, shifting all other values similarly. In the example above, the first segment
            would start at 0, the second at 0, and the third at 0, with each ending at 4 hours. Note that the
            pivot values are restored when the ROM is evaluated. Using \xmlString{first}, each segment
            subspace restarts at the value of the first segment. This is useful in the event subspace 0 is not
            a desirable value.""")
    segment.addSub(subspace)
    spec.addSub(segment)

    return spec

  def __init__(self):
    """
      Constructor.
      @ In, kwargs, dict, options and initialization settings (from XML)
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Segmented ROM'
    self._divisionInstructions = {}    # which parameters are clustered, and how, and how much?
    self._divisionMetrics = None       # requested metrics to apply; if None, implies everything we know about
    self._divisionInfo = {}            # data that should persist across methods
    self._divisionPivotShift = {}      # whether and how to normalize/shift subspaces
    self._indexValues = {}             # original index values, by index
    self.divisions = None              # trained subdomain division information
    # allow some ROM training to happen globally, seperate from individual segment training
    ## see design note for Clusters
    self._romGlobalAdjustments = None  # global ROM settings, provided by the templateROM before clustering

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    # set up segmentation
    # get input specifications from inputParams
    inputSpecs = paramInput.findFirst('Segment')
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
  def setAdditionalParams(self, params):
    """
      Stores (and later passes through) additional parameters to the sub-roms
      @ In, params, dict, parameters to set, dependent on ROM
      @ Out, None
    """
    Collection.setAdditionalParams(self, params)
    for rom in list(self._roms) + [self._templateROM]:
      rom.setAdditionalParams(params)

  def train(self, tdict, skipAssembly=False):
    """
      Trains the SVL and its supporting SVLs. Overwrites base class behavior due to special clustering needs.
      @ In, trainDict, dict, dicitonary with training data
      @ In, skipAssembly, bool, optional, if True then don't assemble objects from assembler (was handled externally, probably)
      @ Out, None
    """
    # read in assembled objects, if any
    if not skipAssembly:
      self.readAssembledObjects()
    # subdivide space
    ## [0] is normal segments, [1] is potentially the odd-shaped last segment
    divisions = self._subdivideDomain(self._divisionInstructions, tdict)
    if divisions[1]:
      # we can't currently handle unequal segments during sampling, it creates nightmarish
      # realization structures (i.e. ragged value and index arrays)
      # for the DataObject to make sense of. For now, error out instead of being inscrutable.
      indices = divisions[1][0]
      pivot = self._templateROM.pivotParameterID
      pivots = [tdict[pivot][0][i] for i in indices]
      delta = pivots[1] - pivots[0]
      self.raiseAnError(RuntimeError, 'Domain was not subdivided into equal segments! ' +
          f'Last segment is from "{pivot}" = {pivots[0]} to {pivots[1]}, with pivotLength = {delta} ' +
          f'and covering {indices[1]-indices[0]} entries!')
    self.divisions = divisions
    self._divisionInfo['delimiters'] = divisions[0] + divisions[1]
    # allow ROM to handle some global training
    self._romGlobalAdjustments, newTrainingDict = self._templateROM.getGlobalRomSegmentSettings(tdict, divisions)
    # train segments
    self._trainBySegments(divisions, newTrainingDict)
    self.amITrained = True
    self._templateROM.amITrained = True

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
    result = self._templateROM.finalizeGlobalRomSegmentEvaluation(self._romGlobalAdjustments, result, weights=None, slicer=None)
    return result

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    rlz = self._writeSegmentsRealization(writeTo)
    writeTo.addRealization(rlz)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    # write global information
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
  #This is the place have debugg file
  def _evaluateBySegments(self, evaluationDict):
    """
      Evaluate ROM by evaluating its segments
      @ In, evaluationDict, dict, realization to evaluate
      @ Out, result, dict, dictionary of results
    """
    # slicing tool; this means grab everything in a dimension. We use it several times.
    allSlice = slice(None, None, None)
    # TODO assuming only subspace is pivot param
    pivotID = self._templateROM.pivotParameterID
    lastEntry = self._divisionInfo['historyLength']
    result = {}
    nextEntry = 0  # index to fill next data set into
    self.raiseADebug('Sampling from {} segments ...'.format(len(self._roms)))
    roms = self._getSequentialRoms()
    for r, rom in enumerate(roms):
      self.raiseADebug('Evaluating ROM segment', r)
      subResults = rom.evaluate(evaluationDict)
      ## DEBUGGING OPTIONS
      # year = getattr(self, 'DEBUGGYEAR', 0)
      #This is the place have debugg file
      # os.system('mv signal_bases.csv year_{}_segment_{}_signals.csv'.format(year,r))
      ## END DEBUGG
      # NOTE the pivot values for subResults will be wrong (shifted) if shifting is used in training
      ## however, we will set the pivotID values all at once after all results are gathered, so it's okay.
      # build "results" structure if not already done -> easier to do once we gather the first sample
      if not result:
        # check if we're working with any ND data
        indexMap = subResults.pop('_indexMap', {})
        # build a list of the dimensional variables (the indexes)
        dimensionVars = set([pivotID])
        # construct a np zeros placeholder for each target, of the approprate shape
        for target, values in subResults.items():
          dims = indexMap.get(target, None)
          # if no ND, then just use the history length
          if dims is None:
            result[target] = np.zeros(lastEntry)
            pivotIndex = 0
          else:
            dimensionVars.update(set(dims))
            # build a tuple of dimensional lengths
            lens = list(values.shape)
            # record the location of the pivot index for future use
            pivotIndex = dims.index(pivotID)
            # final history dimensionality should be nominal ND dims EXCEPT for the
            #    pivot parameter, along which we are concatenating
            lens[pivotIndex] = lastEntry
            result[target] = np.zeros(lens)

      # place subresult into overall result # TODO this assumes consistent history length! True for ARMA at least.
      entries = len(subResults[pivotID])
      # There's a problem here, if using Clustering; the residual shorter-length element at the end might be represented
      #   by a ROM that expects to deliver the full signal.  TODO this should be handled in a better way,
      #   but for now we can truncate the signal to the length needed
      for target, values in subResults.items():
        # skip the pivotID
        if target in dimensionVars or target in ['_indexMap']: #== pivotID:
          continue
        dims = indexMap.get(target, [pivotID])
        pivotIndex = dims.index(pivotID)
        ### TODO are we ND?? We need to make slices carefully to make sure ND targets are treated right
        # check the amount of remaining history we need to collect to fill the history
        endSelector = tuple((slice(nextEntry, None, None) if dim == pivotID else allSlice) for dim in dims)
        distanceToEnd = result[target][endSelector].shape[pivotIndex]
        # "full" refers to the full reconstructed history
        #  -> fullSlice is where in the full history that the pivot values should be for this subrom
        #  -> fullSelector is fullSlice, but expanded to include other non-ND variables
        # Similarly, "sub" refers to the results coming from the subRom
        ## if there's more data coming from the ROM than we need to fill our history, take just what we need
        if distanceToEnd < values.shape[pivotIndex]:
          fullSlice = slice(nextEntry, None, None)
          subSlice = slice(None, distanceToEnd, None)
        ## otherwise, take all of the sub ROM's results
        else:
          fullSlice = slice(nextEntry, nextEntry+entries, None)
          subSlice = allSlice
        ## create the data selectors
        fullSelector = tuple((fullSlice if dim == pivotID else allSlice) for dim in dims)
        subSelector = tuple((subSlice if dim == pivotID else allSlice) for dim in dims)
        # insert the subrom results into the correct place in the final history
        result[target][fullSelector] = values[subSelector]
        # loop back to the next target
      # update next subdomain storage location
      nextEntry += entries
    # place pivot, other index values
    ## TODO this assumes other dimensions are completely synched!!!
    for dim in dimensionVars:
      if dim == pivotID:
        result[pivotID] = self._indexValues[pivotID]
      else:
        result[dim] = subResults[dim]
    # add the indexMap
    if indexMap:
      result['_indexMap'] = indexMap
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
        # only store bounds, not all indices in between -> note that this is INCLUSIVE!
        counter = list((c[0], c[-1]) for c in counter)
        # Note that "segmented" doesn't have "unclustered" since chunks are evenly sized
      elif mode == 'value':
        segmentValue = value # renamed for clarity
        # divide the subspace into segments with roughly the same pivot length (e.g. time length)
        pivot = trainingSet[subspace][0]
        # find where the data passes the requested length, and make dividers
        floor = 0                # where does this pivot segment start?
        nextOne = segmentValue + pivot[0]   # how high should this pivot segment go?
        counter = []
        # TODO speedup; can we do this without looping?
        dt = pivot[1] - pivot[0]
        while floor < dataLen - 1:
          cross = np.searchsorted(pivot, nextOne-0.5*dt) # half dt if for machine precision error
          # if the next crossing point is past the end, put the remainder piece
          ## into the "unclustered" grouping, since it might be very oddly sized
          ## and throw off segmentation (specifically for clustering)
          if cross == len(pivot):
            remaining = pivot[-1] - pivot[floor-1]
            oneLess = pivot[-2] - pivot[floor-1]
            test1 = abs(remaining-segmentValue)/segmentValue < 1e-6
            test2 = abs(oneLess-segmentValue)/segmentValue < 1e-6
            if not (test1 or test2):
              unclustered.append((floor, cross - 1))
              break
            cross = len(pivot)
          # add this segment, only really need to know the first and last index (inclusive)
          counter.append((floor, cross - 1)) # Note: indices are INCLUSIVE
          # update search parameters
          floor = cross
          if floor >= len(pivot):
            break
          nextOne = pivot[floor] + segmentValue

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
      # slicer for data selection
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
      newROM.adjustLocalRomSegment(self._romGlobalAdjustments, picker)
      self.raiseADebug('Training segment', i, picker)
      newROM.train(data)
      roms.append(newROM)
    # format array for future use
    roms = np.array(roms)
    return roms

  def _writeSegmentsRealization(self, writeTo):
    """
      Writes pointwise data about segmentation to a realization.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """

    # realization to add eventually
    rlz = {}
    segmentNames = range(len(self._divisionInfo['delimiters']))
    # pivot for all this stuff is the segment number
    rlz['segment_number'] = np.asarray(segmentNames)
    iS, iE, pS, pE = self._getSegmentData(full=True) # (i)ndex | (p)ivot, (S)tarts | (E)nds
    # start indices
    varName = 'seg_index_start'
    writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
    rlz[varName] = iS
    # end indices
    varName = 'seg_index_end'
    writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
    rlz[varName] = iE
    # pivot start values
    varName = 'seg_{}_start'.format(self._templateROM.pivotParameterID)
    writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
    rlz[varName] = pS
    # pivot end values
    varName = 'seg_{}_end'.format(self._templateROM.pivotParameterID)
    writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
    rlz[varName] = pE
    return rlz

  def _getSegmentData(self, full=None):
    """
      Provides information about segmenting.
      @ In, full, bool, optional, if True then use full representation (not clustered)
      @ Out, indexStarts, np.array, array of indices where segments start
      @ Out, indexEnds, np.array, array of indices where segments end
      @ Out, pivotStarts, np.array, array of VALUES where segments start
      @ Out, pivotEnds, np.array, array of VALUES where segments end
    """
    if full is None:
      full = True # TODO this is just for clustered ....
    pivotID = self._templateROM.pivotParameterID
    pivot = self._indexValues[pivotID]
    indexStarts = np.asarray(list(d[0] for d in self._divisionInfo['delimiters']))
    pivotStarts = np.asarray(list(pivot[i] for i in indexStarts))
    indexEnds = np.asarray(list(d[-1] for d in self._divisionInfo['delimiters']))
    pivotEnds = np.asarray(list(pivot[i] for i in indexEnds))
    return indexStarts, indexEnds, pivotStarts, pivotEnds

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

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""The cluster \xmlNode{Segment} provides an alternative way to build the ROM. When
    this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
    requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
    during the subspace the ROM representation of the signal changes significantly. For example, if the signal
    is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
    segments. By default, no segmentation occurs."""
    segment = spec.popSub('Segment')
    clusterEvalModeEnum = InputTypes.makeEnumType('clusterEvalModeEnum', 'clusterEvalModeType', ['clustered', 'truncated', 'full'])
    segment.addSub(InputData.parameterInputFactory('evalMode', strictMode=True, contentType=clusterEvalModeEnum,
        descr=r"""changes the structure of the samples for Clustered
        Segmented ROMs. Determines how the evaluations are
        represented, as follows:
        \begin{itemize}
          \item \xmlString{full}, reproduce the full signal using representative cluster segments,
          \item \xmlString{truncated}, reproduce a history containing exactly segment from each
            cluster placed back-to-back, with the \xmlNode{pivotParameter} spanning the clustered
            dimension. Note this will almost surely not be the same length as the original signal;
            information about indexing can be found in the ROM's XML metadata.
          \item \xmlString{clustered}, reproduce a N-dimensional object with the variable
            \texttt{\_ROM\_cluster} as one of the indexes for the ROM's sampled variables. Note that
            in order to use the option, the receiving \xmlNode{DataObject} should be of type
            \xmlNode{DataSet} with one of the indices being \texttt{\_ROM\_cluster}.
        \end{itemize}""", default='truncated'))
    segment.addSub(InputData.parameterInputFactory('evaluationClusterChoice', strictMode=True,
        contentType=InputTypes.makeEnumType('choiceGroup', 'choiceGroupType', ['first', 'random', 'centroid']),
        descr=r"""one of \xmlString{first} or
        \xmlString{random}, determines, if \xmlAttr{grouping}$=cluster$, which
        strategy needs to be followed for the evaluation stage. If ``first'', the
        first ROM (representative segmented ROM),in each cluster, is considered to
        be representative of the full space in the cluster (i.e. the evaluation is always performed
        interrogating the first ROM in each cluster); If ``random'', a random ROM, in each cluster,
        is choosen when an evaluation is requested.
        \nb if ``first'' is used, there is \emph{substantial} memory savings when compared to using
        ``random''.""", default='first'))
    ## clusterFeatures
    segment.addSub(InputData.parameterInputFactory('clusterFeatures', contentType=InputTypes.StringListType,
        descr=r"""if clustering then delineates
        the fundamental ROM features that should be considered while clustering. The available features are
        ROM-dependent, and an exception is raised if an unrecognized request is given. See individual ROMs
        for options. \default All ROM-specific options."""))
    ## max cycles (for Interpolated ROMCollection)
    segment.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType,
                                                    descr=r"""maximum number of cycles to run (default no limit)""", default=None))
    ## classifier
    clsfr = InputData.parameterInputFactory('Classifier', strictMode=True, contentType=InputTypes.StringType,
        descr=r"""associates a \xmlNode{PostProcessor}
        defined in the \xmlNode{Models} block to this segmentation. If clustering is enabled (see
        \xmlAttr{grouping} above), then this associated Classifier will be used to cluster the segmented ROM
        subspaces. The attributes \xmlAttr{class}=\xmlString{Models} and
        \xmlAttr{type}=\xmlString{PostProcessor} must be set, and the text of this node is the \xmlAttr{name}
        of the requested Classifier. Note this Classifier must be a valid Classifier; not all PostProcessors
        are suitable. For example, see the DataMining PostProcessor subtype Clustering.""")
    clsfr.addParam('class', InputTypes.StringType, True, descr=r"""\xmlAttr{class}=\xmlString{Models}""")
    clsfr.addParam('type', InputTypes.StringType, True, descr=r"""\xmlAttr{type}=\xmlString{PostProcessor}""")
    segment.addSub(clsfr)
    ## metric
    metric = InputData.parameterInputFactory('Metric', strictMode=True, contentType=InputTypes.StringType,
        descr=r"""Metrics for clustering subdomain ROMs, the text of this node is the \xmlAttr{name}
        of the requested Metric.""")
    metric.addParam('class', InputTypes.StringType, True, descr=r"""\xmlAttr{class}=\xmlString{Metrics}""")
    metric.addParam('type', InputTypes.StringType, True, descr=r"""\xmlAttr{type}=\xmlString{Metric}""")
    segment.addSub(metric)
    segment.addSub(InputData.parameterInputFactory('macroParameter', contentType=InputTypes.StringType,
        descr=r"""pivot parameter for macro steps (e.g. years)"""))
    spec.addSub(segment)
    return spec

  ## Constructors ##
  def __init__(self):
    """
      Constructor.
      @ In, kwargs, dict, options and initialization settings (from XML)
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Clustered ROM'
    self._divisionClassifier = None      # Classifier to cluster subdomain ROMs
    self._metricClassifiers = None       # Metrics for clustering subdomain ROMs
    self._clusterInfo = {}               # contains all the useful clustering results
    self._evaluationMode = None          # evaluations returning full histories or truncated ones?
    self._clusterFeatures = None         # dict of lists, features to cluster on
    self._featureTemplate = '{target}|{metric}|{id}' # created feature ID template
    self._clusterVariableID = '_ROM_Cluster' # name by which clustering dimension shall be known

  def setTemplateROM(self, romInfo):
    """
      Set the ROM that will be used in this grouping
      @ In, romInfo, dict, {'name':romName, 'modelInstance':romInstance}, the information used to set up template ROM
      @ Out, None
    """
    super().setTemplateROM(romInfo)
    # check if ROM has methods to cluster on (errors out if not)
    if not self._templateROM.isClusterable():
      self.raiseAnError(NotImplementedError, 'Requested ROM "{}" does not yet have methods for clustering!'.format(self._romName))

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    segmentNode = paramInput.findFirst('Segment')
    # evaluation mode
    evalModeNode = segmentNode.findFirst('evalMode')
    if evalModeNode is not None:
      self._evaluationMode = evalModeNode.value
    else:
      self.raiseAMessage('No evalMode specified for clustered ROM, so defaulting to "truncated".')
      self._evaluationMode = 'truncated'
    self.raiseADebug('Clustered ROM evaluation mode set to "{}"'.format(self._evaluationMode))

    # how to choose representative cluster: static or random
    evalChoice = segmentNode.findFirst('evaluationClusterChoice')
    if evalChoice is not None:
      self._evaluationChoice = evalChoice.value
    else:
      self.raiseAMessage('No evaluationClusterChoice specified for clustered ROM, so defaulting to "first".')
      self._evaluationChoice = 'first'

    # interpret clusterable parameter requests, if any
    inputRequestsNode = segmentNode.findFirst('clusterFeatures')
    if inputRequestsNode is None:
      userRequests = None
    else:
      inputRequests = inputRequestsNode.value
      userRequests = self._extrapolateRequestedClusterFeatures(inputRequests)
    self._clusterFeatures = self._templateROM.checkRequestedClusterFeatures(userRequests)

  def readAssembledObjects(self):
    """
      Collects the entities from the Assembler as needed.
      Clusters need the classifer to cluster by, as well as any additional clustering metrics
      @ In, None
      @ Out, None
    """
    # get the classifier to use, if any, from the Assembler
    ## this is used to cluster the ROM segments
    classifier = self._assembledObjects.get('Classifier', [[None]*4])[0][3]
    if classifier is not None:
      # Try using the pp directly, not just the uSVE
      classifier = classifier._pp.unSupervisedEngine
    else:
      self.raiseAnError(IOError, 'Clustering was requested, but no <Classifier> provided!')
    self._divisionClassifier = classifier
    self._metricClassifiers = self._assembledObjects.get('Metric', None)

  ## API ##
  def setAdditionalParams(self, params):
    """
      Stores (and later passes through) additional parameters to the sub-roms
      @ In, params, dict, parameters to set, dependent on ROM
      @ Out, None
    """
    for sub in params['paramInput'].subparts:
      if sub.name == 'clusterEvalMode':
        self._evaluationMode = sub.value
        break
    Segments.setAdditionalParams(self, params)

  def evaluate(self, edict):
    """
      Method to evaluate a point or set of points via surrogate.
      Overwritten for special needs in this ROM
      @ In, edict, dict, evaluation dictionary
      @ Out, result, np.array, evaluated points
    """
    if self._evaluationMode == 'full':
      # TODO there's no input-based way to request this mode right now.
      ## It has been manually tested, but needs a regression tests once this is opened up.
      ## Right now consider it as if it wasn't an available feature, cuz it kinda isn't.
      result = Segments.evaluate(self, edict)
    elif self._evaluationMode in ['truncated', 'clustered']:
      # NOTE: the needs of "truncated" and "clustered" are very similar, so they are only
      ## differentiated by a couple small differences in this "elif".
      if self._evaluationMode == 'truncated':
        result, weights = self._createTruncatedEvaluation(edict)
      else:
        result, weights = self._createNDEvaluation(edict)
      clusterStartIndex = 0 # what index does this cluster start on in the truncated signal?
      globalLocalPicker = []
      for r, rom in enumerate(self._roms):
        # "r" is the cluster label
        # find ROM in cluster
        if self._evaluationChoice == 'first':
          clusterIndex = 0
        else:
          clusterIndex = list(self._clusterInfo['map'][r]).index(rom)
        # find ROM in full history
        segmentIndex, _ = self._getSegmentIndexFromClusterIndex(r, self._clusterInfo['labels'], clusterIndex=clusterIndex)
        # make local modifications based on global settings
        delim = self._divisionInfo['delimiters'][segmentIndex]
        #where in the original signal does this cluster-representing segment come from
        globalPicker = slice(delim[0], delim[-1] + 1)
        segmentLen = globalPicker.stop - globalPicker.start
        globalLocalPicker.append(globalPicker)
        # where in the truncated signal does this cluster sit?
        if self._evaluationMode == 'truncated':
          localPicker = slice(clusterStartIndex, clusterStartIndex + segmentLen)
        else:
          localPicker = r #slice(None, None, None)
        # make final local modifications to truncated evaluation
        result = rom.finalizeLocalRomSegmentEvaluation(self._romGlobalAdjustments, result, globalPicker, localPicker)
        # update the cluster start index
        clusterStartIndex += segmentLen
      # make final modifications to full signal based on global settings
      ## for truncated mode, this is trivial.
      ## for clustered mode, this is complicated.
      result = self._templateROM.finalizeGlobalRomSegmentEvaluation(self._romGlobalAdjustments, result, weights=weights, slicer=globalLocalPicker)
    # TODO add cluster multiplicity to "result" as meta to the output
    #if self._evaluationMode == 'clustered':
    #  result['cluster_multiplicity'] = np.asarray([len(x) for c, x in self._clusterInfo['map'].items() if c != 'unclustered'])
    #  result['_indexMap']['cluster_multiplicity'] = np.atleast_1d([self._clusterVariableID])
    return result

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    rlz = self._writeSegmentsRealization(writeTo)
    # modify the segment entries to have the correct length
    ## this is because segmenting doesn't throw out excess bits, but clustering sure does
    correctLength = len(self._clusterInfo['labels'])
    for key, value in rlz.items():
      rlz[key] = value[:correctLength]
    # add some cluster stuff
    # cluster features
    ## both scaled and unscaled
    labels = self._clusterInfo['labels']
    featureNames = sorted(list(self._clusterInfo['features']['unscaled'].keys()))
    for scaling in ['unscaled', 'scaled']:
      for name in featureNames:
        varName = 'ClusterFeature|{}|{}'.format(name, scaling)
        writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
        rlz[varName] = np.asarray(self._clusterInfo['features'][scaling][name])
    varName = 'ClusterLabels'
    writeTo.addVariable(varName, np.array([]), classify='meta', indices=['segment_number'])
    rlz[varName] = np.asarray(labels)

    writeTo.addRealization(rlz)

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    # do everything that Segments do
    Segments.writeXML(self, writeTo, targets, skip)
    # add some stuff specific to clustering
    indStart, indEnd, pivStart, pivEnd = self._getSegmentData()
    main = writeTo.getRoot()
    labels = self._clusterInfo['labels']
    for i, repRom in enumerate(self._roms):
      # find associated node
      modify = xmlUtils.findPath(main, 'SegmentROM[@segment={}]'.format(i))
      # make changes to reflect being a cluster
      modify.tag = 'ClusterROM'
      modify.attrib['cluster'] = modify.attrib.pop('segment')
      modify.append(xmlUtils.newNode('segments_represented',
                                     text=', '.join(str(x) for x in np.arange(len(labels))[labels==i])))
      # delimiters (since we can't really use the pointwise data for this)
      starts = xmlUtils.newNode('indices')
      starts.text = ', '.join(str(x) for x in (indStart[i], indEnd[i]))
      modify.append(starts)
      # TODO pivot values, index delimiters as well?

  def getSegmentRoms(self, full=False):
    """
      Provide list of the segment ROMs depending on how they're desired
      @ In, full, bool, if True then give all roms not just clusters
      @ Out, getSegmentRoms, list, list of roms
    """
    if full:
      return self._getSequentialRoms()
    else:
      return self._roms

  ## Utilities ##
  def _classifyROMs(self, classifier, features, clusterFeatures):
    """
      Classifies the subdomain ROMs.
      @ In, classifier, Models.PostProcessor, classification model to use
      @ In, features, list(str), ordered list of features
      @ In, clusterFeatures, dict, data on which to train classifier
      @ Out, labels, list(int), ordered list of labels corresponding to ROM subdomains
    """
    # "classifier" is a unSupervisedLearning object (e.g. SciKitLearn or similar)
    ## NOTE that it currently CANNOT be the QDataMining object, as that cannot be pickled!
    # update classifier features
    classifier.updateFeatures(features)
    ## version for the unSupervisedLearning object
    classifier.train(clusterFeatures)
    labels = classifier.evaluate(clusterFeatures)
    # assure testable ordering
    labels = mathUtils.orderClusterLabels(labels)
    ## version for QDataMining object, keep for future reference
    #res = classifier.run({'Features': clusterFeatures})
    #labels = res['outputs'][classifier.labelFeature]
    return labels

  def _collectClusteredEvaluations(self, evaluationDict, pivotID):
    """
      Collect evaluations from each clustered ROM and return them.
      @ In, evaluationDict, dict, realization to evaluate
      @ In, pivotID, str, name of pivot variable
      @ Out, result, dict, dictionary of results (arrays of nparrays)
      @ Out, indexMap, dict, example index map from one sample
      @ Out, sampleWeights, np.array, array of cluster weights (normalized)
      @ Out, pivotLen, int, total length of sampled ROMs (for truncated expression)
    """
    result = None       # populate on first sample -> could use defaultdict, but that's more lenient
    sampleWeights = []  # importance of each cluster
    # sample signal, one piece for each segment
    labelMap = self._clusterInfo['labels']
    clusters = sorted(list(set(labelMap)))
    pivotLen = 0
    for cluster in clusters:
      # choose a ROM
      # TODO implement a distribution-based method for representative ROMs
      if self._evaluationChoice == 'first':
        ## option 1: just take the first one
        rom = self._roms[cluster]
      elif self._evaluationChoice == 'random':
        ## option 2: choose randomly
        segmentIndex, clusterIndex = self._getSegmentIndexFromClusterIndex(cluster, labelMap, chooseRandom=True)
        rom = self._clusterInfo['map'][cluster][clusterIndex]
      # evaluate the ROM
      subResults = rom.evaluate(evaluationDict)
      # collect results
      newLen = len(subResults[pivotID])
      pivotLen += newLen
      # if we're getting ND objects, identify indices and target-index dependence
      indexMap = subResults.pop('_indexMap', {})
      allIndices = set([pivotID])
      for target, indices in indexMap.items():
        allIndices.update(indices)
      # populate results storage
      if result is None:
        result = dict((target, subResults[target] if target in allIndices else []) for target in subResults)
        # FIXME the Indices might not be the same for all ROMs; for instance, if cluster by weeks
        # for a year, there's one day left over for its own cluster!
      # populate weights
      sampleWeights.append(np.ones(len(subResults[pivotID])) * len(self._clusterInfo['map'][cluster]))
      for target, values in subResults.items():
        if target in allIndices:
          continue
        result[target].append(values)
    # TODO can we reduce the number of things we return? This is a little ridiculous.
    return result, indexMap, sampleWeights, pivotLen

  def _createTruncatedEvaluation(self, evaluationDict):
    """
      Evaluates truncated representation of ROM
      @ In, evaluationDict, dict, realization to evaluate
      @ Out, result, dict, dictionary of results
      @ Out, sampleWeights, np.array, array of cluster weights (normalized)
    """
    pivotID = self._templateROM.pivotParameterID
    result, indexMap, sampleWeights, pivotLen = self._collectClusteredEvaluations(evaluationDict, pivotID)
    allIndices = set()
    for target, indices in indexMap.items():
      allIndices.update(indices)
    if not allIndices:
      allIndices.update([pivotID])

    # combine histories (we stored each one as a distinct array during collecting)
    for target, values in result.items():
      if target == pivotID:
        continue
      stackIndex = indexMap.get(target, [pivotID]).index(pivotID)
      result[target] = np.concatenate(values, axis=stackIndex)
    # put in the indexes
    for index in allIndices:
      if index == pivotID:
        result[pivotID] = self._indexValues[pivotID][:pivotLen]
      else:
        # NOTE this assumes all the non-pivot dimensions are synchronized between segments!!
        result[index] = indexValues[index]
    if indexMap:
      result['_indexMap'] = indexMap
    # combine history weights
    sampleWeights = np.hstack(sampleWeights)
    sampleWeights /= sum(sampleWeights)
    return result, sampleWeights

  def _createNDEvaluation(self, evaluationDict):
    """
      Evaluates truncated representation of ROM
      @ In, evaluationDict, dict, realization to evaluate
      @ Out, result, dict, dictionary of results
      @ Out, sampleWeights, np.array, array of cluster weights (normalized)
    """
    pivotID = self._templateROM.pivotParameterID
    # collect cluster evaluations
    result, indexMap, sampleWeights, _ = self._collectClusteredEvaluations(evaluationDict, pivotID)
    # create index list for checking against
    allIndices = set()
    for target, indices in indexMap.items():
      allIndices.update(indices)
    # update shapes for sampled variables
    for target, values in result.items():
      if target in allIndices:
        # indices shouldn't be shaped
        continue
      # Update the indexMap.
      ## first dimention is the cluster ID, then others after.
      indexMap[target] = np.asarray([self._clusterVariableID] + list(indexMap.get(target, [pivotID])))
      # convert the list of arrays to a pure array
      # FIXME -> except, you don't get a pure array if the length of each entry isn't the same
      # For example:
      # if you have 3 clusters with length (100, 100, 10), then you get an array of numpy arrays,
      #    shape = (3,),
      # if you have 3 clusters with length (70, 70, 70), then you get a single numpy array
      result[target] = np.asarray(values)
    # store the index map results
    result['_indexMap'] = indexMap
    # also store the unique cluster label values
    ## TODO is this correctly preserving the order???
    labels = self._clusterInfo['labels']
    result[self._clusterVariableID] = np.asarray(range(min(labels), max(labels) + 1))
    # TODO how to handle sampleWeights in this cluster-style formatting?
    ## for now, let's just return them.
    return result, sampleWeights

  def _extrapolateRequestedClusterFeatures(self, requests):
    """
      Extrapolates from user input (or similar) which clustering features should be used.
      @ In, requests, list(str), requests from input as [featureSet.feature] <or> [featureSet]
      @ Out, reqFeatures, dict(list), by featureSet which cluster features should be used
    """
    # extrapolate which parameters are going to be used in clustering
    reqFeatures = defaultdict(list)
    # this should end up looking like:
    #   reqFeatures {'Fourier': [sin, cos],
    #                'ARMA':    [arparams, maparams, sigma]
    #               } etc
    for feature in requests: # for example, Fourier.arparams, Fourier
      hierarchy = feature.split('.')
      featureSet = hierarchy[0].lower()
      if len(hierarchy) == 1:
        # we want the whole feature set
        reqFeatures[featureSet] = 'all'
      else:
        subFeat = hierarchy[1].lower()
        if reqFeatures[featureSet] == 'all':
          # this is redundant, we already are doing "all", so ignore the request
          continue
        reqFeatures[featureSet].append(subFeat)
    return reqFeatures

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
      clusterIndex = randomUtils.randomIntegers(0, len(eligible) - 1, self)
    # global index
    segmentIndex = eligible[clusterIndex]
    return segmentIndex, clusterIndex

  def _gatherClusterFeatures(self, roms, counter, clusterParams=None):
    """
      Collects features of the ROMs for clustering purposes
      @ In, roms, list, list of segmented SVL ROMs
      @ In, counter, list(tuple), instructions for dividing subspace into subdomains
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
      # get ROM-specific metrics
      romData = rom.getLocalRomClusterFeatures(self._featureTemplate, self._romGlobalAdjustments, self._clusterFeatures, picker=picker)

      for feature, val in romData.items():
        clusterFeatures[feature].append(val)
    return clusterFeatures

  def _getSequentialRoms(self):
    """
      Returns ROMs in sequential order.
      @ In, None
      @ Out, _getSequentialRoms, list of ROMs in order (pointer, not copy)
    """
    # Always returns the first cluster currently. Could be done differently.
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
    self._clusterSegments(roms, divisions)
    # if there were some segments that won't compare well (e.g. leftovers), handle those separately
    if len(remainder):
      unclusteredROMs = self._trainSubdomainROMs(self._templateROM, remainder, trainingSet, self._romGlobalAdjustments)
    else:
      unclusteredROMs = []
    ## unclustered
    self._clusterInfo['map']['unclustered'] = unclusteredROMs

  def _clusterSegments(self, roms, divisions):
    """
      Perform clustering for segment ROMs
      @ In, roms, list, list of ROMs to cluster
      @ In, divisions, tuple, segmentation information
      @ Out, None
    """
    counter, remainder = divisions
    # collect ROM features (basic stats, etc)
    clusterFeatures = self._gatherClusterFeatures(roms, counter)
    # future: requested metrics
    ## TODO someday
    # store clustering info, unweighted
    self._clusterInfo['features'] = {'unscaled': copy.deepcopy(clusterFeatures)}
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
    self._clusterInfo['features']['scaled'] = copy.deepcopy(clusterFeatures)
    # perform clustering
    labels = self._classifyROMs(self._divisionClassifier, features, clusterFeatures)
    uniqueLabels = sorted(list(set(labels))) # note: keep these ordered! Many things hinge on this.
    self.raiseAMessage('Identified {} clusters while training clustered ROM "{}".'.format(len(uniqueLabels), self._romName))
    # make cluster information dict
    self._clusterInfo['labels'] = labels
    ## clustered
    if self._evaluationChoice == 'first':
      # save memory!
      romMapping = dict((label, roms[labels == label]) for label in uniqueLabels)
      allIndices = np.arange(0, len(roms))
      self._clusterInfo['map'] = dict((label, allIndices[labels == label]) for label in uniqueLabels)
      self._roms = list(romMapping[label][0] for label in uniqueLabels)
    elif self._evaluationChoice == 'random':
      # save options!
      self._clusterInfo['map'] = dict((label, roms[labels==label]) for label in uniqueLabels)
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
    for f, feature in enumerate(features):
      clusterFeatures[feature] = clusterFeatures[feature] * weights[f]
    return clusterFeatures

  def _getSegmentData(self, full=None):
    """
      Provides information about segmenting, extended from base class.
      @ In, full, bool, optional, if True then use full representation (not clustered)
      @ Out, indexStarts, np.array, array of indices where segments start
      @ Out, indexEnds, np.array, array of indices where segments end
      @ Out, pivotStarts, np.array, array of VALUES where segments start
      @ Out, pivotEnds, np.array, array of VALUES where segments end
    """
    # default to the same segment data as the evaluation mode.
    if full is None:
      full = self._evaluationMode == 'full'
    # if full segmentation data, then Segments knows how to do that.
    if full:
      return Segments._getSegmentData(self, full=True)
    # otherwise, return the indices corresponding to the clustered roms
    roms = self.getSegmentRoms(full=False)
    pivotID = self._templateROM.pivotParameterID
    indexEdges = np.zeros(len(roms)+1, dtype=int)
    pivotEdges = np.zeros(len(roms)+1, dtype=int)
    pivotVals = self._indexValues[pivotID]
    pivotIndex = 0
    for r, rom in enumerate(roms):
      indexEdges[r] = pivotIndex
      pivotEdges[r] = pivotVals[pivotIndex]
      pivotIndex += len(rom.pivotParameterValues)
    indexEdges[-1] = pivotIndex
    pivotEdges[-1] = pivotVals[pivotIndex if pivotIndex < len(pivotVals) else -1]
    return indexEdges[:-1], indexEdges[1:], pivotEdges[:-1], pivotEdges[1:]









#
#
#
#
class Interpolated(SupervisedLearning):
  """ In addition to clusters for each history, interpolates between histories. """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""Provides an alternative way to build the ROM. In addition to clusters for each history, interpolates between histories."""
    # segmenting and clustering
    segment = InputData.parameterInputFactory("Segment", strictMode=True,
                                              descr=r"""provides an alternative way to build the ROM. When
                                                this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
                                                requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
                                                during the subspace the ROM representation of the signal changes significantly. For example, if the signal
                                                is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
                                                segments. By default, no segmentation occurs.""")
    segmentGroups = InputTypes.makeEnumType('segmentGroup', 'sesgmentGroupType', ['segment', 'cluster', 'interpolate'])
    segment.addParam('grouping', segmentGroups, descr=r"""enables the use of ROM subspace clustering in
        addition to segmenting if set to \xmlString{cluster}. If set to \xmlString{segment}, then performs
        segmentation without clustering. If clustering, then an additional node needs to be included in the
        \xmlNode{Segment} node.""", default='segment')
    subspace = InputData.parameterInputFactory('subspace', contentType=InputTypes.StringType, descr=r"""designates the subspace to divide. This
        should be the pivot parameter (often ``time'') for the ROM.""")
    subspace.addParam('divisions', InputTypes.IntegerType, False, descr=r"""as an alternative to
            \xmlAttr{pivotLength}, this attribute can be used to specify how many data points to include in
            each subdivision, rather than use the pivot values. The algorithm will attempt to split the data
            points as equally as possible.""")
    subspace.addParam('pivotLength', InputTypes.FloatType, False, descr=r"""provides the value in the subspace
            that each segment should attempt to represent, independently of how the data is stored. For
            example, if the subspace has hourly resolution, is measured in seconds, and the desired
            segmentation is daily, the \xmlAttr{pivotLength} would be 86400.
            Either this option or \xmlAttr{divisions} must be provided.
            Either this option or \xmlAttr{pivotLength} must be provided.""")
    subspace.addParam('shift', InputTypes.StringType, False, descr=r"""governs the way in which the subspace is
            treated in each segment. By default, the subspace retains its actual values for each segment; for
            example, if each segment is 4 hours long, the first segment starts at time 0, the second at 4
            hours, the third at 8 hours, and so forth. Options to change this behavior are \xmlString{zero}
            and \xmlString{first}. In the case of \xmlString{zero}, each segment restarts the pivot with the
            subspace value as 0, shifting all other values similarly. In the example above, the first segment
            would start at 0, the second at 0, and the third at 0, with each ending at 4 hours. Note that the
            pivot values are restored when the ROM is evaluated. Using \xmlString{first}, each segment
            subspace restarts at the value of the first segment. This is useful in the event subspace 0 is not
            a desirable value.""")
    segment.addSub(subspace)
    clusterEvalModeEnum = InputTypes.makeEnumType('clusterEvalModeEnum', 'clusterEvalModeType', ['clustered', 'truncated', 'full'])
    segment.addSub(InputData.parameterInputFactory('evalMode', strictMode=True, contentType=clusterEvalModeEnum,
        descr=r"""changes the structure of the samples for Clustered
        Segmented ROMs. Determines how the evaluations are
        represented, as follows:
        \begin{itemize}
          \item \xmlString{full}, reproduce the full signal using representative cluster segments,
          \item \xmlString{truncated}, reproduce a history containing exactly segment from each
            cluster placed back-to-back, with the \xmlNode{pivotParameter} spanning the clustered
            dimension. Note this will almost surely not be the same length as the original signal;
            information about indexing can be found in the ROM's XML metadata.
          \item \xmlString{clustered}, reproduce a N-dimensional object with the variable
            \texttt{\_ROM\_cluster} as one of the indexes for the ROM's sampled variables. Note that
            in order to use the option, the receiving \xmlNode{DataObject} should be of type
            \xmlNode{DataSet} with one of the indices being \texttt{\_ROM\_cluster}.
        \end{itemize}""", default='truncated'))
    segment.addSub(InputData.parameterInputFactory('evaluationClusterChoice', strictMode=True,
        contentType=InputTypes.makeEnumType('choiceGroup', 'choiceGroupType', ['first', 'random', 'centroid']),
        descr=r"""one of \xmlString{first} or
        \xmlString{random}, determines, if \xmlAttr{grouping}$=cluster$, which
        strategy needs to be followed for the evaluation stage. If ``first'', the
        first ROM (representative segmented ROM),in each cluster, is considered to
        be representative of the full space in the cluster (i.e. the evaluation is always performed
        interrogating the first ROM in each cluster); If ``random'', a random ROM, in each cluster,
        is choosen when an evaluation is requested.
        \nb if ``first'' is used, there is \emph{substantial} memory savings when compared to using
        ``random''.""", default='first'))
    ## clusterFeatures
    segment.addSub(InputData.parameterInputFactory('clusterFeatures', contentType=InputTypes.StringListType,
        descr=r"""if clustering then delineates
        the fundamental ROM features that should be considered while clustering. The available features are
        ROM-dependent, and an exception is raised if an unrecognized request is given. See individual ROMs
        for options. \default All ROM-specific options."""))
    ## max cycles (for Interpolated ROMCollection)
    segment.addSub(InputData.parameterInputFactory('maxCycles', contentType=InputTypes.IntegerType,
                                                    descr=r"""maximum number of cycles to run (default no limit)""", default=None))
    ## classifier
    clsfr = InputData.parameterInputFactory('Classifier', strictMode=True, contentType=InputTypes.StringType,
        descr=r"""associates a \xmlNode{PostProcessor}
        defined in the \xmlNode{Models} block to this segmentation. If clustering is enabled (see
        \xmlAttr{grouping} above), then this associated Classifier will be used to cluster the segmented ROM
        subspaces. The attributes \xmlAttr{class}=\xmlString{Models} and
        \xmlAttr{type}=\xmlString{PostProcessor} must be set, and the text of this node is the \xmlAttr{name}
        of the requested Classifier. Note this Classifier must be a valid Classifier; not all PostProcessors
        are suitable. For example, see the DataMining PostProcessor subtype Clustering.""")
    clsfr.addParam('class', InputTypes.StringType, True, descr=r"""\xmlAttr{class}=\xmlString{Models}""")
    clsfr.addParam('type', InputTypes.StringType, True, descr=r"""\xmlAttr{type}=\xmlString{PostProcessor}""")
    segment.addSub(clsfr)
    ## metric
    metric = InputData.parameterInputFactory('Metric', strictMode=True, contentType=InputTypes.StringType,
        descr=r"""Metrics for clustering subdomain ROMs, the text of this node is the \xmlAttr{name}
        of the requested Metric.""")
    metric.addParam('class', InputTypes.StringType, True, descr=r"""\xmlAttr{class}=\xmlString{Metrics}""")
    metric.addParam('type', InputTypes.StringType, True, descr=r"""\xmlAttr{type}=\xmlString{Metric}""")
    segment.addSub(metric)
    segment.addSub(InputData.parameterInputFactory('macroParameter', contentType=InputTypes.StringType,
        descr=r"""pivot parameter for macro steps (e.g. years)"""))
    spec.addSub(segment)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, kwargs, dict, initialization options
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Interp. Cluster ROM'
    self._maxCycles = None # maximum number of cycles to run (default no limit)
    self._macroTemplate = Clusters()

  def setTemplateROM(self, romInfo):
    """
      Set the ROM that will be used in this grouping
      @ In, romInfo, dict, {'name':romName, 'modelInstance':romInstance}, the information used to set up template ROM
      @ Out, None
    """
    self._macroTemplate.setTemplateROM(romInfo)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    # notation: "pivotParameter" is for micro-steps (e.g. within-year, with a Clusters ROM representing each year)
    #           "macroParameter" is for macro-steps (e.g. from year to year)
    inputSpecs = paramInput.findFirst('Segment')
    try:
      self._macroParameter = inputSpecs.findFirst('macroParameter').value # pivot parameter for macro steps (e.g. years)
    except AttributeError:
      self.raiseAnError(IOError, '"interpolate" grouping requested but no <macroParameter> provided!')
    maxCycles = inputSpecs.findFirst('maxCycles')
    if maxCycles is not None:
      self._maxCycles = maxCycles.value
      self.raiseAMessage(f'Truncating macro parameter "{self._macroParameter}" to "{self._maxCycles}" successive steps.')
    self._macroSteps = {}                                               # collection of macro steps (e.g. each year)

    self._macroTemplate._handleInput(paramInput)            # example "yearly" SVL engine collection
  # passthrough to template
  def setAdditionalParams(self, params):
    """
      Sets additional parameters, usually when pickling or similar
      @ In, params, dict, params to set
      @ Out, setAdditionalParams, dict, additional params set
    """
    # max cycles
    for sub in params['paramInput'].subparts:
      if sub.name == 'maxCycles':
        self._maxCycles = sub.value
        self.raiseAMessage(f'Truncating macro parameter "{self._macroParameter}" to "{self._maxCycles}" successive step{"s" if self._maxCycles > 1 else ""}.')
        break
    for step, collection in self._macroSteps.items():
      # deepcopy is necessary because clusterEvalMode has to be popped out in collection
      collection.setAdditionalParams(copy.deepcopy(params))
    self._macroTemplate.setAdditionalParams(params)
    return super().setAdditionalParams(params)

  def setAssembledObjects(self, *args, **kwargs):
    """
      Sets up the assembled objects for this class.
      @ In, args, list, list of arguments
      @ In, kwargs, dict, dict of keyword arguments
      @ Out, None
    """
    self._macroTemplate.setAssembledObjects(*args, **kwargs)

  def readAssembledObjects(self):
    """
      Reads in assembled objects
      @ In, None
      @ Out, None
    """
    for step in self._macroSteps.values():
      step.readAssembledObjects()

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    pass # we don't have a good way to write any info right now
    # TODO this may be useful in the future
    # for year, model in self._macroSteps.items():
    #   print('')
    #   print('year indices',year)
    #   iS, iE, pS, pE = model._getSegmentData() # (i)ndex | (p)ivot, (S)tarts | (E)nds
    #   for i in range(len(iS)):
    #     print(i, iS[i], iE[i], pS[i], pE[i])

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    # write global information
    newNode = xmlUtils.StaticXmlElement('InterpolatedMultiyearROM')
    ## macro steps information
    newNode.getRoot().append(xmlUtils.newNode('MacroParameterID', text=self._macroParameter))
    newNode.getRoot().append(xmlUtils.newNode('MacroSteps', text=len(self._macroSteps)))
    newNode.getRoot().append(xmlUtils.newNode('MacroFirstStep', text=min(self._macroSteps)))
    newNode.getRoot().append(xmlUtils.newNode('MacroLastStep', text=max(self._macroSteps)))
    writeTo.getRoot().append(newNode.getRoot())
    # write info about EACH macro step
    main = writeTo.getRoot()
    for macroID, step in self._macroSteps.items():
      newNode = xmlUtils.StaticXmlElement('MacroStepROM', attrib={self._macroParameter: str(macroID)})
      step.writeXML(newNode, targets, skip)
      main.append(newNode.getRoot())

  ############### TRAINING ####################
  def train(self, tdict):
    """
      Trains the SVL and its supporting SVLs etc. Overwrites base class behavior due to
        special clustering and macro-step needs.
      @ In, trainDict, dict, dicitonary with training data
      @ Out, None
    """
    # tdict should have two parameters, the pivotParameter and the macroParameter -> one step per realization
    if self._macroParameter not in tdict:
      self.raiseAnError(IOError, 'The <macroParameter> "{}" was not found in the training DataObject! Training is not possible.'.format(self._macroParameter))
    ## TODO how to handle multiple realizations that aren't progressive, e.g. sites???
    # create each progressive step
    self._macroTemplate.readAssembledObjects()
    for macroID in tdict[self._macroParameter]:
      macroID = macroID[0]
      new = self._copyAssembledModel(self._macroTemplate)
      self._macroSteps[macroID] = new

    # train the existing steps
    for s, step in enumerate(self._macroSteps.values()):
      self.raiseADebug('Training Statepoint Year {} ...'.format(s))
      trainingData = dict((var, [tdict[var][s]]) for var in tdict.keys())
      step.train(trainingData, skipAssembly=True)
    self.raiseADebug('  Statepoints trained ')
    # interpolate missing steps
    self._interpolateSteps(tdict)
    self.amITrained = True

  def _interpolateSteps(self, trainingDict):
    """
      Master method for interpolating missing ROMs for steps
      @ In, trainingDict, dict, training information
      @ Out, None
    """
    # acquire interpolatable information
    exampleModel = list(self._macroSteps.values())[0] # example MACRO model (e.g. example year)
    ### TODO FIXME WORKING
    # the exampleModel has self._divisionInfo, but the macroTemplate does not!
    # HOWEVER, you can't currently retrain the macroTemplate, but the copied newModel
    # interpolated ROMS don't have the divisionInfo! Apparently there's a missing step in
    # here somewhere. Right now we raise an error with the already-trained Classifier,
    # maybe we can just reset that sucker.
    exampleRoms = exampleModel.getSegmentRoms(full=True)
    numSegments = len(exampleModel._clusterInfo['labels'])
    ## TODO can we reduce the number of unique transitions between clusters?
    ## For example, if the clusters look like this for years 1 and 3:
    ##   Year 1: A1 B1 A1 B1 A1 B1 A1 B1
    ##   Year 2: A2 B2 C2 A2 B2 C2 A2 B2
    ## the total transition combinations are: A1-A2, A1-B2, A1-C2, B1-A2, B1-B2, B1-C2
    ## which is 6 interpolations, but doing all of them we do 8 (or more in a full example).
    ## This could speed up the creation of the interpolated clustered ROMs possibly.
    ## Then, whenever we interpolate, we inquire from whom to whom?
    ## Wait, if you have more than 2 statepoints, this will probably not be worth it.
    ##         - rambling thoughts, talbpaul, 2019
    interps = [] # by segment, the interpreter to make new data
    ## NOTE interps[0] is the GLOBAL PARAMS interpolator!!!
    # statepoint years
    statepoints = list(self._macroSteps.keys())
    # all years
    allYears = list(range(min(statepoints), max(statepoints)))
    # what years are missing?
    missing = list(y for y in allYears if y not in statepoints)
    # if any years are missing, make the interpolators for each segment of each statepoint ROM
    if missing:
      # interpolate global features
      globalInterp = self._createSVLInterpolater(self._macroSteps, index='global')
      # interpolate each segment
      for segment in range(numSegments):
        interp = self._createSVLInterpolater(self._macroSteps, index=segment)
        # store interpolators, by segment
        interps.append(interp)
    self.raiseADebug('Interpolator trained')
    # interpolate new data
    ## now we have interpolators for every segment, so for each missing segment, we
    ## need to make a new Cluster model and assign its subsequence ROMs (pre-clustering).
    years = list(self._macroSteps.keys())
    models = []
    # TODO assuming integer years! And by years we mean MacroSteps, except we leave it as years right now!
    for y in range(min(years), max(years)):
      # don't replace statepoint years
      if y in years:
        self.raiseADebug('Year {} is a statepoint, so no interpolation needed.'.format(y))
        models.append(self._macroSteps[y])
        continue
      # otherwise, create new instances
      else:
        self.raiseADebug('Interpolating year {}'.format(y))
        newModel = self._interpolateSVL(trainingDict, exampleRoms, exampleModel, self._macroTemplate, numSegments, globalInterp, interps, y)
        models.append(newModel)
        self._macroSteps[y] = newModel

  def _createSVLInterpolater(self, modelDict, index=None):
    """
      Generates an interpolation object for a supervised engine
      @ In, modelDict, dict, models to interpolate
      @ In, index, int, optional, segment under consideration
      @ Out, interp, scipy.interp1d instance, interpolater
    """
    # index is the segment
    interp = {}
    df = None
    for step, model in modelDict.items():
      # step is the macro step, e.g. year
      if index is None:
        raise NotImplementedError
      # if the input model is not clustered (maybe impossible currently?), no segmenting consideration
      elif index == 'global':
        params = model._roms[0].parametrizeGlobalRomFeatures(model._romGlobalAdjustments)
      # otherwise, need to capture segment information as well as the global information
      else:
        params = model.getSegmentRoms(full=True)[index].getFundamentalFeatures(None)
      newDf = pd.DataFrame(params, index=[step])
      if df is None:
        df = newDf
      else:
        df = df.append(newDf)

    df.fillna(0.0) # FIXME is 0 really the best for all signals??
    # create interpolators
    interp['method'] = {}
    for header in params:
      interp['method'][header] = interp1d(df.index.values, df[header].values)
    # DEBUGG tools
    #fname = 'debug_statepoints_{}.pk'.format(index)
    #with open(fname, 'wb') as f:
    #  df.index.name = 'year'
    #  pk.dump(df, f)
    #print('DEBUGG interpolation data has been dumped to', fname)
    # END debugg
    return interp

  def _interpolateSVL(self, trainingDict, exampleRoms, exampleModel, template, N, globalInterp, segmentInterps, index):
    """
      interpolates a single engine for a single macro step (e.g. a single year)
      @ In, trainingDict, dict, dictionary with training data
      @ In, exampleRoms, list, segment roms from an interpolation setpoint year
      @ In, exampleModel, ROMCollection instance, master model from an interpolation setpoint year
      @ In, template, SupervisedLearning instance, template ROM for constructing new ones
      @ In, N, int, number of segments in play
      @ In, globalInterp, scipy.interp1d instance, interpolator for global settings
      @ In, segmentInterps, scipy.interp1d instance, interpolator for local settings
      @ In, index, int, year for which interpolation is being performed
      @ Out, newModel, SupervisedEngine instance, interpolated model
    """
    newModel = copy.deepcopy(exampleModel)
    segmentRoms = [] # FIXME speedup, make it a numpy array from the start
    for segment in range(N):
      params = dict((param, interp(index)) for param, interp in segmentInterps[segment]['method'].items())
      # DEBUGG, leave for future development
      #fname = 'debugg_interp_y{}_s{}.pk'.format(index, segment)
      #with open(fname, 'wb') as f:
      #  print('Dumping interpolated params to', fname)
      #  pk.dump(params, f)
      newRom = copy.deepcopy(exampleRoms[segment])
      inputs = newRom.readFundamentalFeatures(params)
      newRom.setFundamentalFeatures(inputs)
      segmentRoms.append(newRom)

    segmentRoms = np.asarray(segmentRoms)
    # add global params
    params = dict((param, interp(index)) for param, interp in globalInterp['method'].items())
    # DEBUGG, leave for future development
    #with open('debugg_interp_y{}_sglobal.pk'.format(index), 'wb') as f:
    #  pk.dump(params, f)

    # TODO assuming histories!
    pivotID = exampleModel._templateROM.pivotParameterID
    pivotValues = trainingDict[pivotID][0] # FIXME assumes pivot is the same for each year
    params = exampleModel._roms[0].setGlobalRomFeatures(params, pivotValues)
    newModel._romGlobalAdjustments = params
    # finish training by clustering
    newModel._clusterSegments(segmentRoms, exampleModel.divisions)
    newModel.amITrained = True
    return newModel

  def _copyAssembledModel(self, model):
    """
      Makes a copy of assembled model and re-performs assembling
      @ In, model, object, entity to copy
      @ Out, new, object, deepcopy of model
    """
    new = copy.deepcopy(model)
    # because assembled objects are excluded from deepcopy, add them back here
    new.setAssembledObjects({})
    return new

  ############### EVALUATING ####################
  def evaluate(self, edict):
    """
      Evaluate the set of interpolated models
      @ In, edict, dict, dictionary of evaluation parameters
      @ Out, result, dict, result of evaluation
    """
    # can we run SupervisedLearning.evaluate? Should this be an evaluateLocal?
    ## set up the results dict with the correct dimensionality
    ### actually, let's wait for the first sample to come in.
    self.raiseADebug('Evaluating interpolated ROM ...')
    results = None
    ## TODO set up right for ND??
    forcedMax = self._maxCycles if self._maxCycles is not None else np.inf
    numMacro = min(len(self._macroSteps), forcedMax)
    macroIndexValues = []
    for m, (macroStep, model) in enumerate(sorted(self._macroSteps.items(), key=lambda x: x[0])):
      if m + 1 > numMacro:
        break
      # m is an index of the macro step, in order of the macro values (e.g. in order of years)
      # macroStep is the actual macro step value (e.g. the year)
      # model is the ClusterROM instance for this macro step
      macroIndexValues.append(macroStep)
      self.raiseADebug(f' ... evaluating macro step "{macroStep}" ({m+1} / {numMacro})')
      subResult = model.evaluate(edict) # TODO same input for all macro steps? True for ARMA at least...
      indexMap = subResult.get('_indexMap', {})
      # if not set up yet, then frame results structure
      if results is None:
        results = {}
        finalIndexMap = indexMap # in case every rlz doesn't use same order, which would be lame
        pivotID = model._templateROM.pivotParameterID
        indices = set([pivotID, self._macroParameter])
        for indexes in finalIndexMap.values():
          indices.update(set(indexes))
        #pivotVals = subResult[pivotID]
        #numPivot = len(pivotVals)
        for target, values in subResult.items():
          # if an index, just set the values now # FIXME assuming always the same!
          ## FIXME thing is, they're not always the same, we're clustering, so sometimes there's diff num days!
          ## TODO for now, we simply require using a classifier that always has the same number of entries.
          if target in [pivotID, '_indexMap'] or target in indices:
            results[target] = values
          else:
            # TODO there's a strange behavior here where we have nested numpy arrays instead of
            # proper matrices sometimes; maybe it has to be this way for unequal clusters
            # As a result, we use the object dtype, onto which we can place a whole numpy array.
            results[target] = np.zeros([numMacro] + list(values.shape), dtype=object)
      # END setting up results structure, if needed
      # FIXME reshape in case indexMap is not the same as finalIndexMap?
      for target, values in subResult.items():
        if target in [pivotID, '_indexMap'] or target in indices:# indexMap:
          continue
        indexer = tuple([m] + [None]*len(values.shape))
        try:
          results[target][indexer] = values
        except ValueError:
          self.raiseAnError(RuntimeError, 'The shape of the histories along the pivot parameter is not consistent! Try using a clustering classifier that always returns the same number of clusters.')
    results['_indexMap'] = {} #finalIndexMap
    for target, vals in results.items():
      if target not in indices and target not in ['_indexMap']: # TODO get a list of meta vars?
        default = [] if vals.size == 1 else [pivotID]
        results['_indexMap'][target] = [self._macroParameter] + list(finalIndexMap.get(target, default))
    results[self._macroParameter] = macroIndexValues
    return results

  ############### DUMMY ####################
  # dummy methods that are required by SVL and not generally used
  def __confidenceLocal__(self, featureVals):
    """
      This should return an estimation of the quality of the prediction.
      This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
      @ In, featureVals, 2-D numpy array , [n_samples,n_features]
      @ Out, __confidenceLocal__, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    return {}

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    return {}

  # Are private-ish so should not be called directly, so we don't implement them, as they don't fit the collection.
  def __evaluateLocal__(self, featureVals):
    """
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """
    pass

  def _train(self, featureVals, targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    pass
