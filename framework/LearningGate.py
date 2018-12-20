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
Created on December 6, 2016

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import inspect
import abc
import copy
import collections
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import mathUtils
from utils import utils
import SupervisedLearning
import Metrics
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------

class supervisedLearningGate(utils.metaclass_insert(abc.ABCMeta,BaseType),MessageHandler.MessageUser):
  """
    This class represents an interface with all the supervised learning algorithms
    It is a utility class needed to hide the discernment between time-dependent and static
    surrogate models
  """
  def __init__(self, ROMclass, messageHandler, **kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object (static or time-dependent)
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, ROMclass, string, the surrogate model type
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag              = 'SupervisedGate'
    self.messageHandler        = messageHandler
    self.initializationOptions = kwargs
    self.amITrained            = False
    self.ROMclass              = ROMclass
    # members for clustered roms
    self._usingRomClustering   = False             # are we using ROM clustering?
    self._romClusterDivisions  = {}                # which parameters do we cluster, and how are they subdivided?
    self._romClusterLengths    = {}                # OR which parameters do we cluster, and how long should each be?
    self._romClusterFeatureTemplate = '{target}|{metric}|{id}' # standardized for consistency
    self._romClusterMetrics    = None              # list of requested metrics to apply (defaults to everything)
    self._romClusterInfo       = {}                # data that should persist across methods
    self._romClusterPivotShift = None              # whether and how to normalize/shift subspaces
    self._romClusterMap        = None              # maps labels to the ROMs that are represented by it

    #the ROM is instanced and initialized
    #if ROM comes from a pickled rom, this gate is just a placeholder and the Targets check doesn't apply
    self.pickled = self.initializationOptions.pop('pickled',False)
    if not self.pickled:
      # check how many targets
      if not 'Target' in self.initializationOptions.keys():
        self.raiseAnError(IOError,'No Targets specified!!!')
    # check if pivotParameter is specified and in case store it
    self.pivotParameterId = self.initializationOptions.get("pivotParameter",'time')
    # return instance of the ROMclass
    modelInstance = SupervisedLearning.returnInstance(ROMclass,self,**self.initializationOptions)
    # check if the model can autonomously handle the time-dependency
    # (if not and time-dep data are passed in, a list of ROMs are constructed)
    self.canHandleDynamicData = modelInstance.isDynamic()
    # is this ROM  time-dependent ?
    self.isADynamicModel = False
    # if it is dynamic and time series are passed in, self.supervisedContainer is not going to be expanded, else it is going to
    self.supervisedContainer = [modelInstance]
    self.historySteps = []

    ### ClusteredRom ###
    self.romName = self.initializationOptions.get('name','unnamed')
    self._usingRomClustering = "Cluster" in self.initializationOptions
    if self._usingRomClustering:
      # first check if ROM known how to be clustered
      clusterMetrics = modelInstance.getRomClusterParams()
      # get node from the input specs
      clusterSpec = self.initializationOptions['paramInput'].findFirst('Cluster')
      for node in clusterSpec.subparts:
        # subspace: defines the space to subdivide and cluster
        if node.name == 'subspace':
          if 'divisions' in node.parameterValues:
            self._romClusterDivisions[node.value] = node.parameterValues['divisions']
          if 'pivotLength' in node.parameterValues:
            self._romClusterLengths[node.value] = node.parameterValues['pivotLength']
            # can't give both)
            if len(self._romClusterDivisions):
              self.raiseAnError(IOError,'Cannot provide both \'pivotLength\' and \'divisions\' for subspace!')
          if 'shift' in node.parameterValues:
            self._romClusterPivotShift = node.parameterValues['shift'].lower()
      # quality checking
      ## either pivot lengths or divisions should have been provided
      if not len(self._romClusterDivisions) and not len(self._romClusterLengths):
        self.raiseAnError(IOError, 'Must provide either \'pivotLength\' or \'divisions\' for subspace!')
      ## subspace shifting should be None, 'zero', or 'first'
      shiftOK = ['zero', 'first']
      if self._romClusterPivotShift not in [None] + shiftOK:
        self.raiseAnError(IOError, 'If used, <subspace> "shift" must be one of {}; got "{}"'.format(shiftOK, self._romClusterPivotShift))

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    # clear input specs, as they should all be read in by now
    ## this isn't a great implementation; we should make paramInput picklable instead!
    self.initializationOptions.pop('paramInput',None)
    for eng in self.supervisedContainer:
      eng.initOptionDict.pop('paramInput',None)
    # capture what is normally pickled
    state = self.__dict__.copy()
    if not self.amITrained:
      supervisedEngineObj = state.pop("supervisedContainer")
      del supervisedEngineObj
    return state

  def __setstate__(self, newstate):
    """
      Initialize the ROM with the data contained in newstate
      @ In, newstate, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(newstate)
    if not self.amITrained:
      # NOTE this will fail if the ROM requires the paramInput spec! Fortunately, you shouldn't pickle untrained.
      modelInstance             = SupervisedLearning.returnInstance(self.ROMclass,self,**self.initializationOptions)
      self.supervisedContainer  = [modelInstance]

  def reset(self):
    """
      This method is aimed to reset the ROM
      @ In, None
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reset()
    self.amITrained = False

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedContainer[-1].returnInitialParameters()
    return paramDict

  def train(self, trainingSet, assembledObjects=None):
    """
      This function train the ROM this gate is linked to. This method is aimed to agnostically understand if a "time-dependent-like" ROM needs to be constructed.
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ In, assembledObjects, dict, optional, objects that the ROM Model has assembled via the Assembler
      @ Out, None
    """
    if type(trainingSet).__name__ not in  'dict':
      self.raiseAnError(IOError,"The training set is not a dictionary!")
    if len(trainingSet.keys()) == 0:
      self.raiseAnError(IOError,"The training set is empty!")
    if assembledObjects is None:
      assembledObjects = {}

    # if training using clustering, special treatment
    if self._usingRomClustering:
      self._romClassifier = assembledObjects.get('Classifier',[[None]*4])[0][3]
      self._metricClassifiers = assembledObjects.get('Metric',None)
      self._trainByCluster(self._romClassifier, self._romClusterDivisions, self._romClusterLengths, trainingSet, metrics=self._metricClassifiers)
      self.amITrained = True
      return

    # otherwise, traditional training
    ## time-dependent or static ROM?
    if any(type(x).__name__ == 'list' for x in trainingSet.values()):
      # we need to build a "time-dependent" ROM
      self.isADynamicModel = True
      if self.pivotParameterId not in trainingSet.keys():
        self.raiseAnError(IOError,"the pivot parameter "+ self.pivotParameterId +" is not present in the training set. A time-dependent-like ROM cannot be created!")
      if type(trainingSet[self.pivotParameterId]).__name__ != 'list':
        self.raiseAnError(IOError,"the pivot parameter "+ self.pivotParameterId +" is not a list. Are you sure it is part of the output space of the training set?")
      self.historySteps = trainingSet.get(self.pivotParameterId)[-1]
      if len(self.historySteps) == 0:
        self.raiseAnError(IOError,"the training set is empty!")
      # intrinsically time-dependent or does the Gate need to handle it?
      if self.canHandleDynamicData:
        # the ROM is able to manage the time dependency on its own
        self.supervisedContainer[0].train(trainingSet)
      else:
        # we need to construct a chain of ROMs
        # the check on the number of time steps (consistency) is performed inside the historySnapShoots method
        # get the time slices
        newTrainingSet = mathUtils.historySnapShoots(trainingSet, len(self.historySteps))
        assert(type(newTrainingSet).__name__ == 'list')
        # copy the original ROM
        originalROM = self.supervisedContainer[0]
        # start creating and training the time-dep ROMs
        self.supervisedContainer = [] # [copy.deepcopy(originalROM) for _ in range(len(self.historySteps))]
        # train
        for ts in range(len(self.historySteps)):
          self.supervisedContainer.append(copy.deepcopy(originalROM))
          self.supervisedContainer[-1].train(newTrainingSet[ts])
    else:
      #self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      self.supervisedContainer[0].train(trainingSet)
    self.amITrained = True

  ######################
  # CLUSTERING METHODS #
  ######################
  ### TRAINING ###
  def _trainByCluster(self, classifier, clusterParams, clusterLengths, trainingSet, metrics=None):
    """
      Train ROM by training many ROMs depending on the input/index space clustering.
      @ In, classifier, Models.PostProcessor, entity to classify roms
      @ In, clusterParams, dict, dictionary of inputs/indices to cluster on mapped to number of subdivisions to make
      @ In, clusterLengths, dict, dictionary of inputs/indices to cluster on mapped to length of pivot values to include
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ In, metrics, list(Metrics.Metric), optional, metrics with which to compare subdivided data (not from ROM training)
      @ Out, None
    """
    # TODO user option, once we can do other things
    clusterStrategy = 'segments'
    #clusterStrategy = 'continuous'
    #clusterStrategy = 'cluster'

    templateRom = self.supervisedContainer[0]

    if metrics is None:
      metrics = []
    if len(metrics):
      self.raiseAnError(NotImplementedError,'Metrics have not been implemented for training by cluster yet!')

    # subdivide domain
    counter, remainder = self._subdivideDomain(clusterParams,clusterLengths,trainingSet,templateRom.pivotParameterID)
    # store counter info
    numSegments = len(counter)
    self.raiseADebug('Enabling ClusteredROM for "{}":'.format(self.romName))
    self.raiseADebug('Dividing {:^20s} into {:^5d} divisions for clustering.'.format(templateRom.pivotParameterID,
                                                                                     numSegments))
    # perform any distance metrics
    # TODO not implemented yet
    #for metricID in metrics:
    #  name = '_'.join(metricID[:3])
    #  metric = metricID[3]
    #  print('DEBUGG',name,metric)
    #  #value = metric.evaluate

    # evaluate basic metrics and train subdomain roms
    ## START CASE: clusteringStrategy
    self.targetDatas = None # DEBUGG only
    if clusterStrategy == 'segments':
      _, roms = self._trainSubdomainRoms(templateRom, counter, trainingSet, clusterStrategy)

      # TODO common to multiple methods
      # train remainder roms
      if len(remainder):
        _, unclusteredROMs = self._trainSubdomainRoms(templateRom, remainder, trainingSet, clusterStrategy)
        roms = np.hstack([roms, unclusteredROMs])

      self._romClusterMap = dict((i, roms[i]) for i in range(len(roms)))

    elif clusterStrategy == 'continuous':
      # TODO not implemented yet!
      self.raiseAnError(NotImplementedError,'"continuous" strategy not yet implemented!')
      if len(remainder):
        self.raiseADebug('"{}" division(s) are being excluded from clustering consideration.'.format(len(remainder)))

    elif clusterStrategy == 'cluster':
      # TODO started implementing, but some work needs to be done on the Evaluation side before it's ready
      self.raiseAnError(NotImplementedError,'"cluster" strategy not yet implemented!')
      if len(remainder):
        self.raiseADebug('"{}" division(s) are being excluded from clustering consideration.'.format(len(remainder)))

#      clusterFeatureDict, roms = self._trainSubdomainRoms(templateRom, counter, trainingSet, clusterStrategy)
#      # if only segmenting, we're done!
#
#      features = sorted(clusterFeatureDict.keys())
#
#      ## metric heirarchy
#      featureGroups = collections.defaultdict(list)
#      for feature in features:
#        target, metric, ident = feature.split('|',2)
#        # the same might show up for multiple targets
#        if ident not in featureGroups[metric]:
#          featureGroups[metric].append(ident)
#
#      # weight and scale data
#      weightingStrategy = 'uniform' # TODO input from user
#      #weightingStrategy = 'variance'
#      #weightingStrategy = None
#      clusterFeatureDict = self._weightAndScaleClusters(features, featureGroups, clusterFeatureDict, weightingStrategy)
#
#      # cluster ROMs
#      labels = self._classifyROMs(classifier, features, clusterFeatureDict)
#      self.raiseAMessage('Identified "{}" clusters while training clustered ROM "{}"'.format(len(set(labels)),self.romName))
#
#      # train unclustered roms
#      if len(unclustered):
#        _, unclusteredROMs = self._trainSubdomainRoms(templateRom, unclustered, trainingSet, clusterStrategy)
#        labels = np.hstack([labels, [-1]*len(unclusteredROMs)])
#        roms = np.hstack([roms, unclusteredROMs])
#
#      #########
#      # debug #
#      #########
#      # try something
#      import pandas as pd
#      trainDF = pd.DataFrame(clusterFeatureDict)
#      # add labels
#      trainDF['labels'] = labels[labels != -1]
#      trainDF.to_csv('clustering.csv')
#
#      ## plot points, centers by feature pairs
#      if False:
#        self._plotPointsCenters(features,labels,clusterFeatureDict,centers)
#      ## plot signals as clustered
#      if True:
#        self._plotSignalsClustered(labels,roms,self.targetDatas)
#      #############
#      # END debug #
#      #############
#
#      # who's the best prototypical ROM for each cluster?
#      ## for the ARMA, we can pass in the Fourier coefficients along with the AVERAGE RESIDUAL training data
#      ## TODO this also depends on our strategy (segment, continuous, or clustered)
#      self._romClusterMap = dict((label, roms[labels==label]) for label in labels)
#    ## END CASE: clusteringStrategy
#
#  def _classifyROMs(self, classifier, features, clusterFeatureDict):
#    """
#      Classifies the subdomain roms.
#      @ In, classifier, Models.PostProcessor, classification model to use
#      @ In, features, list(str), ordered list of features
#      @ In, clusterFeatureDict, dictionary of data on which to train classifier
#      @ Out, labels, list(int), ordered list of labels corresponding to the ROM subdomains
#    """
#    # actual classifier is the unSupervisedEngine of the QDataMining of the Model
#    ## this is the unSupervisedLearning.SciKitLearn (or other) instance
#    classifier = classifier.interface.unSupervisedEngine
#    # update classifier features
#    classifier.updateFeatures(features)
#    # make the clustering instance
#    classifier.train(clusterFeatureDict)
#    # label the training data
#    labels = classifier.evaluate(clusterFeatureDict)
#    return labels
#
#  def _weightAndScaleClusters(self, features, featureGroups, clusterFeatureDict, weightingStrategy):
#    """
#      Applies normalization and weighting to cluster training features.
#      @ In, features, list(str), ordered list of features
#      @ In, featureGroups, dict, hierarchal structure of requested features
#      @ In, clusterFeatureDict, dict, features mapped to arrays of values (per ROM)
#      @ In, weightingStrategy, str, weighting strategy to use
#      @ Out, clusterFeatureDict, dict, weighted and scaled feature space
#    """
#    # scaling = {} # DEBUGG only
#    weights = np.zeros(len(features))
#    for f,feat in enumerate(features):
#      data = np.array(clusterFeatureDict[feat])
#      loc, scale = mathUtils.normalizationFactors(data, mode='scale')
#      # scaling[feat] = (loc,scale) # DEBUGG only
#      clusterFeatureDict[feat] = (data-loc)/scale
#      # apply weighting
#      _,metric,ID = feat.split('|',2)
#      if weightingStrategy == 'uniform':
#        weight = 1.0 # normalize later / float(len(features))
#      elif weightingStrategy == 'variance':
#        # weight is variance: MORE variance means MORE importance
#        std = np.std(clusterFeatureDict[feat])
#        weight = std
#      else:
#        # groupWeight = 1.0 / float(len(featureGroups))
#        # weight = groupWeight / float(len(featureGroups[metric]))
#        # normalize weights later
#        weight = 1.0 / float(len(featureGroups[metric]))
#      # DEBUGG
#      # apply special weighting
#      if metric == 'Basic' and ID in ['mean','min','max']:
#        weight *= 2
#      # scale training points by weights
#      # TODO do this after normalization # clusterFeatureDict[feat] *= weight
#      weights[f] = weight
#    # normalize weights
#    ## METHOD: sum of weights should be unity
#    scale = np.sum(weights)
#    ## METHOD: by volume, assuming all weights are 1.0 initially before preference
#    # vol = np.product(list(np.max(v) for v in clusterFeatureDict.values()))
#    # print('DEBUGG original volume:',vol)
#    # renormalize the entirety of the space to have the same hypervolume as before weighting
#    # newVolume = np.product(weights)
#    # oldVolume = 1.0 # because we scaled between 0 and 1, this will fail if you don't
#    # scale = (oldVolume/newVolume)**(1.0/float(len(features)))
#    ## END by volume
#    for feature,vals in clusterFeatureDict.items():
#      clusterFeatureDict[feature] = vals * scale
#      v = clusterFeatureDict[feature]
#      print('DEBUGG val range: {:15.15s} {:1.3e} {:1.3e} {:1.3e}'.format(feature,np.min(v),np.average(v),np.max(v)))
#    vol = np.product(list(np.max(v) for v in clusterFeatureDict.values()))
#    print('DEBUGG volume:',vol)
#    return clusterFeatureDict

  def _trainSubdomainRoms(self, templateRom, counter, trainingSet, strategy):
    """
      Trains the ROMs on each clusterable subdomain, and calculates features based on the data, rom
      @ In, templateRom, SupervisedLearning instance, base ROM as a template for training
      @ In, counter, list(tuple), instructions for subdividing domain into subdomains
      @ In, trainingSet, dict, data on which ROMs should be trained
      @ In, strategy, str, clustering strategy (e.g. "segment", "continuous", "cluster")
      @ Out, clusterFeatureDict, dict, clustering information as {feature: [rom values]}
      @ Out, roms, np.array(SubpervisedLearning instances), trained ROMs for each subdomain
    """
    # identify targets that ROM needs to train to
    targets = templateRom.target[:]
    # clear indices from the training list, since they're independents
    pivotID = templateRom.pivotParameterID
    targets.remove(pivotID)
    # stash the pivot parameter values, since we'll lose those while training segments
    self.historySteps = trainingSet[pivotID][0]
    # DEBUGG
    if self.targetDatas is None: # DEBUGG only
      self.targetDatas = [] # DEBUGG only
    # loop over clusters and train data
    clusterFeatureDict = collections.defaultdict(list)
    roms = []
    for i,subdiv in enumerate(counter):
      # slicer for data selection
      picker = slice(subdiv[0], subdiv[-1]+1)
      ## TODO only consider one sample at 0? Should do more for non-ARMA ROMs!
      ##  -> for now, only ARMAs can be used with this method, so we can address this later
      data = dict((var,[copy.deepcopy(trainingSet[var][0][picker])]) for var in trainingSet)
      # renormalize pivot -> usually by shifting values
      # TODO unit test these features
      if self._romClusterPivotShift == 'zero':
        data[templateRom.pivotParameterID][0] -= data[templateRom.pivotParameterID][0][0]
      elif self._romClusterPivotShift == 'first':
        delta = data[templateRom.pivotParameterID][0][0] - trainingSet[var][0][0]
        data[templateRom.pivotParameterID][0] -= delta
      targetData = dict((var,data[var][0]) for var in targets)
      self.targetDatas.append(targetData) # DEBUGG only
      # create a new ROM and train it
      newRom = copy.deepcopy(templateRom)
      self.raiseADebug('Training segment',i,picker)
      newRom.train(data)
      roms.append(newRom)
      # if clustering, evaluate metrics
      if strategy in ['continuous','cluster']:
        ## -> basic metrics (mean, variance, etc)
        basicData = self._evaluateBasicMetrics(targetData)
        for feature, val in basicData.items():
          clusterFeatureDict[feature].append(val)
        ## -> user-provided metrics
        ### TODO self._evaluateMetrics()
        ## -> ROM metrics
        romData = newRom.getRomClusterValues(self._romClusterFeatureTemplate)
        for feature,val in romData.items():
          clusterFeatureDict[feature].append(val)
    # fix rom list type
    roms = np.array(roms)
    return clusterFeatureDict, roms

  def _subdivideDomain(self, clusterParams, clusterLengths, trainingSet, pivotParam):
    """
      Creates markers for subdividing the pivot parameter domain, either based on number of subdivisions
      or on requested pivotValue lengths.
      @ In, clusterParams, dict, dictionary of inputs/indices to cluster on mapped to number of subdivisions to make
      @ In, clusterLengths, dict, dictionary of inputs/indices to cluster on mapped to length of pivot values to include
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ In, pivotParam, str, name of pivot parameter variable on which do subdivide
      @ Out, counter, list(tuple), indices that belong to each division; at minimum (first index, last index)
      @ Out, unclustered, list(tuple), as "counter" but for segments that will not be clustered
    """
    # subdivide domain
    unclustered = [] # data ranges that didn't get clustered because they are particular
    if len(clusterParams):
      # segment by equal spacing
      index, segments = list(clusterParams.items())[0]
      dataLen = len(trainingSet[index][0])
      self._romClusterInfo['historyLength'] = dataLen
      ## TODO assumption: ARMA only trains on a single sample
      counter = np.arange(dataLen)
      counter = np.array_split(counter, segments)
    else:
      # segment by value
      ## TODO assumption: ARMA only trains on a single sample
      pivot = trainingSet[pivotParam][0]
      index, length = list(clusterLengths.items())[0]
      dataLen = len(trainingSet[index][0])
      self._romClusterInfo['historyLength'] = dataLen
      # find where the data passes the requested length and make dividers
      floor = 0
      nextOne = length
      counter = []
      # FIXME this could potentially be slow since it's a loop
      while pivot[floor] < pivot[-1]:
        cross = np.searchsorted(pivot,nextOne)
        if cross == len(pivot):
          unclustered.append((floor,cross-1))
          break
        counter.append((floor,cross-1))
        floor = cross
        nextOne += length
    return counter, unclustered

#  def _evaluateBasicMetrics(self,data):
#    """
#      Evaluates basic statistical data for clustering.
#      For now does mean and std; in the future could leverage BasicStatistics?
#      @ In, data, dict, data to compute metrics for.
#      @ Out, metrics, dict, {feature:value} for features like "<target>_mean" etc
#    """
#    # TODO currently disabled
#    metrics = {}
#    for target,values in data.items():
#      feature = self._romClusterFeatureTemplate.format(target=target, metric='Basic', id='mean')
#      metrics[feature] = np.average(values)
#      feature = self._romClusterFeatureTemplate.format(target=target, metric='Basic', id='std')
#      metrics[feature] = np.std(values)
#      feature = self._romClusterFeatureTemplate.format(target=target, metric='Basic', id='max')
#      metrics[feature] = np.max(values)
#      feature = self._romClusterFeatureTemplate.format(target=target, metric='Basic', id='min')
#      metrics[feature] = np.min(values)
#    return metrics
#
#  def _plotSignalsClustered(self, labels, roms, targetDatas):
#    """
#      Debug tool. Should be removed or relocated when clustered ROMs are fully implemented.
#      Plots the original data, colored by ROM clusters.
#      @ In, labels, list(str), cluster labels corresponding to ROM order
#      @ In, roms, list(SupervisedLearning), trained subset ROMs in the same order as labels
#      @ In, targetDatas, dict, debugging tool
#      @ Out, None
#    """
#    targetDatas = np.array(targetDatas)
#    # TODO remove
#    import matplotlib.pyplot as plt
#    from matplotlib.lines import Line2D
#    fig,ax = plt.subplots()
#    ax.set_title('Clustered (Fourier)')
#    legends = []
#    for label in set(labels):
#      # legend
#      clr = ('C'+str(label)) if label >= 0 else 'k'
#      legends.append(Line2D([0],[0],color=clr))
#      #figS,axS = plt.subplots()
#      #axS.set_title('Compared: Cluster {}'.format(label))
#      mask = labels == label
#      for r in range(sum(mask)):
#        rom = roms[mask][r]
#        target = targetDatas[mask][r]
#        x = rom.pivotParameterValues
#        y = target['Demand']
#        index = list(roms).index(rom)+1
#        ax.plot(x, y, color=clr)
#        ax.plot([x[ 0]]*2, [5000,20000], 'k:')
#        ax.plot([x[-1]]*2, [5000,20000], 'k:')
#        if (index - 1) % 4 == 0:
#          ax.plot([x[ 0]]*2, [5000,20000], 'k-')
#        ax.text(np.average(x),6000,str(index), ha='center')
#        #axS.plot(x - x[0], y, label=str(list(roms).index(rom)+1))
#      #axS.legend(loc=0)
#    ax.legend(legends, list(set(labels)))
#    plt.savefig('clusters.png')
#    plt.show()

  ### EVALUATING ###
  def _evaluateByCluster(self, request, uniqueClusters=False):
    """
      Evaluate this ROM via clustering.
      TODO this should be possible either by abbreviated representation or full representation
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ In, uniqueClusters, bool, optional, if True then only evaluate each cluster once
      @ Out, result, dict, dictionary of results ({target1:np.array,'target2':np.array}).
    """
    # template, for when generic info is needed
    templateRom = self._romClusterMap.values()[0]
    pivotID = templateRom.pivotParameterID
    # evaluation storage
    lastEntry = self._romClusterInfo['historyLength']
    result = None # because we don't know the targets yet, wait until we get the first evaluation back to set this up
    nextEntry = 0 # index to fill next data set in
    # TODO looping directly over labels only works for "segment" strategy!
    labels = range(max(self._romClusterMap.keys())+1)
    self.raiseADebug('sampling from {} clusters'.format(len(labels)))
    for label in labels:
      rom = self._romClusterMap[label]
      # sample each ROM
      subResults = rom.evaluate(request)
      # NOTE the pivotID values for the sub will be shifted if shifting is used here
      #   however, we will set the pivotID all at once after the values are stored.
      # construct results structure if it's not already in place; easier to make it once we have the first sample
      if result is None:
        result = dict((target,np.zeros(lastEntry)) for target in subResults.keys())
      # stitch them together
      # TODO assuming history set shape for data ... true for ARMA
      entries = len(subResults.values()[0])
      for target,values in subResults.items():
        if target == pivotID:
          # directly re-insert the pivot at the end
          continue
        result[target][nextEntry:nextEntry+entries] = values
      nextEntry += entries
    result[pivotID][:] = self.historySteps # [:] allows for a sizing sanity check, maybe should be removed for user's sake
    return result

  ##########################
  # END CLUSTERING METHODS #
  ##########################

  def confidence(self, request):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ Out, confidenceDict, dict, the dictionary where the confidence is stored for each target
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    confidenceDict = {}
    for rom in self.supervisedContainer:
      sliceEvaluation = rom.confidence(request)
      if len(confidenceDict.keys()) == 0:
        confidenceDict.update(sliceEvaluation)
      else:
        for key in confidenceDict.keys():
          confidenceDict[key] = np.append(confidenceDict[key],sliceEvaluation[key])
    return confidenceDict

  def evaluate(self,request):
    """
      Method to perform the evaluation of a point or a set of points through the linked surrogate model
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ Out, resultsDict, dict, dictionary of results ({target1:np.array,'target2':np.array}).
    """
    if self.pickled:
      self.raiseAnError(RuntimeError,'ROM "'+self.initializationOptions['name']+'" has not been loaded yet!  Use an IOStep to load it.')
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    resultsDict = {}
    if self._usingRomClustering:
      resultsDict = self._evaluateByCluster(request)
    else:
      for rom in self.supervisedContainer:
        sliceEvaluation = rom.evaluate(request)
        if len(resultsDict.keys()) == 0:
          resultsDict.update(sliceEvaluation)
        else:
          for key in resultsDict.keys():
            resultsDict[key] = np.append(resultsDict[key],sliceEvaluation[key])
    return resultsDict

  def reseed(self,seed):
    """
      Used to reset the seed of the underlying ROMs.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reseed(seed)

__interfaceDict                         = {}
__interfaceDict['SupervisedGate'      ] = supervisedLearningGate
__base                                  = 'supervisedGate'

def returnInstance(gateType, ROMclass, caller, **kwargs):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the instance to create
    @ In, caller, instance, object that will share its messageHandler instance
    @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
    @ Out, returnInstance, instance, an instance of a ROM
  """
  try:
    return __interfaceDict[gateType](ROMclass, caller.messageHandler,**kwargs)
  except KeyError as e:
    if gateType not in __interfaceDict:
      caller.raiseAnError(NameError,'not known '+__base+' type '+str(gateType))
    else:
      raise e

def returnClass(ROMclass,caller):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the class to retrieve
    @ In, caller, instnace, object that will share its messageHandler instance
    @ Out, returnClass, the class definition of a ROM
  """
  try:
    return __interfaceDict[ROMclass]
  except KeyError:
    caller.raiseAnError(NameError,'not known '+__base+' type '+ROMclass)
