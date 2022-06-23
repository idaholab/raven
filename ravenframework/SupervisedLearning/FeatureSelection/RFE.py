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
  Created on May 8, 2018

  @author: alfoa
"""

#External Modules--------------------------------------------------------------------------------
import sys
import copy
import gc
import os, psutil
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from collections import OrderedDict
import itertools
#External Modules End----------------------------------------------------------------------------

#Internal Modules--------------------------------------------------------------------------------
from ...utils import mathUtils
from ...utils import InputData, InputTypes
from ...BaseClasses import BaseInterface
#Internal Modules End----------------------------------------------------------------------------

class RFE(BaseInterface):
  """
    Feature ranking with recursive feature elimination.
    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained through a
    ``feature_importances_``  property.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.
    References
    ----------
    Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
    for cancer classification using support vector machines",
    Mach. Learn., 46(1-3), 389--422, 2002.
  """

  needROM = True # the estimator is needed

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
    spec.addSub(InputData.parameterInputFactory('parametersToInclude',contentType=InputTypes.StringListType,
        descr=r"""List of IDs of features/variables to exclude from the search.""", default=None))
    spec.addSub(InputData.parameterInputFactory('nFeaturesToSelect',contentType=InputTypes.IntegerType,
        descr=r"""Exact Number of features to select""", default=None))
    spec.addSub(InputData.parameterInputFactory('maxNumberFeatures',contentType=InputTypes.IntegerType,
                                                      descr=r"""Maximum Number of features to select, the algorithm will automatically determine the
        feature list to minimize a total score.""", default=None))
    spec.addSub(InputData.parameterInputFactory('searchTol',contentType=InputTypes.FloatType,
                                                      descr=r"""Relative tolerance for serarch! Only if maxNumberFeatures is set""",
                                                      default=1e-4))
    spec.addSub(InputData.parameterInputFactory('applyClusteringFiltering',contentType=InputTypes.BoolType,
                                                      descr=r"""Applying clustering before RFE search?""",
                                                      default=False))
    spec.addSub(InputData.parameterInputFactory('whichSpace',contentType=InputTypes.StringType,
        descr=r"""Which space to search? Target or Feature (this is temporary till MR #1718)""", default="Feature"))
    spec.addSub(InputData.parameterInputFactory('step',contentType=InputTypes.FloatType,
        descr=r"""If greater than or equal to 1, then step corresponds to the (integer) number
                  of features to remove at each iteration. If within (0.0, 1.0), then step
                  corresponds to the percentage (rounded down) of features to remove at
                  each iteration.""", default=1))
    return spec

  def __init__(self):
    super().__init__()
    self.printTag = 'FEATURE SELECTION - RFE'
    self.estimator = None
    self.nFeaturesToSelect = None
    self.maxNumberFeatures = None
    self.searchTol = None
    self.applyClusteringFiltering = False
    self.parametersToInclude = None
    self.whichSpace = "feature"
    self.step = 1

  def setEstimator(self, estimator):
    """
      Set estimator
      @ In, estimator, instance, instance of the ROM
      @ Out, None
    """
    self.estimator = estimator

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['parametersToInclude', 'step','nFeaturesToSelect',
                                                            'whichSpace','maxNumberFeatures','searchTol','applyClusteringFiltering'])
    assert(not notFound)
    self.step = nodes['step']
    self.nFeaturesToSelect = nodes['nFeaturesToSelect']
    self.maxNumberFeatures = nodes['maxNumberFeatures']
    self.searchTol = nodes['searchTol']
    self.parametersToInclude = nodes['parametersToInclude']
    self.whichSpace = nodes['whichSpace'].lower()
    self.applyClusteringFiltering = nodes['applyClusteringFiltering']
    # checks
    if self.parametersToInclude is None:
      self.raiseAnError(ValueError, '"parametersToInclude" must be present (for now)!' )
    if self.nFeaturesToSelect is not None and self.maxNumberFeatures is not None:
      raise self.raiseAnError(ValueError, '"nFeaturesToSelect" and "maxNumberFeatures" have been both set. They are mutually exclusive!' )
    if self.nFeaturesToSelect and self.nFeaturesToSelect > len(self.parametersToInclude):
      raise self.raiseAnError(ValueError, '"nFeaturesToSelect" > number of parameters in "parametersToInclude"!' )
    if self.maxNumberFeatures and self.maxNumberFeatures > len(self.parametersToInclude):
      raise self.raiseAnError(ValueError, '"maxNumberFeatures" > number of parameters in "parametersToInclude"!' )
    self.parametersToInclude = nodes['parametersToInclude']
    if self.step <= 0:
      raise self.raiseAnError(ValueError, '"step" parameter must be > 0' )

  def run(self, features, targets, X, y):
    """
      Run the RFE model and then the underlying estimator
      on the selected features.
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, X, numpy.array, feature data (nsamples,nfeatures) or (nsamples, nTimeSteps, nfeatures)
      @ In, y, numpy.array, target data (nsamples,nTargets) or (nsamples, nTimeSteps, nTargets)
      @ Out, newFeatures or newTargets, list, list of new features/targets
      @ Out, supportOfSupport_, np.array, boolean mask of the selected features
      @ Out, whichSpace, str, which space?
      @ Out, vals, dict, dictionary of new values
    """
    maskFeatures = None
    maskTargets = None
    #if self.parametersToInclude is not None:
    if self.whichSpace == 'feature':
      maskFeatures = [False]*len(features)
    else:
      maskTargets = [False]*len(targets)
    for param in self.parametersToInclude:
      if maskFeatures is not None and param in features:
        maskFeatures[features.index(param)] = True
      if maskTargets is not None and param in targets:
        maskTargets[targets.index(param)] = True
    if maskTargets is not None and np.sum(maskTargets) != len(self.parametersToInclude):
      self.raiseAnError(ValueError, "parameters to include are both in feature and target spaces. Only one space is allowed!")
    if maskFeatures is not None and np.sum(maskFeatures) != len(self.parametersToInclude):
      self.raiseAnError(ValueError, "parameters to include are both in feature and target spaces. Only one space is allowed!")
    return self._train(X, y, features, targets, maskF=maskFeatures, maskT=maskTargets)

  def _train(self, X, y, featuresIds, targetsIds, maskF = None, maskT = None, step_score=None):
    #FIXME: support and ranking for targets is only needed now because
    #       some features (e.g. DMDC state variables) are stored among the targets
    #       This will go away once (and if) MR #1718 (https://github.com/idaholab/raven/pull/1718) gets merged
    #       whatever marked with ""FIXME 1718"" will need to be modified
    process = psutil.Process(os.getpid())
    self.raiseAMessage("STARTING MEMORY (Mb): {}".format(process.memory_info().rss/1e6))
    # Initialization
    nFeatures = X.shape[-1]
    nTargets = y.shape[-1]
    #FIXME 1718
    nParams = len(self.parametersToInclude)
    #nParams = nFeatures if self.parametersToInclude is None else len(self.parametersToInclude)

    # support and ranking for features
    support_ = np.ones(nParams, dtype=np.bool)
    featuresForRanking = np.arange(nParams)[support_]
    ranking_ = np.ones(nParams, dtype=np.int)
    supportOfSupport_ = np.ones(nFeatures, dtype=np.bool) if self.whichSpace == 'feature' else np.ones(nTargets, dtype=np.bool)
    mask = maskF if self.whichSpace == 'feature' else maskT

    # features to select
    nFeaturesToSelect = self.nFeaturesToSelect if self.nFeaturesToSelect is not None else self.maxNumberFeatures
    # if both None ==> nFeatures/2
    if nFeaturesToSelect is None:
      nFeaturesToSelect = nFeatures // 2
    nFeaturesToSelect = nFeaturesToSelect if self.parametersToInclude is None else min(nFeaturesToSelect,nParams)

    # get estimator parameter
    originalParams = self.estimator.paramInput
    
    # clustering appraoch here
    if self.applyClusteringFiltering:
      from scipy.stats import spearmanr, pearsonr
      from scipy.cluster import hierarchy
      from scipy.spatial.distance import squareform
      from collections import defaultdict
            
      if self.whichSpace == 'feature':
        space = X[:, mask] if len(X.shape) < 3 else np.average(X[:, :,mask],axis=0)
      else:
        space = y[:, mask] if len(y.shape) < 3 else  np.average(y[:, :,mask],axis=0)
 
      # compute spearman     
      # we fill nan with 1.0 (so the distance for such variables == 0 (will be discarded)
      corr = np.nan_to_num(spearmanr(space,axis=0).correlation,nan=1.0)
      corr = (corr + corr.T) / 2.
      np.fill_diagonal(corr, 1)
      print(corr.shape)
      with open("corr.csv","w") as fo:
        towrite = ""
        for i in range(corr.shape[0]):
          for j in range(corr.shape[1]):
            towrite+=str(corr[i,j]) +","
          towrite+="\n" 
        fo.write(towrite)
      # We convert the correlation matrix to a distance matrix before performing
      # hierarchical clustering using Ward's linkage.
      distance_matrix = 1. - np.abs(corr)
      dist_linkage = hierarchy.ward(squareform(distance_matrix))
      t = float('{:.3e}'.format(0.000001*np.max(dist_linkage)))
      self.raiseAMessage("Applying hierarchical clustering on feature to eliminate possible collinearities")
      self.raiseAMessage(f"Applying distance clustering tollerance of <{t}>")
      cluster_ids = hierarchy.fcluster(dist_linkage, t, criterion="distance")
      cluster_id_to_feature_ids = defaultdict(list)
      for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
      selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
      self.raiseAMessage(f"Features reduced via clustering (before RFE search) from {len(support_)} to {len(selected_features)}!")
      support_[:] = False
      support_[np.asarray(selected_features)] = True

    # compute number of steps
    setStep = int(self.step) if self.step > 1 else int(max(1, self.step * nParams))
    nSteps = (int(np.sum(support_)) - nFeaturesToSelect)/setStep
    lowerStep = int(nSteps)
    diff = nSteps - lowerStep
    firstStep = int(setStep * (1+diff))
    step = firstStep
    # Elimination
    while np.sum(support_) > nFeaturesToSelect:
      # Remaining features
      supportIndex = 0
      raminingFeatures = int(np.sum(support_))
      featuresForRanking = np.arange(nParams)[support_]
      for idx in range(len(supportOfSupport_)):
        if mask[idx]:
          supportOfSupport_[idx] = support_[supportIndex]
          supportIndex=supportIndex+1
      if self.whichSpace == 'feature':
        features = np.arange(nFeatures)[supportOfSupport_]
        targets = np.arange(nTargets)
      else:
        features = np.arange(nFeatures)
        targets = np.arange(nTargets)[supportOfSupport_]
      # Rank the remaining features
      estimator = copy.deepcopy(self.estimator)
      self.raiseAMessage("Fitting estimator with %d features." % np.sum(support_))
      toRemove = [self.parametersToInclude[idx] for idx in range(nParams) if not support_[idx]]
      vals = {}
      if toRemove:
        for child in originalParams.subparts:
          if isinstance(child.value,list):
            newValues = copy.copy(child.value)
            for el in toRemove:
              if el in child.value:
                newValues.pop(newValues.index(el))
            vals[child.getName()] = newValues
        estimator.paramInput.findNodesAndSetValues(vals)
        estimator._handleInput(estimator.paramInput)
      estimator._train(X[:, features] if len(X.shape) < 3 else X[:, :,features], y[:, targets] if len(y.shape) < 3 else y[:, :,targets])

      # Get coefs
      coefs = None
      if hasattr(estimator, 'featureImportances_'):
        importances = estimator.featureImportances_
         # since we get the importance, highest importance must be kept => we get the inverse of coefs
        coefs = 1./np.asarray([importances[imp] for imp in importances if imp in self.parametersToInclude])
        coefs = np.asarray([importances[imp] for imp in importances if imp in self.parametersToInclude])
        if coefs.shape[0] == raminingFeatures:
          coefs = coefs.T

      if coefs is None:
        coefs = np.ones(raminingFeatures)

      # Get ranks (for sparse case ranks is matrix)
      ranks = np.ravel(np.argsort(np.sqrt(coefs).sum(axis=0)) if coefs.ndim > 1 else np.argsort(np.sqrt(coefs)))

      # Eliminate the worse features
      threshold = min(step, np.sum(support_) - nFeaturesToSelect)
      step = setStep

      # Compute step score on the previous selection iteration
      # because 'estimator' must use features
      # that have not been eliminated yet
      support_[featuresForRanking[ranks][:threshold]] = False
      ranking_[np.logical_not(support_)] += 1
      del estimator
      gc.collect()

    # now we check if maxNumberFeatures is set and in case perform an
    # additional reduction based on score
    # removing the variables one by one

    if self.maxNumberFeatures is not None:
      featuresForRanking = np.arange(nParams)[support_]
      f = None
      if f is None:
        f = np.asarray(self.parametersToInclude)
      self.raiseAMessage("Starting Features are {}".format( " ".join(f[np.asarray(featuresForRanking)]) ))
      threshold = len(featuresForRanking) - 1
      coefs = coefs[:,:-1] if coefs.ndim > 1 else coefs[:-1]
      initialRanks = copy.deepcopy(ranks)
      #######
      # NEW SEARCH
      # in here we perform a best subset search
      actualScore = 0.0
      previousScore = self.searchTol*2
      error = 1.0
      add = 0
      initialNumbOfFeatures = int(np.sum(support_))
      featuresForRanking = np.arange(nParams)[support_]
      originalSupport = copy.copy(support_)
      scorelist = []
      featureList = []
      numbFeatures = []
      bestForNumberOfFeatures = {}
      # this can be time consuming
      for k in range(1,initialNumbOfFeatures + 1):
        #Looping over all possible combinations: from initialNumbOfFeatures choose k
        iteration = 0
        for combo in itertools.combinations(featuresForRanking,k):
          iteration+=1
          support_ = copy.copy(originalSupport)
          support_[featuresForRanking] = False
          support_[np.asarray(combo)] = True
          supportIndex = 0
          for idx in range(len(supportOfSupport_)):
            if mask[idx]:
              supportOfSupport_[idx] = support_[supportIndex]
              supportIndex=supportIndex+1
          if self.whichSpace == 'feature':
            features = np.arange(nFeatures)[supportOfSupport_]
            targets = np.arange(nTargets)
          else:
            features = np.arange(nFeatures)
            targets = np.arange(nTargets)[supportOfSupport_]


          estimator = copy.deepcopy(self.estimator)
          self.raiseAMessage("Iteration {}. Fitting estimator with {} features.".format(iteration,np.sum(support_)))
          toRemove = [self.parametersToInclude[idx] for idx in range(nParams) if not support_[idx]]
          survivors = [self.parametersToInclude[idx] for idx in range(nParams) if support_[idx]]
          vals = {}
          if toRemove:
            for child in originalParams.subparts:
              if isinstance(child.value,list):
                newValues = copy.copy(child.value)
                for el in toRemove:
                  if el in child.value:
                    newValues.pop(newValues.index(el))
                vals[child.getName()] = newValues
            estimator.paramInput.findNodesAndSetValues(vals)
            estimator._handleInput(estimator.paramInput)
          estimator._train(X[:, features] if len(X.shape) < 3 else X[:, :,features], y[:, targets] if len(y.shape) < 3 else y[:, :,targets])

          # evaluate
          score = 0.0
          for samp in range(X.shape[0]):
            evaluated = estimator._evaluateLocal(X[samp:samp+1, features] if len(X.shape) < 3 else np.atleast_2d(X[samp:samp+1, :,features]))
            previousScore = actualScore
            scores = {}
            dividend = 0.
            # stateW = 1/float(len(combo))
            for target in evaluated:
              #if target in ['Electric_Power','Turbine_Pressure']:
              #if target in targetsIds and target not in self.parametersToInclude:
              if target in targetsIds:
                if target not in self.parametersToInclude: # if not state variable, then this target is output variable
                  w = 1/float(len(targets)-1-len(combo)) #1/ny (targets contains the index to Time, state and output)
                else:
                  w = 1/float(len(combo)) # 1/nx, the weight of Haoyu's GA cost function

                tidx = targetsIds.index(target)
                avg = np.average(y[:,tidx] if len(y.shape) < 3 else y[samp,:,tidx])
                std = np.std(y[:,tidx] if len(y.shape) < 3 else y[samp,:,tidx])
                if avg == 0: avg = 1
                if std == 0: std = 1.
                ev = (evaluated[target] - avg)/std
                ref = ((y[:,tidx] if len(y.shape) < 3 else y[samp,:,tidx]) - avg )/std
                s = np.sum(np.square(ref-ev))
                scores[target] = s*w
                score +=  s*w
          self.raiseAMessage("Score for iteration {} is {}".format(iteration,score))
          self.raiseAMessage("Variables are: {}".format(" ".join(survivors)))
          self.raiseAMessage("MEMORY (Mb): {}".format(process.memory_info().rss/1e6))
          del estimator
          gc.collect()
          if f is None:
            f = np.asarray(self.parametersToInclude)
          if k in bestForNumberOfFeatures.keys():
            if bestForNumberOfFeatures[k][0] > score:
              bestForNumberOfFeatures[k] = [score,f[np.asarray(combo)]]
          else:
            bestForNumberOfFeatures[k] = [score,f[np.asarray(combo)]]
          scorelist.append(score)
          featureList.append(combo)
          numbFeatures.append(len(combo))

      idxx = np.argmin(scorelist)
      support_ = copy.copy(originalSupport)
      support_[featuresForRanking] = False
      support_[np.asarray(featureList[idxx])] = True
      for k in bestForNumberOfFeatures:
        self.raiseAMessage(f"Best score for {k} features is {bestForNumberOfFeatures[k][0]} with the following features {bestForNumberOfFeatures[k][1]} ")

    # Set final attributes
    supportIndex = 0
    for idx in range(len(supportOfSupport_)):
      if mask[idx]:
        supportOfSupport_[idx] = support_[supportIndex]
        supportIndex = supportIndex + 1
    if self.whichSpace == 'feature':
      features = np.arange(nFeatures)[supportOfSupport_]
      targets = np.arange(nTargets)
    else:
      features = np.arange(nFeatures)
      targets = np.arange(nTargets)[supportOfSupport_]

    self.estimator_ = copy.deepcopy(self.estimator)
    toRemove = [self.parametersToInclude[idx] for idx in range(nParams) if not support_[idx]]
    vals = {}
    if toRemove:
      for child in originalParams.subparts:
        if isinstance(child.value,list):
          newValues = copy.copy(child.value)
          for el in toRemove:
            if el in child.value:
              newValues.pop(newValues.index(el))
          vals[child.getName()] = newValues
      self.estimator_.paramInput.findNodesAndSetValues(vals)
      self.estimator_._handleInput(self.estimator_.paramInput)
    self.estimator_._train(X[:, features] if len(X.shape) < 3 else X[:, :,features], y[:, targets] if len(y.shape) < 3 else y[:, :,targets])

    # Compute step score when only nFeaturesToSelect features left
    #if step_score:
    #  self.scores_.append(step_score(self.estimator_, features))
    self.nFeatures_ = support_.sum()
    self.support_ = support_
    self.ranking_ = ranking_

    return features if self.whichSpace == 'feature' else targets, supportOfSupport_, self.whichSpace, vals
