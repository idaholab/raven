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
  Base subclass definition for DynamicModeDecomposition ROM (transferred from alfoa in SupervisedLearning)
"""

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from utils import InputData, InputTypes
from BaseClasses import BaseInterface
#Internal Modules End--------------------------------------------------------------------------------

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
        descr=r"""Number of features to select""", default=None))
    spec.addSub(InputData.parameterInputFactory('whichSpace',contentType=InputTypes.StringType,
        descr=r"""Which space to search? Target or Feature (this is temporary till MR #1718)""", default="Feature"))
    spec.addSub(InputData.parameterInputFactory('applyOutputScore',contentType=InputTypes.BoolType,
        descr=r"""use the output scores as additional selection metric""", default="False"))
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
    self.parametersToInclude = None
    self.whichSpace = "feature"
    self.applyOutputScore = False
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
    nodes, notFound = paramInput.findNodesAndExtractValues(['parametersToInclude', 'step','nFeaturesToSelect','whichSpace','applyOutputScore'])
    assert(not notFound)
    self.step = nodes['step']
    self.nFeaturesToSelect = nodes['nFeaturesToSelect']
    self.parametersToInclude = nodes['parametersToInclude']
    self.applyOutputScore = nodes['applyOutputScore']
    self.whichSpace = nodes['whichSpace'].lower()
    if self.step <= 0:
      raise self.raiseAnError(ValueError, '"step" parameter must be > 0' )
    if self.parametersToInclude is None:
      self.raiseAnError(ValueError, '"parametersToInclude" must be present (for now)!' )
    if self.nFeaturesToSelect > len(self.parametersToInclude):
      raise self.raiseAnError(ValueError, '"nFeaturesToSelect" > number of parameters in "parametersToInclude"!' )

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

    # Initialization
    nFeatures = X.shape[-1]
    nTargets = y.shape[-1]
    #FIXME 1718
    nParams = len(self.parametersToInclude)
    #nParams = nFeatures if self.parametersToInclude is None else len(self.parametersToInclude)
    # compute number of steps
    step = int(self.step) if self.step > 1 else int(max(1, self.step * nParams))


    # support and ranking for features
    support_ = np.ones(nParams, dtype=np.bool)
    featuresForRanking = np.arange(nParams)[support_]
    ranking_ = np.ones(nParams, dtype=np.int)
    supportOfSupport_ = np.ones(nFeatures, dtype=np.bool) if self.whichSpace == 'feature' else np.ones(nTargets, dtype=np.bool)
    mask = maskF if self.whichSpace == 'feature' else maskT

    # features to select
    if self.nFeaturesToSelect is None:
      nFeaturesToSelect = nFeatures // 2
      nFeaturesToSelect = nFeaturesToSelect if self.parametersToInclude is None else min(nFeaturesToSelect,nParams)
    else:
      nFeaturesToSelect = self.nFeaturesToSelect

    #if maskT is not None and np.sum(maskT) > 0:
    #  supportT_ = np.ones(nTargets, dtype=np.bool)
    #  rankingT_ = np.ones(nTargets, dtype=np.int)
    #else:
    #  supportT_ = np.zeros(nTargets, dtype=np.bool)
    #  rankingT_ = np.zeros(nTargets, dtype=np.bool)

    # get estimator parameter
    originalParams = self.estimator.paramInput
    #if step_score:
    #  self.scores_ = []
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
        coefs = np.asarray([importances[imp] for imp in importances if imp in self.parametersToInclude])
        if coefs.shape[0] == raminingFeatures:
          coefs = coefs.T

      if coefs is None:
        coefs = np.ones(raminingFeatures)

      # Get ranks
      if coefs.ndim > 1:
        ranks = np.argsort(np.sqrt(coefs).sum(axis=0))
      else:
        ranks = np.argsort(np.sqrt(coefs))

      # for sparse case ranks is matrix
      ranks = np.ravel(ranks)

      # Eliminate the worse features
      threshold = min(step, np.sum(support_) - nFeaturesToSelect)

      # Compute step score on the previous selection iteration
      # because 'estimator' must use features
      # that have not been eliminated yet
      support_[featuresForRanking[ranks][:threshold]] = False
      ranking_[np.logical_not(support_)] += 1

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
