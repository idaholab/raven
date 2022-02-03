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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import mathUtils
from utils import InputData, InputTypes
from BaseClasses import BaseType
#Internal Modules End--------------------------------------------------------------------------------

class RFE(BaseType):
  """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through a
    ``coef_`` attribute or through a ``feature_importances_`` attribute.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    nFeaturesToSelect : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
 

    Attributes
    ----------
    nFeatures_ : int
        The number of selected features.

    support_ : array of shape [nFeatures]
        The mask of selected features.

    ranking_ : array of shape [nFeatures]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, nFeatures=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    See also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
  """
  def __init__(self, estimator, nFeaturesToSelect=None, step=1):
    super().__init__()
    self.estimator = estimator
    self.nFeaturesToSelect = nFeaturesToSelect
    self.step = step

  @property
  def _estimator_type(self):
    return self.estimator.type

  @property
  def classes_(self):
    return self.estimator_.classes_

  def train(self, X, y):
    """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, nFeatures]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
    """
    return self._train(X, y)

  def _train(self, X, y, step_score=None):
    # Parameter step_score controls the calculation of self.scores_
    # step_score is not exposed to users
    # and is used when implementing RFECV
    # self.scores_ will not be calculated when calling _fit through fit

    # Initialization
    nFeatures = X.shape[1]
    if self.nFeaturesToSelect is None:
      nFeaturesToSelect = nFeatures // 2
    else:
      nFeaturesToSelect = self.nFeaturesToSelect

    if 0.0 < self.step < 1.0:
      step = int(max(1, self.step * nFeatures))
    else:
      step = int(self.step)
    if step <= 0:
      raise ValueError("Step must be >0")

    support_ = np.ones(nFeatures, dtype=np.bool)
    ranking_ = np.ones(nFeatures, dtype=np.int)

    if step_score:
        self.scores_ = []

    # Elimination
    while np.sum(support_) > nFeaturesToSelect:
      # Remaining features
      features = np.arange(nFeatures)[support_]

      # Rank the remaining features
      estimator = copy.deepcopy(self.estimator)
      
      print("Fitting estimator with %d features." % np.sum(support_))

      estimator._train(X[:, features], y)
      coefs = None
      # Get coefs
      estimator.featureImportances_
      if hasattr(estimator, 'featureImportances_'):
        coefs = np.abs(estimator.featureImportances_)
      if coefs is None:
        coefs = np.ones(nFeatures)
      
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
      #if step_score:
      #  self.scores_.append(step_score(estimator, features))
      support_[features[ranks][:threshold]] = False
      ranking_[np.logical_not(support_)] += 1

    # Set final attributes
    features = np.arange(nFeatures)[support_]
    self.estimator_ = copy.deepcopy(self.estimator)
    self.estimator_._train(X[:, features], y)

    # Compute step score when only nFeaturesToSelect features left
    #if step_score:
    #  self.scores_.append(step_score(self.estimator_, features))
    self.nFeatures_ = support_.sum()
    self.support_ = support_
    self.ranking_ = ranking_

    return features, support_
