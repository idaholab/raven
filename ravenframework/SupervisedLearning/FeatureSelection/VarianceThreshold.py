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
  Created on June 04, 2022
  @author: alfoa
  Variance Threshold feature selection insperied by sklearn
"""

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import mathUtils
from ...utils import InputData, InputTypes
from .FeatureSelectionBase import FeatureSelectionBase
#Internal Modules End--------------------------------------------------------------------------------


class VarianceThreshold(FeatureSelectionBase):
  """
    Variance Threshold feature selection from sklearn
    Feature selector that removes all low-variance features.
    This feature selection algorithm looks only at the features (X)
  """
  needROM = False

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
    spec.description = r"""The \xmlString{VarianceThreshold} is a feature selector that removes
    all low-variance features. This feature selection algorithm looks only at the features and not
    the desired outputs. The variance threshold can be set by the user."""
    spec.addSub(
        InputData.parameterInputFactory(
            'threshold',
            contentType=InputTypes.FloatType,
            descr=
            r"""Features with a training-set variance lower than this threshold
                  will be removed. The default is to keep all features with non-zero
                  variance, i.e. remove the features that have the same value in all
                  samples.""",
            default=0.0))
    return spec

  def __init__(self):
    """
      Feature selection class based on variance reduction
      @ In, None
      @ Out, None
    """
    super().__init__()
    # variance threshold
    self.threshold = 0.0

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['threshold'])
    self.threshold = nodes['threshold']
    if self.threshold < 0:
      raise self.raiseAnError(ValueError, '"threshold" parameter must be > 0')

  def _train(self, X, y, featuresIds, targetsIds, maskF=None, maskT=None):
    """
      Train the feature selection model and perform search of best features
      @ In, X, numpy.array, feature data (nsamples,nfeatures) or (nsamples, nTimeSteps, nfeatures)
      @ In, y, numpy.array, target data (nsamples,nTargets) or (nsamples, nTimeSteps, nTargets)
      @ In, featuresIds, list, list of features
      @ In, targetsIds, list, list of targets
      @ In, maskF, optional, np.array, indeces of features to search within
                    (parameters to include None if search is whitin targets)
      @ In, maskT, optional, np.array, indeces of targets to search within
                    (parameters to include None if search is whitin features)
      @ Out, newFeatures or newTargets, list, list of new features/targets
      @ Out, supportOfSupport_, np.array, boolean mask of the selected features
    """
    from sklearn.feature_selection import VarianceThreshold as vt
    # if time dependent, we work on the expected value of the features
    nFeatures = X.shape[-1]
    nTargets = y.shape[-1]

    if self.whichSpace == 'feature':
      space = X[:, maskF] if len(X.shape) < 3 else np.average(X[:, :,maskF],axis=0)
      supportOfSupport_, mask = np.ones(nFeatures,dtype=bool), maskF
    else:
      space = y[:, maskT] if len(y.shape) < 3 else  np.average(y[:, :,maskT],axis=0)
      supportOfSupport_, mask = np.ones(nTargets,dtype=bool), maskT
    # fit estimator
    estimator = vt(threshold=self.threshold).fit(space)
    supportOfSupport_[mask] = estimator.get_support()
    if self.whichSpace == 'feature':
      newVariables = np.arange(nFeatures)[supportOfSupport_]
    else:
      newVariables = np.arange(nTargets)[supportOfSupport_]

    return newVariables, supportOfSupport_
