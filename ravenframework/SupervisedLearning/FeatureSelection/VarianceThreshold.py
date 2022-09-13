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
  Variance Threshold feature selection from sklearn
"""

#External Modules------------------------------------------------------------------------------------
import sys
import numpy as np
from sklearn.feature_selection import VarianceThreshold as vt
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import mathUtils
from ...utils import InputData, InputTypes
from ...BaseClasses import BaseInterface
#Internal Modules End--------------------------------------------------------------------------------

class VarianceThreshold(BaseInterface):
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
    spec.addSub(InputData.parameterInputFactory('parametersToInclude',contentType=InputTypes.StringListType,
        descr=r"""List of IDs of features/variables to include in the search.""", default=None))
    spec.addSub(InputData.parameterInputFactory('whichSpace',contentType=InputTypes.StringType,
        descr=r"""Which space to search? Target or Feature (this is temporary till MR 1718)""", default="Feature"))
    spec.addSub(InputData.parameterInputFactory('threshold',contentType=InputTypes.FloatType,
        descr=r"""Features with a training-set variance lower than this threshold
                  will be removed. The default is to keep all features with non-zero
                  variance, i.e. remove the features that have the same value in all
                  samples.""", default=0.0))
    return spec

  def __init__(self):
    super().__init__()
    self.parametersToInclude = None
    self.whichSpace = "feature"
    self.threshold = 0.0

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['parametersToInclude', 'threshold','whichSpace'])
    assert(not notFound)
    self.threshold = nodes['threshold']
    self.parametersToInclude = nodes['parametersToInclude']
    self.whichSpace = nodes['whichSpace'].lower()
    if self.step <= 0:
      raise self.raiseAnError(ValueError, '"threshold" parameter must be > 0' )
    if self.parametersToInclude is None:
      self.raiseAnError(ValueError, '"parametersToInclude" must be present (for now)!' )

  def run(self, features, targets, X, y):
    """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, nFeatures]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
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
    # if time dependent, we work on the expected value of the features
    if self.whichSpace == 'feature':
      space = X if len(X.shape) < 3 else X.mean(axis=(1))
    else:
      space = y if len(y.shape) < 3 else y.mean(axis=(1))
    estimator = vt.fit(space)
    estimator.get_support()
