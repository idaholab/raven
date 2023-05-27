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
  Created on Dec 3, 2022

  @author: Andrea Alfonsi
  Utils module for feature selection
"""

#External Modules------------------------------------------------------------------------------------
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...SupervisedLearning import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

def screenInputParams(support, params, includedParams, addOnKeys = None):
  """
    Method to screen estimator input parameters (removing the ones that have been
    identified to be removed)
    @ In, support, np.array(bool), the support of the parameters to keep
    @ In, includedParams, list, list of parameters part of this search
    @ In, params, InputData.ParameterInput, the original parameters.
    @ In, addOnKeys, list, optional, list of additional keys to remove
    @ Out, vals, dict, dictionary of init input params
  """
  toRemove = [includedParams[idx] for idx in range(len(includedParams)) if not support[idx]]
  if addOnKeys is not None:
    toRemove +=addOnKeys
  vals = {}
  if toRemove:
    for child in params.subparts:
      if isinstance(child.value,list):
        newValues = copy.copy(child.value)
        for el in toRemove:
          if el in child.value:
            newValues.pop(newValues.index(el))
        vals[child.getName()] = newValues
  return vals

def screenAndTrainEstimator(Xreduced, yreduced, estimator, support, params, includedParams, addOnKeys = None):
  """
    Method to screen estimator input parameters (removing the ones that have been
    identified to be removed) and re-train it.
    X and y must have the shape of the "surviving" features
    @ In, estimator, instance, instance of the ROM
    @ In, Xreduced, numpy.array, feature data (nsamples,nfeatures) or (nsamples, nTimeSteps, nfeatures)
    @ In, yreduced, numpy.array, target data (nsamples,nTargets) or (nsamples, nTimeSteps, nTargets)
    @ In, support, np.array(bool), the support of the parameters to keep
    @ In, params, InputData.ParameterInput, the original parameters.
    @ In, includedParams, list, list of parameters part of this search
    @ In, addOnKeys, list, optional, list of additional keys to remove
    @ Out, None
  """
  msg = f"estimator class str(estimator.__class__) is not a SupervisedLearning derived class"
  assert issubclass(estimator.__class__, SupervisedLearning.SupervisedLearning), msg
  vals = screenInputParams(support, params, includedParams, addOnKeys = addOnKeys)
  if vals:
    estimator.paramInput.findNodesAndSetValues(vals)
    estimator._handleInput(estimator.paramInput)
  estimator._train(Xreduced, yreduced)
