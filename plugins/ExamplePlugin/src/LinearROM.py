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
  Created July 26, 2021
  @author: talbpaul
  Example plugin ROM for mechanics testing
"""

import numpy as np
from scipy import interpolate

from ravenframework.PluginBaseClasses.SupervisedLearningPlugin import SupervisedLearningPlugin
from ravenframework.utils import InputData, InputTypes

class LinearROM(SupervisedLearningPlugin):
  """
    Compares moments from time series analysis to determine differences between signals
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._fit = None # interp1d object

  def _handleInput(self, inp):
    """
      Function to handle the parameter input.
      @ In, inp, ParameterInput, the already-parsed input.
      @ Out, None
    """
    super()._handleInput(inp)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize this object before use in each Step
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(self.features) > 1:
      self.raiseAnError(IOError, 'ExamplePlugin LinearROM expects only 1 feature and 1 target!')

  def _train(self, featureVals, targetVals):
    """
      Perform training on samples in featureVals with responses y.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ In, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
      @ Out, None
    """
    target = self.target
    feature = self.features[0]
    itp = interpolate.interp1d(featureVals[:, 0], targetVals[:, 0])
    self._fit = itp

  def __evaluateLocal__(self, featureVals):
    """
      Evaluates the model based on input features
      @ In, featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals, dict, {feature: [n_samples]}
    """
    targetVals = np.zeros(featureVals.shape[0])
    for v, val in enumerate(featureVals[:, 0]):
      targetVals[v] = self._fit(val)
    return {self.target[0]: targetVals}

