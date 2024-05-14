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
from .SyntheticHistory import SyntheticHistory
# import pickle as pk # TODO remove me!
import os
#
#
#
#
class MultiResolutionTSA(SupervisedLearning):
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
    spec = SyntheticHistory.addTSASpecs(spec)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, kwargs, dict, initialization options
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Multiresolution Synthetic History'
    self._globalROM = SyntheticHistory()
    self._decompParams = {}

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    self._globalROM._handleInput(paramInput)
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.

  def _train(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
      @ Out, None
    """
    self._globalROM.trainTSASequential(targetVals)

  def __evaluateLocal__(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    rlz = self._globalROM.evaluateTSASequential()
    return rlz

  ### ESSENTIALLY UNUSED ###
  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure, since we do not desire normalization in this implementation.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass
