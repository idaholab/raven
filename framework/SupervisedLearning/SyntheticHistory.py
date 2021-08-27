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
  Created on Jan 5, 2020

  @author: talbpaul, wangc
  Originally from ARMA.py, split for modularity
  Uses TimeSeriesAnalysis (TSA) algorithms to train then generate synthetic histories
"""
import numpy as np

from utils import InputData, xmlUtils
from TSA import TSAUser

from .SupervisedLearning import SupervisedLearning

class SyntheticHistory(SupervisedLearning, TSAUser):
  """
    Leverage TSA algorithms to train then generate synthetic signals.
  """
  # class attribute
  ## define the clusterable features for this ROM.
  # _clusterableFeatures = TODO # get from TSA
  @classmethod
  def getInputSpecification(cls):
    """
      Establish input specs for this class.
      @ In, None
      @ Out, spec, InputData.ParameterInput, class for specifying input template
    """
    specs = super().getInputSpecification()
    specs.description = r"""A ROM for characterizing and generating synthetic histories. This ROM makes use of
           a variety of TimeSeriesAnalysis (TSA) algorithms to characterize and generate new
           signals based on training signal sets. """
    specs = cls.addTSASpecs(specs)
    return specs

  ### INHERITED METHODS ###
  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    # general infrastructure
    SupervisedLearning.__init__(self)
    TSAUser.__init__(self)
    self.printTag = 'SyntheticHistoryROM'
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    SupervisedLearning._handleInput(self, paramInput)
    self.readTSAInput(paramInput)

  def __trainLocal__(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
      @ Out, None
    """
    self.raiseADebug('Training...')
    self.trainTSASequential(targetVals)

  def __evaluateLocal__(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    rlz = self.evaluateTSASequential()
    return rlz

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    pass # TODO

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    self.writeTSAtoXML(writeTo)

  ### Segmenting and Clustering ###
  def isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # clustering methods have been added
    return False # TODO

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
