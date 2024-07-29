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
  Created on July 29, 2024

  @author: sotogj
  Class to specifically handle multi-resolution time series analysis training
"""
# internal libraries
from .SupervisedLearning import SupervisedLearning
from .SyntheticHistory import SyntheticHistory
#
#
#
#
class MultiResolutionTSA(SupervisedLearning):
  """ Class to specifically handle multi-resolution time series analysis training and evaluation."""

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
    spec.description = r"""Class to specifically handle multi-resolution time series analysis training and evaluation."""
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
    self.decompositionAlgorithm = None

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    self._globalROM._handleInput(paramInput)
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.

    # check that there is a multiresolution algorithm
    allAlgorithms = self._globalROM._tsaAlgorithms
    allAlgorithms.extend(self._globalROM._tsaGlobalAlgorithms)
    foundMRAalgorithm = False
    for algo in allAlgorithms:
      if algo.canTransform():
        if algo.isMultiResolutionAlgorithm():
          foundMRAalgorithm = True
          self.decompositionAlgorithm = algo.name
          break
    if not foundMRAalgorithm:
      msg = 'The MultiResolutionTSA ROM class requires a TSA algorithm capable of '
      msg += ' multiresolution time series analysis. None were found. Example: FilterBankDWT.'
      self.raiseAnError(IOError, msg)

  def _train(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
      @ Out, None
    """
    self._globalROM.trainTSASequential(targetVals)

  def _getMRTrainedParams(self):
    """
      Get trained params for multi-resolution time series analysis. Returns number of decomposition levels
      and a sorted (or re-indexed) version of the trained params dictionary.
      @ In, None
      @ Out, numLevels, int, number of decomposition levels
      @ Out, sortedTrainedParams, dict, dictionary of trained parameters
    """
    # get all trained parameters from final algorithm (should be multiresolution transformer)
    trainedParams = list(self._globalROM._tsaTrainedParams.items())
    mrAlgo, mrTrainedParams = trainedParams[-1]

    # eh, maybe this should live in TSAUser in the future...
    # extract settings used for that last algorithm (should have some sort of "levels")
    numLevels = mrAlgo._getDecompositionLevels()

    # reformat the trained params
    sortedTrainedParams = mrAlgo._sortTrainedParamsByLevels(mrTrainedParams)
    return numLevels, sortedTrainedParams

  def _updateMRTrainedParams(self, params):
    """
      Method to update trained parameter dictionary from this class using imported `params` dictionary
      containing multiple-decomposition-level trainings
      @ In, params, dict, dictionary of trained parameters from previous decomposition levels
      @ Out, None
    """
    # get all trained parameters from final algorithm (should be multiresolution transformer)
    trainedParams = list(self._globalROM._tsaTrainedParams.items())
    mrAlgo, mrTrainedParams = trainedParams[-1]

    mrAlgo._combineTrainedParamsByLevels(mrTrainedParams, params)


  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """

  def __evaluateLocal__(self, featureVals):
    """
    Evaluate algorithms for ROM generation
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
