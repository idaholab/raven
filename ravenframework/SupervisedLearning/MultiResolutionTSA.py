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
    spec.description = r"""A ROM for characterizing and generating synthetic histories using multi-resolution time
      series analysis. This ROM, like the SyntheticHistory ROM, makes use of the algorithms within the TimeSeriesAnalysis (TSA)
      module here in RAVEN. The available algorithms are discussed in more detail below. They can be categorized as
      characterizing, transforming, or generating algorithms. Given three algorithms in order $A_1$, $A_2$, and $A_3$, the
      algorithms are applied to given signals as $A_1 \rightarrow A_2 \rightarrow A_3$ during the training step if the
      algorithms are capable of characterizing or transforming the signal. In the generating step, so long as an inverse
      operator for each algorithm exists, the algorithms are applied in reverse: $A_3^{-1} \rightarrow A_2^{-1} \rightarrow A_1^{-1}$.
      This is the same process that the SyntheticHistory ROM performs. Where this ROM differs is that it can handle
      multi-resolution decompositions of a signal. That is, some algorithms are capable of characterizing and splitting
      a signal at different timescales to better learn the signal dynamics at those timescales. See the `filterbankdwt`
      below for an example of such an algorithm. The MultiResolutionTSA particularly handles the combination of learned
      characteristics and signal generation from the different decomposition levels. This ROM also requires a SegmentROM
      node of the "decomposition" type (an example is given below for the XML input structure).
      //
      In order to use this Reduced Order Model, the \xmlNode{ROM} attribute \xmlAttr{subType} needs to be
      \xmlString{MultiResolutionTSA}. It must also have a SegmentROM node of \xmlAttr{subType} \xmlString{decomposition}"""
    spec = SyntheticHistory.addTSASpecs(spec)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Multi-Resolution Synthetic History'
    # _globalROM is an instance of the SyntheticHistoryROM:
    #     It handles all TSA algorithms up to and including one that splits signal (or creates decompositions).
    #     In other words, it handles all TSA training and evaluation on the "global" signal before decomposition.
    #     A decomposition TSA algorithms is specified with _multiResolution=True within its class in the TSA module.
    self._globalROM = SyntheticHistory()
    self.decompositionAlgorithm = None
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.

  def getGlobalROM(self):
    """
      Getter for the globalROM (i.e., SyntheticHistory ROM)
      @ In, None
      @ Out, _globalROM, SyntheticHistory ROM, instance of a SyntheticHistory ROM
    """
    return self._globalROM

  def setGlobalROM(self, globalROM):
    """
      Setter for the globalROM (i.e., SyntheticHistory ROM)
      @ In, _globalROM, SyntheticHistory ROM, instance of a SyntheticHistory ROM
      @ Out, None
    """
    self._globalROM = globalROM

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    globalROM = self.getGlobalROM()
    globalROM._handleInput(paramInput)

    # check that there is a multiresolution algorithm
    allAlgorithms = globalROM.getTsaAlgorithms()               # get non-global algorithms
    allAlgorithms.extend(globalROM.getGlobalTsaAlgorithms())   # add all global algorithms
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
    globalROM = self.getGlobalROM()

    # get all trained parameters from final algorithm so far (should be multiresolution transformer)
    trainedParams = list(globalROM.getTsaTrainedParams().items())
    mrAlgo, mrTrainedParams = trainedParams[-1]
    if mrAlgo.name != self.decompositionAlgorithm:
      msg = f'TSA Algorithm {self.decompositionAlgorithm} for multiresolution time series analysis must be '
      msg += 'the *last* algorithm applied before the <Segment> node.'
      self.raiseAnError(IOError, msg)

    # eh, maybe this should live in TSAUser in the future...
    # extract settings used for that last algorithm (should have some sort of "levels")
    numLevels = mrAlgo.getDecompositionLevels()

    # reformat the trained params
    sortedTrainedParams = mrAlgo.sortTrainedParamsByLevels(mrTrainedParams)
    return numLevels, sortedTrainedParams

  def _updateMRTrainedParams(self, params):
    """
      Method to update trained parameter dictionary from this class using imported `params` dictionary
      containing multiple-decomposition-level trainings
      @ In, params, dict, dictionary of trained parameters from previous decomposition levels
      @ Out, None
    """
    # get all trained parameters from final algorithm (should be multiresolution transformer)
    trainedParams = list(self._globalROM.getTsaTrainedParams().items())
    mrAlgo, mrTrainedParams = trainedParams[-1]

    mrAlgo.combineTrainedParamsByLevels(mrTrainedParams, params)


  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    self._globalROM.writeTSAtoXML(writeTo)

  def __evaluateLocal__(self, featureVals):
    """
      Evaluate algorithms for ROM generation
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    rlz = self._globalROM.evaluateTSASequential()
    return rlz


  ### ESSENTIALLY UNUSED ###
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
