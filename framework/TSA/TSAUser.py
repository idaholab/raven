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
Created on August 3, 2021

@author: talbpaul
"""
import numpy as np

from utils import xmlUtils

from .Factory import factory

class TSAUser:
  """
    Add-on class for inheriting commonly-used TSA algorithms, such as reading in algorithms
  """
  @classmethod
  def addTSASpecs(cls, spec):
    """
      Make input specs for TSA algorithm inclusion.
      @ In, spec, InputData.parameterInput, input specs to which TSA algos should be added
      @ Out, None
    """
    for typ in factory.knownTypes():
      c = factory.returnClass(typ)
      spec.addSub(c.getInputSpecification())

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    # delay import of factory to assure linear imports
    self._tsaAlgoSettings = {}       # initialization settings for each algorithm
    self._tsaTrainedParams = {}      # holds results of training each algorithm
    self._tsaAlgorithms = []         # list and order for TSA algorithms to use
    self.pivotParameterID = None     # string name for time-like pivot parameter # TODO base class?
    self.pivotParameterValues = None # values for the time-like pivot parameter  # TODO base class?

  def readTSAInput(self, spec):
    """
      Read in TSA algorithms
    """
    self.pivotParameterID = spec.findFirst('pivotParameter').value # TODO does a base class do this?
    for sub in spec.subparts:
      if sub.name in factory.knownTypes():
        algo = factory.returnInstance(sub.name)
        self._tsaAlgoSettings[algo] = algo.handleInput(sub)
        self._tsaAlgorithms.append(algo)
        foundTSAType = True
    if foundTSAType is False:
      options = ', '.join(factory.knownTypes())
      # NOTE this assumes that every TSAUser is also an InputUser!
      self.raiseAnError(IOError, f'No known TSA type found in input. Available options are: {options}')
    if self.pivotParameterID not in self.target:
      # NOTE this assumes that every TSAUser is also an InputUser!
      self.raiseAnError(IOError, 'The pivotParameter must be included in the target space.')

  def trainTSASequential(self, targetVals):
    """
      Train TSA algorithms using a sequential removal-and-residual approach.
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], array of time series data
        NOTE: this should be a single history/realization, not an array of realizations
      @ Out, None
    """
    pivotName = self.pivotParameterID
    # NOTE assumption: self.target exists!
    pivotIndex = self.target.index(pivotName)
    # NOTE assumption: only one training signal (otherwise need a [0, ...] for the slicer)
    pivots = targetVals[:, pivotIndex]
    self.pivotParameterValues = pivots[:] # TODO any way to avoid storing these?
    residual = targetVals[:, :] # deep-ish copy, so we don't mod originals
    numAlgo = len(self.tsaAlgorithms)
    for a, algo in enumerate(self.tsaAlgorithms):
      settings = self.algoSettings[algo]
      targets = settings['target']
      indices = tuple(self.target.index(t) for t in targets)
      signal = residual[:, indices].T # using tuple "indices" transposes, so transpose back
      params = algo.characterize(signal, pivots, targets, settings)
      # store characteristics
      self.trainedParams[algo] = params
      # obtain residual; the part of the signal not characterized by this algo
      # workaround: skip the last one, since it's often the ARMA and the residual isn't known for
      #             the ARMA
      if a < numAlgo - 1:
        algoResidual = algo.getResidual(signal, params, pivots, settings)
        residual[0, :, indices] = algoResidual.T # transpose, again because of indices
      # TODO meta store signal, residual?

  def evaluateTSASequential(self):
    """
      Evaluate TSA algorithms using a sequential linear superposition approach
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    pivots = self.pivotParameterValues
    # NOTE assumption: self.target exists!
    result = np.zeros((self.pivotParameterValues.size, len(self.target) - 1)) # -1 is pivot
    for algo in self.tsaAlgorithms[::-1]:
      settings = self.algoSettings[algo]
      targets = settings['target']
      indices = tuple(self.target.index(t) for t in targets)
      params = self.trainedParams[algo]
      if not algo.canGenerate():
        self.raiseAnError(IOError, "This TSA algorithm cannot generate synthetic histories.")
      signal = algo.generate(params, pivots, settings)
      result[:, indices] += signal
    # RAVEN realization construction
    rlz = dict((target, result[:, t]) for t, target in enumerate(self.target) if target != self.pivotParameterID)
    rlz[self.pivotParameterID] = self.pivotParameterValues
    return rlz

  def writeTSAtoXML(self, xml):
    """
      Write properties of TSA algorithms to XML
      @ In, xml, xmlUtils.StaticXmlElement, entity to write to
      @ Out, None
    """
    root = xml.getRoot()
    for algo in self.tsaAlgorithms:
      algoNode = xmlUtils.newNode(algo.name)
      algo.writeXML(algoNode, self.trainedParams[algo])
      root.append(algoNode)
