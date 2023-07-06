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

Contains a utility base class for accessing commonly-used TSA functions.
"""
import numpy as np
import pandas as pd
from inspect import isabstract

from ..utils import xmlUtils, InputData, InputTypes

from .Factory import factory

class TSAUser:
  """
    Add-on class for inheriting commonly-used TSA algorithms, such as reading in algorithms
  """
  @classmethod
  def addTSASpecs(cls, spec, subset=None):
    """
      Make input specs for TSA algorithm inclusion.
      @ In, spec, InputData.parameterInput, input specs to which TSA algos should be added
      @ In, subset, str, optional, one of (None, 'generate', 'characterize'), subset of allowable algorithms
      @ Out, None
    """
    # NOTE we use an assertion here, since only developer actions should be picking subType
    assert subset in (None, 'characterize', 'generate'), f'Invalid subset for TSAUser.addTSASpecs: "{subset}"'
    # need pivot parameter ID, but this might be superceded by a parent class.
    spec.addSub(InputData.parameterInputFactory('pivotParameter', contentType=InputTypes.StringType,
        descr=r"""If a time-dependent ROM is requested, please specifies the pivot
        variable (e.g. time, etc) used in the input HistorySet.""", default='time'))
    for typ in factory.knownTypes():
      c = factory.returnClass(typ)
      if subset == 'characterize' and not c.canCharacterize():
        continue
      elif subset == 'generate' and not c.canGenerate():
        continue
      if isabstract(c):
        # Abstract classes cannot be instantiated, so providing input specs for them is pointless
        continue
      spec.addSub(c.getInputSpecification())
    return spec

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
    self._paramNames = None          # cached list of parameter names
    self._paramRealization = None    # cached dict of param variables mapped to values
    self._tsaTargets = None          # cached list of targets
    self.target = None

  def readTSAInput(self, spec):
    """
      Read in TSA algorithms
      @ In, spec, InputData.parameterInput, input specs filled with user entries
      @ Out, None
    """
    if self.pivotParameterID is None: # might be handled by parent
      self.pivotParameterID = spec.findFirst('pivotParameter').value
    for sub in spec.subparts:
      if sub.name in factory.knownTypes():
        algo = factory.returnInstance(sub.name)
        self._tsaAlgoSettings[algo] = algo.handleInput(sub)
        self._tsaAlgorithms.append(algo)
        foundTSAType = True
    if foundTSAType is False:
      options = ', '.join(factory.knownTypes())
      # NOTE this assumes that every TSAUser is also an InputUser!
      raise IOError(f'TSA: No known TSA type found in input. Available options are: {options}')
    if self.target is None:
      # set up all the expected targets from all the TSAs
      self.target = [self.pivotParameterID] + list(self.getTargets())
    elif self.pivotParameterID not in self.target:
      # NOTE this assumes that every TSAUser is also an InputUser!
      raise IOError('TSA: The pivotParameter must be included in the target space.')

  def canCharacterize(self):
    """
      Checks if any of the algorithms are characterizers
      @ In, None
      @ Out, isCharacterizer, bool, True if this entity is a characterizer
    """
    return any(algo.canCharacterize() for algo in self._tsaAlgorithms)

  def canGenerate(self):
    """
      Checks if any of the algorithms are generators
      @ In, None
      @ Out, isGenerator, bool, True if this entity is a generator
    """
    return any(algo.canGenerate() for algo in self._tsaAlgorithms)

  def _tsaReset(self):
    """
      Resets trained and cached params
      @ In, None
      @ Out, None
    """
    self._tsaTrainedParams = {}      # holds results of training each algorithm
    self._paramNames = None          # cached list of parameter names
    self._paramRealization = None    # cached dict of param variables mapped to values
    self._tsaTargets = None          # cached list of targets

  def getTargets(self):
    """
      Provide ordered target set for the set of algorithms used by this entity
      @ In, None
      @ Out, targets, set, set of targets used among all algorithms
    """
    if self._tsaTargets is None:
      targets = set()
      # if we've trained params, we can use that
      if self._tsaTrainedParams:
        for algo, params in self._tsaTrainedParams.items():
          targets.update(params.keys())
      # otherwise, we use targets from settings
      else:
        for algo, settings in self._tsaAlgoSettings.items():
          targets.update(settings['target'])
      self._tsaTargets = targets
    return self._tsaTargets

  def getCharacterizingVariableNames(self):
    """
      Provide expected training parameters as variable names list
      Note this works even if training has not been performed yet
      @ In, None
      @ Out, names, list, list of parameter names
    """
    if self._paramNames is None:
      if self._paramRealization is not None:
        # if we trained already?,then we can return keys
        self._paramNames = list(self._paramRealization.keys())
      else:
        # otherwise we build the names predictively
        names = []
        for algo in self._tsaAlgorithms:
          names.extend(algo.getParamNames(self._tsaAlgoSettings[algo]))
        self._paramNames = names
    return self._paramNames

  def getParamsAsVars(self):
    """
      Provide training parameters as variable names mapped to values
      Note this is only useful AFTER characterization has been performed/trained
      @ In, None
      @ Out, params, dict, map of {algo_param: value}
    """
    if self._paramRealization is None:
      rlz = {}
      for algo, params in self._tsaTrainedParams.items():
        new = algo.getParamsAsVars(params)
        rlz.update(new)
      self._paramRealization = rlz
    return self._paramRealization

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
    # NOTE assumption: only one training signal
    pivots = targetVals[0, :, pivotIndex]
    self.pivotParameterValues = pivots[:] # TODO any way to avoid storing these?
    ALL_RESIDUALS = pd.DataFrame(index=pivots)
    residual = targetVals[:, :, :] # deep-ish copy, so we don't mod originals
    for a, algo in enumerate(self._tsaAlgorithms):
      settings = self._tsaAlgoSettings[algo]
      targets = settings['target']
      indices = tuple(self.target.index(t) for t in targets)
      signal = residual[0, :, indices].T # using tuple "indices" transposes, so transpose back
      # check if there are missing values in the signal and if algo can accept them
      if np.isnan(signal).any() and not algo.canAcceptMissingValues():
        raise ValueError(f'Missing values (NaN) found in input to {algo.name}, '
                         f'but {algo.name} cannot accept missing values!')
      params = algo.fit(signal, pivots, targets, settings)
      # store characteristics
      self._tsaTrainedParams[algo] = params
      # obtain residual; the part of the signal not characterized by this algo
      # This is only done if the algo produces a residual (is a transformer). Otherwise, the
      # residual signal is not altered.
      if algo.canTransform():
        algoResidual = algo.getResidual(signal, params, pivots, settings)
        residual[0, :, indices] = algoResidual.T # transpose, again because of indices
      # TODO meta store signal, residual?
      for t in targets:
        ALL_RESIDUALS[f'{algo.name}_{t}'] = residual[0, :, self.target.index(t)]
    ALL_RESIDUALS.index = pivots
    ALL_RESIDUALS.to_csv('residuals.csv', index=False)

  def evaluateTSASequential(self):
    """
      Evaluate TSA algorithms using a sequential linear superposition approach
      @ In, None
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    pivots = self.pivotParameterValues
    # the algorithms' targets need to be consistently indexed, but there's no
    # reason to keep including the pivot values every time, so set up an indexing
    # that ignores the pivotParameter on which to index the results variables
    noPivotTargets = [x for x in self.target if x != self.pivotParameterID]
    result = np.zeros((self.pivotParameterValues.size, len(noPivotTargets)))
    ALL_COMPOSITES = pd.DataFrame(index=pivots)
    for algo in self._tsaAlgorithms[::-1]:
      settings = self._tsaAlgoSettings[algo]
      targets = settings['target']
      indices = tuple(noPivotTargets.index(t) for t in targets)
      params = self._tsaTrainedParams[algo]
      signal = result[:, indices]
      if algo.canTransform():  # covers algorithms which are both transformers and generators
        result[:, indices] = algo.getComposite(signal, params, pivots, settings)
      elif algo.canGenerate():
        result[:, indices] = algo.generate(params, pivots, settings)
      else:  # Must be exclusively a TimeSeriesCharacterizer, so there is nothing to evaluate
        continue
      for t in targets:
        ALL_COMPOSITES[f'{algo.name}_{t}'] = result[:, noPivotTargets.index(t)]
    # RAVEN realization construction
    rlz = dict((target, result[:, t]) for t, target in enumerate(noPivotTargets))
    rlz[self.pivotParameterID] = self.pivotParameterValues
    ALL_COMPOSITES.index = pivots
    ALL_COMPOSITES.to_csv('composites.csv', index=False)
    return rlz

  def writeTSAtoXML(self, xml):
    """
      Write properties of TSA algorithms to XML
      @ In, xml, xmlUtils.StaticXmlElement, entity to write to
      @ Out, None
    """
    root = xml.getRoot()
    for algo in self._tsaAlgorithms:
      if algo not in self._tsaTrainedParams:
        continue
      algoNode = xmlUtils.newNode(algo.name)
      algo.writeXML(algoNode, self._tsaTrainedParams[algo])
      root.append(algoNode)
