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
  Randomized Window Decomposition
"""

import collections
import numpy as np
import scipy as sp
from .. import Decorators
import string
import numpy.linalg as LA
import pandas as pd
import copy as cp


from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .. import Distributions
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer

# utility methods
class RWD(TimeSeriesCharacterizer):
  r"""
    Randomized Window Decomposition
  """


  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(RWD, cls).getInputSpecification()
    specs.name = 'rwd'
    specs.description = r"""TimeSeriesAnalysis algorithm for sliding window snapshots to generate features"""

    specs.addSub(InputData.parameterInputFactory('signatureWindowLength', contentType=InputTypes.IntegerType,
                 descr=r"""the size of signature window, which represents as a snapshot for a certain time step;
                       typically represented as $w$ in literature, or $w_sig$ in the code."""))
    specs.addSub(InputData.parameterInputFactory('featureIndex', contentType=InputTypes.IntegerType,
                 descr=r""" Index used for feature selection, which requires pre-analysis for now, will be addresses
                 via other non human work required method """))
    specs.addSub(InputData.parameterInputFactory('sampleType', contentType=InputTypes.IntegerType,
                 descr=r"""Indicating the type of sampling."""))
    specs.addSub(InputData.parameterInputFactory('seed', contentType=InputTypes.IntegerType,
                 descr=r"""Indicating random seed."""))
    return specs


  #
  # API Methods
  #
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    super().__init__(*args, **kwargs)
    self._minBins = 20 # this feels arbitrary; used for empirical distr. of data

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifsications
      @ In, sampleType, integer = 0, 1, 2
      @     sampleType = 0: Sequentially Sampling
      @     sampleType = 1: Randomly Sampling
      @     sampleType = 2: Piecewise Sampling
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['signatureWindowLength'] = spec.findFirst('signatureWindowLength').value
    settings['featureIndex'] = spec.findFirst('featureIndex').value
    settings['sampleType'] = spec.findFirst('sampleType').value
    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'signatureWindowLength' not in settings:
      settings['signatureWindowLength'] = None
      settings['sampleType'] = 1
    if 'engine' not in settings:
      settings['engine'] = randomUtils.newRNG()
    if 'seed' not in settings:
      settings['seed'] = 42
    return settings ####

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict of dict: 1st level contains targets/variables; 2nd contains: U vectors and features
    """
    # lazy import statsmodels
    import statsmodels.api
    # settings:
    #   signatureWindowLength, int,  Signature window length
    #   featureIndex, list of int,  The index that contains differentiable params
    seed = settings['seed']
    if seed is not None:
      randomUtils.randomSeed(seed, engine=settings['engine'], seedBoth=True)

    params = {}

    for tg, target in enumerate(targets):
      history = signal[:, tg]
      if settings['signatureWindowLength'] is None:
        settings['signatureWindowLength'] = len(history)//10
      signatureWindowLength = int(settings['signatureWindowLength'])
      fi = int(settings['featureIndex'])
      sampleType = settings['sampleType']
      allWindowNumber = int(len(history)-signatureWindowLength+1)

      signatureMatrix = np.zeros((signatureWindowLength, allWindowNumber))
      for i in range(allWindowNumber):
        signatureMatrix[:,i] = np.copy(history[i:i+signatureWindowLength])

      # Sequential sampling
      if sampleType == 0:
        baseMatrix = np.copy(signatureMatrix)

      # Randomized sampling
      elif sampleType == 1:
        sampleLimit = len(history)-signatureWindowLength
        windowNumber = sampleLimit//4
        baseMatrix = np.zeros((signatureWindowLength, windowNumber))
        for i in range(windowNumber):
          windowIndex = randomUtils.randomIntegers(0, sampleLimit, caller=None)
          baseMatrix[:,i] = np.copy(history[windowIndex:windowIndex+signatureWindowLength])

      # Piecewise Sampling
      elif sampleType == 2:
        windowNumber = len(history)//signatureWindowLength
        baseMatrix = np.zeros((signatureWindowLength, windowNumber))
        for i in range(windowNumber-1):
          baseMatrix[:,i] = np.copy(history[i*signatureWindowLength:(i+1)*signatureWindowLength])
      U,s,V = mathUtils.computeTruncatedSingularValueDecomposition(baseMatrix,0)
      featureMatrix = U.T @ signatureMatrix
      params[target] = {'uVec'   : U[:,0:fi],
                        'Feature': featureMatrix}
    return params


  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    for target in settings['target']:
      base = f'{self.name}__{target}'
      sw = int(settings['signatureWindowLength'])
      fi = int(settings['featureIndex'])
      for i in range(fi):
        for j in range(sw):
          names.append(f'{base}__uVec{i}_{j}')
    return names

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      (k,l) = (info['uVec']).shape
      for i in range(l):
        for j in range(k):
          rlz[f'{base}__uVec{i}_{j}'] = info['uVec'][j,i]
    return rlz



  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic estimated model signal
    """

    synthetic = np.zeros((len(pivot), len(params)))
    for t, (target, _) in enumerate(params.items()):
      sigMatSynthetic = params[target]['uVec'] @ params[target]['Feature']
      synthetic[:, t] = np.hstack((sigMatSynthetic[0,:-1], sigMatSynthetic[:,-1]))

    return synthetic

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, params from as from self.characterize
      @ Out, None
    """
    counter = 0
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      (m,n) = info["uVec"].shape
      for i in range(n):
        U0 = info["uVec"][:,0]
        counter +=1
        for p, ar in enumerate(U0):
          base.append(xmlUtils.newNode(f'uVec{i}_{p}', text=f'{float(ar):1.9e}'))
