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
np.random.default_rng(seed=42)

import Decorators

import string
import numpy.linalg as LA
import pandas as pd
import copy as cp
from math import*

from utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

import Distributions
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer



# utility methods
class RWD(TimeSeriesCharacterizer):
  r"""
    Randomized Window Decomposition
  """
  # class attribute
  ## define the clusterable features for this trainer.


  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(RWD, cls).getInputSpecification()
    specs.name = 'rwd' # NOTE lowercase because ARMA already has Fourier and no way to resolve right now
    specs.description = r"""TimeSeriesAnalysis algorithm for sliding window snapshots to generate features"""

    specs.addSub(InputData.parameterInputFactory('SignatureWindowLength', contentType=InputTypes.FloatType,
                 descr=r"""the size of signature window, which represents as a snapshot for a certain time step;
                       typically represented as $w$ in literature, or $w_sig$ in the code."""))
    specs.addSub(InputData.parameterInputFactory('FeatureIndex', contentType=InputTypes.FloatType,
                 descr=r""" Index used for feature selection, which requires pre-analysis for now, will be addresses
                 via other non human work required method """))
    specs.addSub(InputData.parameterInputFactory('SampleType', contentType=InputTypes.FloatType,
                 descr=r"""Indicating the type of sampling."""))
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
      @ In, SampleType, integer = 0, 1, 2
      @     SampleType = 0: Sequentially Sampling
      @     SampleType = 1: Randomly Sampling
      @     SampleType = 2: Piecewise Sampling
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['SignatureWindowLength'] = spec.findFirst('SignatureWindowLength').value
    settings['FeatureIndex'] = spec.findFirst('FeatureIndex').value
    settings['SampleType'] = spec.findFirst('SampleType').value


    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    settings = super().setDefaults(settings)
    if 'SignatureWindowLength' not in settings:
      settings['SignatureWindowLength'] = None #len(history)//10
      settings['SampleType'] = 1

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
    #   SignatureWindowLength, int,  Signature window length
    #   FeatureIndex, list of int,  The index that contains differentiable params


    params = {}

    for tg, target in enumerate(targets):
      history = signal[:, tg]
      if settings['SignatureWindowLength'] is None:
        settings['SignatureWindowLength'] = len(history)//10
      SignatureWindowLength = int(settings['SignatureWindowLength'])
      fi = settings['FeatureIndex']
      SampleType = settings['SampleType']
      AllWindowNumber = int(len(history)-SignatureWindowLength+1)
      print(type(AllWindowNumber))
      print('AllWindowNumber',AllWindowNumber)
      print(type(SignatureWindowLength))
      print(SignatureWindowLength)
      SignatureMatrix = np.zeros((SignatureWindowLength, AllWindowNumber))
      for i in range(AllWindowNumber):
        SignatureMatrix[:,i] = np.copy(history[i:i+SignatureWindowLength])

      # Sequential sampling
      if SampleType == 0:
        BaseMatrix = np.copy(SignatureMatrix)

      # Randomized sampling
      elif SampleType == 1:
        sampleLimit = len(history)-SignatureWindowLength

        WindowNumber = sampleLimit//4
        sampleIndex = np.random.randint(sampleLimit, size=WindowNumber)
        BaseMatrix = np.zeros((SignatureWindowLength, WindowNumber))
        for i in range(WindowNumber):
          WindowIndex = sampleIndex[i]
          BaseMatrix[:,i] = np.copy(history[WindowIndex:WindowIndex+SignatureWindowLength])
        #print('sampleIndex: ',sampleIndex)
      # Piecewise Sampling
      else:
        WindowNumber = len(history)//SignatureWindowLength
        BaseMatrix = np.zeros((SignatureWindowLength, WindowNumber))
        for i in range(WindowNumber-1):
          BaseMatrix[:,i] = np.copy(history[i*SignatureWindowLength:(i+1)*SignatureWindowLength])


      U,s,V = mathUtils.computeTruncatedSingularValueDecomposition(BaseMatrix,0)
      print('SignatureMatrix ',SignatureMatrix.shape)

      FeatureMatrix = U.T @ SignatureMatrix
      print('FeatureMatrix ',FeatureMatrix.shape)


      params[target] = {'UVec'   : U,
                        'Feature': FeatureMatrix}
      print('%%%%%%%%%%%%%%%%%%%%%%%%')
      print(list(params))
    return params


  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    names = []
    print('@@@@@@@@@@@@@@@@settings:', settings)
    for target in settings['target']:
      base = f'{self.name}__{target}'
      names.append(f'{base}__Feature')
      names.append(f'{base}__UVec')

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
      print('##################################')
      print('##################################    target:', target)
      print('base', base)
      #print('info',info)
      (m,n) = info['Feature'].shape
      (k,l) = (info['UVec']).shape
      for j in range(m):
        for i in range(n):
          rlz[f'{base}__Feature{j}_{i}'] = info['Feature'][j,i]
      print('^^^^^^^^^^^^^^^^^', info['UVec'].shape)
      for j in range(k):
        for i in range(l):
          rlz[f'{base}__UVec{j}_{i}'] = info['UVec'][j,i]

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
      SigMat_synthetic = params[target]['UVec'] @ params[target]['Feature']
      synthetic[:, t] = np.hstack((SigMat_synthetic[0,:-1], SigMat_synthetic[:,-1]))

    return synthetic

############## problems
  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, params from as from self.characterize
      @ Out, None
    """
    #print('params.items(): ',type(params.items()))
    #print('len(params.items()) ',len(params.items()))
    #print('params.items() ',params.items())
    counter = 0
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      print('  target', target)
      writeTo.append(base)
      print('  ', type(info["UVec"]))
      print('  ', info["UVec"].shape)
      U0 = info["UVec"][:,0]
      U1 = info["UVec"][:,1]
      counter +=1
      print('counter', counter)
      #base.append(xmlUtils.newNode('UVec', text=f'{float():1.9e}'))
      for p, ar in enumerate(U0):
        base.append(xmlUtils.newNode(f'UVec0_{p}', text=f'{float(ar):1.9e}'))
        print(p)
      for p, ar in enumerate(U1):
        base.append(xmlUtils.newNode(f'UVec1_{p}', text=f'{float(ar):1.9e}'))
      #base.append(xmlUtils.newNode('variance', text=f'{float(info["var"]):1.9e}'))

