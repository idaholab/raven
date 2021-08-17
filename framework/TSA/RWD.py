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
    specs.name = 'RWD' # NOTE lowercase because ARMA already has Fourier and no way to resolve right now
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
      settings['SignatureWindowLength'] = 105#len(history)//10
      settings['SampleType'] = 1

    return settings #### 

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict of list: 1st is the uvectors, 2nd is params (LOC, HOC)
    """
    # lazy import statsmodels
    import statsmodels.api
    # settings:
    #   SignatureWindowLength, int,  Signature window length
    #   FeatureIndex, list of int,  The index that contains differentiable params

    params = {}
    for tg, target in enumerate(targets):
      history = signal[:, tg]
      SignatureWindowLength = settings['SignatureWindowLength']
      fi = settings['FeatureIndex']
      SampleType = settings['SampleType']
      AllWindowNumber = len(history)-SignatureWindowLength+1
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
      
      
      params[target] = [U, FeatureMatrix]
      
        
        
    return params 


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
      SigMat_synthetic = params[target][0] @ params[target][1]
      synthetic[:, t] = np.hstack((SigMat_synthetic[0,:-1], SigMat_synthetic[:,-1]))

    return synthetic

############## problems
  def writeXML(self, writeTo, features):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, features, dict, features from as from self.characterize
      @ Out, None
    """
    for target, info in features.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      base.append(xmlUtils.newNode('constant', text=f'{float(info["rwd"]["const"]):1.9e}'))
      for p, ar in enumerate(info['rwd']['UVec']):
        base.append(xmlUtils.newNode(f'AR_{p}', text=f'{float(ar):1.9e}'))

      base.append(xmlUtils.newNode('variance', text=f'{float(info["rwd"]["var"]):1.9e}'))
      
