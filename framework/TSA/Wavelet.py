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
"""
import numpy as np

from utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, utils
from .TimeSeriesAnalyzer import TimeSeriesAnalyzer


# utility methods
class Wavelet(TimeSeriesAnalyzer):
  """
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Wavelet, cls).getInputSpecification()
    specs.name = 'Wavelet'
    specs.descriiption = """FILL THIS IN"""
    specs.addSub(InputData.parameterInputFactory('family', contentType=InputTypes.StringType,
                                                 descr="""FILL THIS IN"""))
    return specs


  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    TimeSeriesAnalyzer.__init__(self, *args, **kwargs)


  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = TimeSeriesAnalyzer.handleInput(self, spec)
    return settings


  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ In, simultFit, bool, optional, if False then fit Fourier individually
      @ Out, params, dict, characteristic parameters
    """
    import pywt
    family = settings['family']
    params = {target: {'results': {}} for target in targets}

    for i, target in enumerate(targets):
      results = params[target]['results']
      results['coeff_a'], results['coeff_d'] = pywt.dwt(signal[:, i], family)

    return params



  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    import pywt
    synthetic = np.zeros((len(pivot), len(params)))
    family = settings['family']
    for t, (target, _) in enumerate(params.items()):
      results = params[target]['results']
      cA = results['coeff_a']
      cD = results['coeff_d']
      synthetic[:, t] = pywt.idwt(cA, cD, family)
    return synthetic


  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.characterize
      @ Out, None
    """
    pass
