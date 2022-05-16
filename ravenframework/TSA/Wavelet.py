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
  Discrete Wavelet Transformation for Characterization of Time-Series Data.
"""
import numpy as np

from ..utils import InputData, InputTypes, xmlUtils
from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer


# utility methods
class Wavelet(TimeSeriesGenerator, TimeSeriesCharacterizer):
  """
    Perform Discrete Wavelet Transformation on time-dependent data.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(Wavelet, cls).getInputSpecification()
    specs.name = 'wavelet'
    specs.description = r"""Discrete Wavelet TimeSeriesAnalysis algorithm. Performs a discrete wavelet transform
        on time-dependent data. Note: This TSA module requires pywavelets to be installed within your
        python environment."""
    specs.addSub(InputData.parameterInputFactory(
      'family',
      contentType=InputTypes.StringType,
      descr=r"""The type of wavelet to use for the transformation.
    There are several possible families to choose from, and most families contain
    more than one variation. For more information regarding the wavelet families,
    refer to the Pywavelets documentation located at:
    https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html (wavelet-families)
    \\
    Possible values are:
    \begin{itemize}
      \item \textbf{haar family}: haar
      \item \textbf{db family}: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11,
        db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23,
        db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35,
        db36, db37, db38
      \item \textbf{sym family}: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10,
        sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
      \item \textbf{coif family}: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8,
        coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
      \item \textbf{bior family}: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6,
        bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5,
        bior6.8
      \item \textbf{rbio family}: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6,
        rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5,
        rbio6.8
      \item \textbf{dmey family}: dmey
      \item \textbf{gaus family}: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
      \item \textbf{mexh family}: mexh
      \item \textbf{morl family}: morl
      \item \textbf{cgau family}: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
      \item \textbf{shan family}: shan
      \item \textbf{fbsp family}: fbsp
      \item \textbf{cmor family}: cmor
    \end{itemize}"""))
    return specs

  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a time-series analysis object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    # general infrastructure
    super().__init__(*args, **kwargs)


  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['family'] = spec.findFirst('family').value
    return settings


  def characterize(self, signal, pivot, targets, settings):
    """
      This function utilizes the Discrete Wavelet Transform to
      characterize a time-dependent series of data.

      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ Out, params, dict, characteristic parameters
    """
    # TODO extend to continuous wavelet transform
    try:
      import pywt
    except ModuleNotFoundError:
      print("This RAVEN TSA Module requires the PYWAVELETS library to be installed in the current python environment")
      raise ModuleNotFoundError


    ## The pivot input parameter isn't used explicity in the
    ## transformation as it assumed/required that each element in the
    ## time-dependent series is independent, uniquely indexed and
    ## sorted in time.
    family = settings['family']
    params = {target: {'results': {}} for target in targets}

    for i, target in enumerate(targets):
      results = params[target]['results']
      results['coeff_a'], results['coeff_d'] = pywt.dwt(signal[:, i], family)

    return params

  def getParamNames(self, settings):
    """
      Return list of expected variable names based on the parameters
      @ In, settings, dict, training parameters for this algorithm
      @ Out, names, list, string list of names
    """
    # FIXME we don't know a priori how many entries will be in the decomp, so we can't register it yet!
    raise NotImplementedError('Cannot predict variables for Wavelet!')
    names = []
    for target in settings['target']:
      base = f'{self.name}__{target}'

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    # FIXME we don't know a priori how many entries will be in the decomp, so we can't register it yet!
    raise NotImplementedError('Cannot predict variables for Wavelet!')
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      for name, values in info['results'].items():
        for v, val in enumerate(values):
          rlz[f'{base}__{name}__{v}'] = val
    return rlz

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    try:
      import pywt
    except ModuleNotFoundError:
      print("This RAVEN TSA Module requires the PYWAVELETS library to be installed in the current python environment")
      raise ModuleNotFoundError

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
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)
      for name, value in info['results'].items():
        base.append(xmlUtils.newNode(name, text=','.join([str(v) for v in value])))
