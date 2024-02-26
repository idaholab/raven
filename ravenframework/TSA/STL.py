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
Created on July 7, 2023
@author: j-bryan

Detrending with STL decomposition
"""

import numpy as np

from .TimeSeriesAnalyzer import TimeSeriesTransformer, TimeSeriesGenerator
from ..utils import importerUtils, InputTypes, InputData, xmlUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())


class STL(TimeSeriesTransformer, TimeSeriesGenerator):
    """
      Performs STL decomposition on the data using the statsmodels.tsa.seasonal.STL class.
      See https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
    """
    _acceptsMissingValues = False

    @classmethod
    def getInputSpecification(cls):
      """
        Method to get a reference to a class that specifies the input data for class cls.

        @ In, None
        @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
      """
      specs = super(STL, cls).getInputSpecification()
      specs.name = 'STL'
      specs.description = r"""
        Decomposes the signal into trend, seasonal, and residual components using the STL method of
        Cleveland et al. (1990).
      """
      specs.addSub(InputData.parameterInputFactory(
         'seasonal', contentType=InputTypes.IntegerType,
         descr=r"""the length of the seasonal smoother."""
      ))
      specs.addSub(InputData.parameterInputFactory(
         'period',
         contentType=InputTypes.IntegerType,
         descr=r"""periodicity of the sequence."""
      ))
      specs.addSub(InputData.parameterInputFactory(
         'trend',
         contentType=InputTypes.IntegerType,
         descr=r"""the length of the trend smoother. Must be an odd integer."""
      ))
      return specs

    def handleInput(self, spec):
      """
        Reads user inputs into this object.
        @ In, spec, InputData.InputParams, input specifications
        @ Out, settings, dict, initialization settings for this algorithm
      """
      settings = super().handleInput(spec)
      for name in ['seasonal', 'period', 'trend']:
        node = spec.findFirst(name)
        if node is not None:
          settings[name] = node.value
      return settings

    def setDefaults(self, settings):
      """
        Allows user to specify default values for settings
        @ In, settings, dict, existing settings
        @ Out, settings, dict, updated settings
      """
      settings = super().setDefaults(settings)

      for key in ['seasonal', 'trend', 'period']:
        if key not in settings:
          settings[key] = None

      return settings

    def fit(self, signal, pivot, targets, settings):
      """
        Fits the algorithm/model using the provided time series ("signal") using methods specific to
        the algorithm.
        @ In, signal, np.array, time-dependent series
        @ In, pivot, np.array, time-like parameter
        @ In, targets, list(str), names of targets
        @ In, settings, dict, additional settings specific to algorithm
        @ Out, params, dict, characterization of signal; structure as:
                            params[target variable][characteristic] = value
      """
      # lazy import statsmodels
      import statsmodels.tsa.api as tsa

      # Unpack the 'seasonal', 'trend', and 'period' settings
      # If the value in settings is None, then the default statsmodels argument is used
      stlSettings = {k: settings[k] for k in ['seasonal', 'trend', 'period'] if settings[k] is not None}

      params = {}
      for tg, target in enumerate(targets):
        model = tsa.STL(signal[:, tg], **stlSettings).fit()
        params[target] = {'model': model}

      return params

    def generate(self, params, pivot, settings):
      """
        Generates a synthetic history from fitted parameters.
        @ In, params, dict, training parameters as from self.characterize
        @ In, pivot, np.array, time-like array values
        @ In, settings, dict, additional settings specific to algorithm
        @ Out, synthetic, np.array(float), synthetic signal
      """
      synthetic = np.zeros((len(pivot), len(params)))
      for t, (target, data) in enumerate(params.items()):
        # Generate consists of the trend and seasonal components since this is the mean behavior
        synthetic[:, t] = data['model'].trend + data['model'].seasonal
      return synthetic

    def getResidual(self, initial, params, pivot, settings):
      """
        Removes trained signal from data and find residual
        @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                                same order as self.target
        @ In, params, dict, training parameters as from self.characterize
        @ In, pivot, np.array, time-like array values
        @ In, settings, dict, additional settings specific to algorithm
        @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
      """
      residual = initial.copy()
      for t, (target, data) in enumerate(params.items()):
        # We remove both the trend and seasonal components, leaving only the residual
        residual[:, t] = data['model'].resid
      return residual

    def getComposite(self, initial, params, pivot, settings):
      """
        Combines two component signals to form a composite signal. This is essentially the inverse
        operation of the getResidual method.
        @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                                same order as self.target
        @ In, params, dict, training parameters as from self.characterize
        @ In, pivot, np.array, time-like array values
        @ In, settings, dict, additional settings specific to algorithm
        @ Out, composite, np.array, resulting composite signal
      """
      composite = initial + self.generate(params, pivot, settings)
      return composite

    def writeXML(self, writeTo, params):
      """
        Allows the engine to put whatever it wants into an XML to print to file.
        @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
        @ In, params, dict, parameters from training this ROM
        @ Out, None
      """
      for target, info in params.items():
        base = xmlUtils.newNode(target)
        writeTo.append(base)
