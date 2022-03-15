
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
 Polynomial Regression
"""
import numpy as np
from ..utils import importerUtils
statsmodels = importerUtils.importModuleLazy("statsmodels", globals())

from ..utils import InputData, InputTypes, randomUtils, xmlUtils, mathUtils, utils
from .TimeSeriesAnalyzer import TimeSeriesCharacterizer, TimeSeriesGenerator


class PolynomialRegression(TimeSeriesGenerator, TimeSeriesCharacterizer):
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
    specs = super(PolynomialRegression, cls).getInputSpecification()
    specs.name = 'PolynomialRegression'
    specs.description = """fits time-series data using a polynomial function of degree one or greater."""
    specs.addSub(InputData.parameterInputFactory('degree', contentType=InputTypes.IntegerType,
                                                 descr="Specifies the degree polynomial to fit the data with."))
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

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = super().handleInput(spec)
    settings['degree'] = spec.findFirst('degree').value
    return settings

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, additional settings specific to this algorithm
      @ Out, params, dict, characteristic parameters
    """
    from sklearn.preprocessing import PolynomialFeatures
    import statsmodels.api as sm

    params = {target: {'model': {}} for target in targets}

    degree = settings['degree']
    features = PolynomialFeatures(degree=degree)
    xp = features.fit_transform(pivot.reshape(-1, 1))

    for target in targets:
      results = sm.OLS(signal, xp).fit()
      params[target]['model']['intercept'] = results.params[0]
      for i, value in enumerate(results.params[1:]):
        params[target]['model'][f'coef{i+1}'] = value
      params[target]['model']['object'] = results
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
      names.append(f'{base}__intercept')
      for i in range(1,settings['degree']):
        names.append(f'{base}__coef{i}')
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
      for name, value in info['model'].items():
        if name == 'object':
          continue
        rlz[f'{base}__{name}'] = value
    return rlz

  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic estimated model signal
    """
    from sklearn.preprocessing import PolynomialFeatures
    synthetic = np.zeros((len(pivot), len(params)))
    degree = settings['degree']
    features = PolynomialFeatures(degree=degree)
    xp = features.fit_transform(pivot.reshape(-1, 1))

    for t, (target, _) in enumerate(params.items()):
      model = params[target]['model']['object']
      synthetic[:, t] = model.predict(xp)

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
      for name, value in info['model'].items():
        if name == 'object':
          continue
        base.append(xmlUtils.newNode(name, text=f'{float(value):1.9e}'))
