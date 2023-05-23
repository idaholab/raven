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
  AutoRegressive Moving Average time series analysis
"""
import sys
import importlib
import numpy as np
from sklearn.base import TransformerMixin

from ..utils import InputData, InputTypes, xmlUtils, importerUtils
statsmodels = importerUtils.importModuleLazy('statsmodels', globals())

from .TimeSeriesAnalyzer import TimeSeriesGenerator, TimeSeriesCharacterizer


def getModuleAttribute(name):
  """
    Utility function which fetches a desired module attribute. For example, this could be a class
    or a function. The module will be imported if it is not found in sys.modules.
    @ In, name, str, full name (including module) of the attribute to fetch
    @ Out, attribute, Any, attribute object
  """
  nameSplit = name.split('.')
  modulePath = '.'.join(nameSplit[:-1])
  attrName = nameSplit[-1]
  # if modulePath not in sys.modules:  # import the module if it hasn't been already
  if not importerUtils.isLibAvail(modulePath):
    importlib.import_module(modulePath)
  attribute = getattr(sys.modules[modulePath], attrName)
  return attribute


# utility methods
class Transformer(TimeSeriesGenerator, TimeSeriesCharacterizer):
  r"""
    Wrapper for scikit-learn data transformation algorithms
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(Transformer, cls).getInputSpecification()
    specs.name = 'transformer'
    specs.description = r"""TimeSeriesAnalysis algorithm for applying a specified transformation
                        to the target values by leveraging scikit-learn's preprocessing module."""
    specs.addParam('subType', param_type=InputTypes.StringType, required=True,
                   descr=r"""specifies the type of transformer to use. This name should match the name
                   of a class in sklearn.preprocessing. Keyword arguments for the transformer can be 
                   passed using subnodes of the form <key>value</key>.""")
    argNode = InputData.parameterInputFactory('arg', contentType=InputTypes.StringType,
                                              descr=r"""specifies a keyword argument to be passed to the
                                              transformer, given in the form key|value.""")
    argNode.addParam('type', param_type=InputTypes.makeEnumType('TransformerArgType',
                     xmlName='xsd:string',
                     enumList=['str', 'int', 'float', 'callable']),
                     descr=r"""specifies the data type of the argument.""",
                     required=True)
    specs.addSub(argNode)
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
    settings['subType'] = spec.parameterValues.get('subType')

    transformerClass = getModuleAttribute(settings['subType'])
    # The only requirement of the transformer is to inheret from sklearn.base.TransformerMixin
    ## because that defines the fit/transform/inverse_transform interface used. Since we import
    ## the desired class specifically, it can potentially be from any available library or from
    ## somewhere else in RAVEN.
    if not issubclass(transformerClass, TransformerMixin):
      raise TypeError(f'Specified transformer class {settings["subType"]} is not a'
                      'subclass of sklearn.base.TransformerMixin.')
    self.Transformer = transformerClass

    # Parse keyword arguments passed in through <arg> nodes
    kwargs = {}
    for argNode in spec.findAll('arg'):
      dataType = argNode.parameterValues['type']
      key, value = argNode.value.split('|')
      typedValue = self._interpretArgumentType(value, dataType)
      kwargs[key] = typedValue
    settings['transformerKwargs'] = kwargs

    return settings
  
  def _interpretArgumentType(self, value, dataType):
    """
      Coerces value to be of type dataType. In the case that dataType="callable", the callable object
      is returned, importing modules as necessary to retrieve the object.
      @ In, value, str, argument value
      @ In, dataType, str, data type that value should take; may be 'str', 'float', 'int', or 'callable'
      @ Out, typedValue, object, value coerced to dataType
    """
    if dataType == 'str':
      typedValue = value
    if dataType == 'float':
      typedValue = float(value)
    elif dataType == 'int':
      typedValue = int(value)
    elif dataType == 'callable':
      typedValue = getModuleAttribute(value)
    else:
      self.raiseAnError(TypeError, f'Unknown data type {dataType} provided for argument value {value}.')

    return typedValue

  def characterize(self, signal, pivot, targets, settings):
    """
      Determines the charactistics of the signal based on this algorithm.
      @ In, signal, np.ndarray, time series with dims [time, target]
      @ In, pivot, np.1darray, time-like parameter values
      @ In, targets, list(str), names of targets in same order as signal
      @ In, settings, dict, settings for this ROM
      @ Out, params, dict, characteristic parameters
    """
    params = {}
    for tg, target in enumerate(targets):
      params[target] = {}
      history = signal[:, [tg]]  # must be formatted as column vectors
      transformer = self.Transformer(**settings['transformerKwargs']).fit(history)
      params[target]['model'] = transformer
      params[target]['modelType'] = settings['subType']
      params[target]['transformerKwargs'] = settings['transformerKwargs']
    return params

  def getResidual(self, initial, params, pivot, settings):
    """
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    residual = np.zeros((len(pivot), len(params)))
    for t, (target, data) in enumerate(params.items()):
      # We need to pass data to transform as column vector, then flatten again
      residual[:, t] = data['model'].transform(initial[:, [t]]).ravel()
    return residual
  
  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, characterization such as otained from self.characterize()
      @ In, pivot, np.array(float), pivot parameter values
      @ In, settings, dict, settings for this ROM
      @ Out, synthetic, np.array(float), synthetic ARMA signal
    """
    # We have to implement this (abstract method) but getComposite needs to do the heavy
    ## lifting here since the transformation operates
    synthetic = np.zeros((len(pivot), len(params)))
    return synthetic

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
    composite = self.generate(params, pivot, settings)
    for t, (target, data) in enumerate(params.items()):
      # We need to pass data to inverse_transform as column vector, then flatten again
      composite[:, t] = data['model'].inverse_transform(initial[:, [t]]).ravel()
    return composite
  
  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      rlz[f'{base}__name'] = info['modelType']
    return rlz

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
      # Write transformer name and any keyword arguments to XML
      transformerNode = xmlUtils.newNode('transformer', attrib={'subType': info['modelType']})
      for k, v in info['transformerKwargs'].items():
        valueText = v if not hasattr(v, '__name__') else v.__name__
        transformerNode.append(xmlUtils.newNode(k, text=valueText))
      base.append(transformerNode)
