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
  TimeSeriesAnalyzer inheritors are specific algorithms for characterizing, reproducing, and
  checking time histories.
"""
import abc

from ..utils import utils, InputData, InputTypes, mathUtils

# utility methods
class TimeSeriesAnalyzer(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Act as base class for objects that coordinate the time series analysis algorithms in RAVEN. Note these
    are not the ROM/SupervisedLearning objects; rather, used by those as well as other
    algorithms throughout the code. Maintain these algorithms in a way they can
    be called without accessing all of RAVEN.
  """
  # class attribute
  ## defines if missing values are accepted by the characterization algorithm
  _acceptsMissingValues = False

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.

      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for time series analysis algorithms used in RAVEN.'
    specs.addParam('target', param_type=InputTypes.StringListType, required=True,
        descr=r"""indicates the variables for which this algorithm will be used for characterization. """)
    specs.addParam('seed', param_type=InputTypes.IntegerType, required=False,
        descr=r"""sets a seed for the underlying random number generator, if present.""")
    return specs

  @classmethod
  def canGenerate(cls):
    """
      Determines if this algorithm is a generator.
      @ In, None
      @ Out, isGenerator, bool, True if this algorithm is a TimeSeriesGenerator
    """
    return issubclass(cls, TimeSeriesGenerator)

  @classmethod
  def canCharacterize(cls):
    """
      Determines if this algorithm is a characterizer.
      @ In, None
      @ Out, isCharacterizer, bool, True if this algorithm is a TimeSeriesCharacterizer
    """
    return issubclass(cls, TimeSeriesCharacterizer)

  @classmethod
  def canTransform(cls):
    """
      Determines if this algorithm is a transformer.
      @ In, None
      @ Out, isTransformer, bool, True if this algorithm is a TimeSeriesTransformer
    """
    return issubclass(cls, TimeSeriesTransformer)

  ### INHERITED METHODS ###
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    self.name = self.__class__.__name__ # the name the class shall be known by during its RAVEN life

  @abc.abstractmethod
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

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, spec, InputData.InputParams, input specifications
      @ Out, settings, dict, initialization settings for this algorithm
    """
    settings = {}
    settings['target'] = spec.parameterValues['target']
    settings['seed'] = spec.parameterValues.get('seed', None)

    settings = self.setDefaults(settings)

    return settings

  def setDefaults(self, settings):
    """
      Fills default values for settings with default values.
      @ In, settings, dict, existing settings
      @ Out, settings, dict, modified settings
    """
    if 'seed' not in settings:
      settings['seed'] = None
    return settings

  def canAcceptMissingValues(self):
    """
      Checks if the algorithm can accept missing values (generally NaN).
      @ In, None
      @ Out, _acceptsMissingValues, bool, if the characterization algorithm accepts missing values
    """
    # NOTE Signals may have missing values, and this is incompatible with some algorithms. As
    ## missing values will generally require special handling specific to the algorithm, this
    ## behavior defaults to False. It is left to each algorithm to implement how these missing
    ## values are handled.
    return self._acceptsMissingValues

  @abc.abstractmethod
  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, parameters from training this ROM
      @ Out, None
    """


class TimeSeriesGenerator(TimeSeriesAnalyzer):
  """
    Act as a mix-in class for algorithms that can generate synthetic time histories. This is
    reserved exclusively for stochastic algorithms. Deterministic generative algorithms should NOT
    inherit from this class.
  """
  # Class attributes
  ## defines if this algorithm is stochastic or deterministic
  _isStochastic = False

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.

      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super().getInputSpecification()
    return specs

  @classmethod
  def isStochastic(cls):
    """
      Method that returns if a Generator algorithm is stochastic or deterministic.

      @ In, None
      @ Out, _isStochastic, bool, True if this algorithm is stochastic and False if it is deterministic
    """
    return cls._isStochastic

  @abc.abstractmethod
  def generate(self, params, pivot, settings):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, synthetic, np.array(float), synthetic signal
    """


class TimeSeriesCharacterizer(TimeSeriesAnalyzer):
  """
    Acts as a mix-in class for algorithms that can generate characterize time-dependent signals. Any
    algorithm that has "useful" information to extract from a time-dependent signal, typically in
    the form of model parameters, should inherit from this class.
  """
  # class attributes
  ## define the clusterable features for this trainer.
  _features = []

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.

      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super().getInputSpecification()
    return specs


  @abc.abstractmethod
  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from characterize)
      @ Out, rlz, dict, realization-style response
    """

    # clustering
  def getClusteringValues(self, nameTemplate: str, requests: list, params: dict) -> dict:
    """
      Provide the characteristic parameters of this ROM for clustering with other ROMs
      @ In, nameTemplate, str, formatting string template for clusterable params (target, metric id)
      @ In, requests, list, list of requested attributes from this ROM
      @ In, params, dict, parameters from training this ROM
      @ Out, features, dict, params as {paramName: value}
    """
    # DEFAULT implementation, overwrite if structure is more complex
    # expected structure for "params" (trained parameters):
    # -> params = {key: #,          # as a scalar, or
    #              key: [#, #, #]}  # as a vector
    # -> cannot handle nested dicts as is
    #
    # nameTemplate convention:
    # -> target is the trained variable (e.g. Signal, Temperature)
    # -> metric is the algorithm used (e.g. Fourier, ARMA)
    # -> id is the subspecific characteristic ID (e.g. sin, AR_0)
    features = {}
    for target, info in params.items():
      for ident, value in info.items():
        if not mathUtils.isSingleValued(value):
          raise NotImplementedError('The default implementation of getClusteringValues for TSA cannot be used for vector values.')
        key = nameTemplate.format(target=target, metric=self.name, id=ident)
        features[key] = value
    return features

  def setClusteringValues(self, fromCluster, params):
    """
      Interpret returned clustering settings as settings for this algorithm.
      Acts somewhat as the inverse of getClusteringValues.
      @ In, fromCluster, list(tuple), (target, identifier, values) to interpret as settings
      @ In, params, dict, trained parameter settings
      @ Out, params, dict, updated parameter settings
    """
    # TODO this needs to be fast, as it's done a lot.
    for target, identifier, value in fromCluster:
      value = float(value)
      params[target][identifier] = value
    return params


class TimeSeriesTransformer(TimeSeriesAnalyzer):
  """
    Acts as a mix-in class for algorithms that can transform time-dependent signals. Algorithms
    which receive an input signal, then produce an output signal should inherit from this class.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.

      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super().getInputSpecification()
    return specs

  @abc.abstractmethod
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

  @abc.abstractmethod
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
