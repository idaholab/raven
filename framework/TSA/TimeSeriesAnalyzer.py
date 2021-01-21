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

from utils import utils, InputData, InputTypes

# utility methods
class TimeSeriesAnalyzer(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Act as base class for objects that coordinate the time series analysis algorithms in RAVEN. Note these
    are not the ROM/SupervisedLearning objects; rather, used by those as well as other
    algorithms throughout the code. Maintain these algorithims in a way they can
    be called without accessing all of RAVEN.
  """
  # class attribute
  ## define the clusterable features for this trainer.
  _features = []

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for time series analysis algorithms used in RAVEN.'
    specs.addParam('target', param_type=InputTypes.StringListType, required=True,
        descr=r"""indicates the variables for which this algorithm will be used for characterization. """)
    return specs

  ### INHERITED METHODS ###
  def __init__(self, *args, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, args, list, an arbitrary list of positional values
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    self.name = self.__class__.__name__ # the name the class shall be known by during its RAVEN life
    self.target = None                  # list of output variables for this TSA algo

  def handleInput(self, spec):
    """
      Reads user inputs into this object.
      @ In, inp, InputData.InputParams, input specifications
      @ Out, None
    """
    self.target = spec.parameterValues['target']

  @abc.abstractmethod
  def characterize(self, signal, pivot, targets, **kwargs):
    """
      Characterizes the provided time series ("signal") using methods specific to this algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, kwargs, dict, unused optional keyword arguments
      @ Out, params, dict, characterization of signal
    """
    pass

  def getResidual(self, initial, params, pivot, randEngine):
    """
      Removes trained signal from data and find residual
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    # DEFAULT IMPLEMENTATION, generate one signal and subtract it from the given one
    # -> overload in inheritors to change behavior
    sample = self.generate(params, pivot, randEngine)
    residual = initial - sample
    return residual

  @abc.abstractmethod
  def generate(self, params, pivot, randEngine):
    """
      Generates a synthetic history from fitted parameters.
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, randEngine, instance, optional, method to call to get random samples (for example "randEngine(size=6)")
      @ Out, synthetic, np.array(float), synthetic signal
    """

  def writeXML(self, writeTo, target):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, target, str, which target to write info for
      @ Out, None
    """
    pass # overwrite in subclasses