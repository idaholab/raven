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
Created on Jul 18 2016

@author: mandd, wangc
"""
import abc

from ...BaseClasses import BaseInterface
from ...utils import utils

class MetricInterface(utils.metaclass_insert(abc.ABCMeta, BaseInterface)):
  """
    This is the general interface to any RAVEN metric object.
    It contains an initialize, a _readMoreXML, and an evaluation (i.e., distance) methods
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.name = cls.__name__
    return inputSpecification

  def __init__(self):
    """
      This is the basic method initialize the metric object
      @ In, none
      @ Out, none
    """
    super().__init__()
    self.type = self.__class__.__name__
    # If True the metric needs to be able to handle (value,probability) where value and probability are lists
    self.acceptsProbability = False
    # If True the metric needs to be able to handle a passed in Distribution
    self.acceptsDistribution = False
    # If True the metric needs to be able to handle dynamic data
    self._dynamicHandling = False
    # If True the metric needs to be able to handle pairwise data
    self._pairwiseHandling = False

  def initialize(self, inputDict):
    """
      This method initialize each metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    pass

  def handleInput(self, spec):
    """
      Method that reads the portion of input that belongs to this specialized class
      and initialize internal parameters
      @ In, spec, InputData.parameterInput, input specs
      @ Out, None
    """
    super().handleInput(spec)

  @abc.abstractmethod
  def run(self, x, y, weights=None, axis=0, **kwargs):
    """
      This method compute the metric between x and y
      @ In, x, numpy.ndarray or instance of Distributions.Distribution, array containing data of x,
        or given distribution.
      @ In, y, numpy.ndarray, or instance of Distributions.Distribution, array containing data of y,
        or given distribution.
      @ In, weights, numpy.ndarray, optional, an array of weights associated with x
      @ In, axis, integer, optional, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float or numpy.array, metric results between x and y
    """

  def isDynamic(self):
    """
      This method is utility function that tells if the metric is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the metric is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  def isPairwise(self):
    """
      This method is utility function that tells if the metric is able to
      treat pairwise data on its own or not
      @ In, None
      @ Out, isPairwise, bool, True if the metric is able to handle pairwise data, False otherwise
    """
    return self._pairwiseHandling
