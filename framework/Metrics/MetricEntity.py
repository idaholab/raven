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

import numpy as np

from ..BaseClasses import PluginReadyEntity
from ..utils import utils, InputData, InputTypes
from .metrics.Factory import factory as MetricFactory

class MetricEntity(utils.metaclass_insert(abc.ABCMeta, PluginReadyEntity)):
  """
    This is the handler for RAVEN general interface metrics
  """
  interfaceFactory = MetricFactory

  #######################
  #
  # Construction
  #
  def __init__(self):
    """
      This is the basic method initialize the metric object
      @ In, none
      @ Out, none
    """
    super().__init__()
    self._metric = None # Interface instance

  #######################
  #
  # Properties
  #
  @property
  def acceptsProbability(self):
    """
      Tell whether this Metric accepts probability. Apparently unused currently.
      @ In, None
      @ Out, acceptsProbability, bool, determiner.
    """
    return self._metric.acceptsProbability if self._metric is not None else False

  @property
  def acceptsDistribution(self):
    """
      Tell whether this Metric accepts distributions. Apparently unused currently.
      @ In, None
      @ Out, acceptsDistribution, bool, determiner.
    """
    return self._metric.acceptsDistribution if self._metric is not None else False

  #######################
  #
  # API
  #
  def initialize(self, inputDict):
    """
      This method initialize each metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self._metric.initialize(inputDict)

  def _readMoreXML(self, xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    spec = self.parseXML(xmlNode)
    reqType = spec.parameterValues['subType']
    metric = self.interfaceFactory.returnInstance(reqType)
    metric.handleInput(spec)
    self._metric = metric

  def _getInterface(self):
    """
      Return the interface associated with this entity.
      @ In, None
      @ Out, _getInterface, object, interface object
    """
    return self._metric

  def evaluate(self, x, y, weights=None, axis=0, **kwargs):
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
    # NOTE, axis=0 is used in the previous implementation in Metric.evaluate calling
    #   Metric.__evaluateLocal__. This doesn't seem right, it should be passing axis along!
    #   However, for consistency, we keep it here for future investigation.
    return self._metric.run(x, y, weights=weights, axis=0, **kwargs)

  def getAlgorithmType(self):
    """
      Provide the metric sub-sub-type (used e.g. in SKL metrics)
      @ In, None
      @ Out, metricType, tuple/None, sub sub type (e.g. not Metric, not SKL, but mean_absolute_error)
    """
    return getattr(self._metric, 'metricType', None)

  def isDynamic(self):
    """
      This method is utility function that tells if the metric is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the metric is able to treat dynamic data, False otherwise
    """
    return self._metric.isDynamic()

  def isPairwise(self):
    """
      This method is utility function that tells if the metric is able to
      treat pairwise data on its own or not
      @ In, None
      @ Out, isPairwise, bool, True if the metric is able to handle pairwise data, False otherwise
    """
    return self._metric.isPairwise()
