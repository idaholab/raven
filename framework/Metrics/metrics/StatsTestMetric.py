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
Created on 2021 September 12

@author: Robert Flanagan
"""
import numpy as np
from scipy import stats

from .MetricInterface import MetricInterface
from Metrics.metrics import MetricUtilities
from utils import InputData
import StatsTestUtils as sU

class StatsTestMetric(MetricInterface):
  """
    Metric to compare two datasets using statistical tests.
  """
  available_tests = {}
  available_tests['full']['f_test'] = stats.f_oneway
  available_tests['full']['chi_square'] = stats.chi_square
  available_tests['full']['ks_test'] = stats.ks_2samp
  available_tests['seg']['f_test'] = sU.compare_segments_f
  available_tests['seg']['chi_square'] = sU.compare_segments_chi
  available_tests['seg']['ks_test'] = sU.compare_segments_ks
  available_tests['int']['f_test'] = sU.compare_interval_f
  available_tests['int']['chi_square'] = sU.compare_interval_chi
  available_tests['int']['ks_test'] = sU.compare_interval_ks
  available_tests['seg_sum']['f_test'] = sU.compare_sums_f
  available_tests['seg_sum']['chi_square'] = sU.compare_sums_chi
  available_tests['seg_sum']['ks_test'] = sU.compare_sums_ks

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # If True the metric needs to be able to handle a passed in Distribution
    self.acceptsDistribution = True
    self.metricType = None

  def handleInput(self, paramInput):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, paramInput, InputData.parameterInput, input specs
      @ Out, None
    """
    self.distParams = {}
    for child in paramInput.subparts:
      if child.getName() == "metricType":
        self.metricType = list(elem.strip() for elem in child.value.split('|'))
      else:
        self.distParams[child.getName()] = child.value
    if self.metricType[0] not in self.__class__.availMetrics.keys() or self.metricType[1] not in self.__class__.availMetrics[self.metricType[0]].keys():
      self.raiseAnError(IOError, "Metric '", self.name, "' with metricType '", self.metricType[0], "|", self.metricType[1], "' is not valid!")

  def run(self, x, y, weights=None, axis=0, **kwargs):
    """
      This method computes difference between two points x and y based on given metric
      @ In, x,  instance of Distributions.Distribution, tuple or list, array containing data of x,
        or given distribution.
      @ In, y, instance of Distributions.Distribution, tuple or list, array containing data of y,
        or given distribution.
      @ In, weights, array_like (numpy.ndarray or list), optional, not used in this metric
      @ In, axis, integer, optional, default is 0, not used for this metric.
      @ In, kwargs,dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result, CDF area difference
    """
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      assert x.shape == y.shape, "Input data x, y should have the same shape!"
      dictTemp = utils.mergeDictionaries(kwargs, self.distParams)
      try:
        value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x, y, metricType[2])
      except TypeError as e:
        self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type', self.metricType[1])
        self.raiseAnError(TypeError, 'Input parameters error: \n', str(e), '\n')
    else:
      self.raiseAnError(IOError, "Input data type is not correct!")
    return value


        
