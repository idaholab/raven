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
Created on Feb. 16 2018

@author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import scipy
import numpy as np
import scipy.spatial.distance as spatialDistance
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .MetricInterface import MetricInterface
from ...utils import utils, InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ScipyMetric(MetricInterface):
  """
    ScipyMetric metrics which can be employed for both pointSets and historySets
  """
  availMetrics = {}
  # Distance functions between two numeric vectors
  availMetrics['paired_distance'] = {}
  availMetrics['paired_distance']['braycurtis'] = spatialDistance.braycurtis
  availMetrics['paired_distance']['canberra']   = spatialDistance.canberra
  availMetrics['paired_distance']['correlation'] = spatialDistance.correlation
  availMetrics['paired_distance']['minkowski']  = spatialDistance.minkowski
  # Distance functions between two boolean vectors
  availMetrics['boolean'] = {}
  availMetrics['boolean']['rogerstanimoto']     = spatialDistance.rogerstanimoto
  availMetrics['boolean']['dice']               = spatialDistance.dice
  availMetrics['boolean']['hamming']            = spatialDistance.hamming
  availMetrics['boolean']['jaccard']            = spatialDistance.jaccard
  availMetrics['boolean']['kulsinski']           = spatialDistance.kulsinski
  availMetrics['boolean']['russellrao']         = spatialDistance.russellrao
  availMetrics['boolean']['sokalmichener']      = spatialDistance.sokalmichener
  availMetrics['boolean']['sokalsneath']        = spatialDistance.sokalsneath
  availMetrics['boolean']['yule']               = spatialDistance.yule

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
    inputSpecification.addSub(InputData.parameterInputFactory("metricType",contentType=InputTypes.StringType),quantity=InputData.Quantity.one)
    inputSpecification.addSub(InputData.parameterInputFactory("w",contentType=InputTypes.FloatListType),quantity=InputData.Quantity.zero_to_one)
    inputSpecification.addSub(InputData.parameterInputFactory("p",contentType=InputTypes.FloatType),quantity=InputData.Quantity.zero_to_one)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    # The type of given metric, None or List of two elements, first element should be in availMetrics.keys()
    # and sencond element should be in availMetrics.values()[firstElement].keys()
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
        if len(self.metricType) != 2:
          self.raiseAnError(IOError, "Metric type: '", child.value, "' is not correct, please check the user manual for the correct metric type!")
      else:
        self.distParams[child.getName()] = child.value

    if self.metricType[0] not in self.__class__.availMetrics.keys() or self.metricType[1] not in self.__class__.availMetrics[self.metricType[0]].keys():
      self.raiseAnError(IOError, "Metric '", self.name, "' with metricType '", self.metricType[0], "|", self.metricType[1], "' is not valid!")

  def run(self, x, y, weights=None, axis=0, **kwargs):
    """
      This method computes difference between two points x and y based on given metric
      @ In, x, 1-D numpy.ndarray, array containing data of x.
      @ In, y, 1-D numpy.ndarray, array containing data of y.
      @ In, weights, array_like (numpy.array or list), optional, weights associated the metric method
      @ In, axis, integer, optional, default is 0, not used in this metric
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result
    """
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      assert x.shape == y.shape, "Input data x, y should have the same shape!"
      # TODO: weights are supported in scipy.spatial.distance for many distance metrics in v1.0.0
      # when we switch to scipy 1.0.0, we can enable weights in our metrics calculations
      sv = str(scipy.__version__).split('.')
      if int(sv[0]) > 0:
        if weights is not None and 'w' not in self.distParams.keys():
          self.distParams['w'] = weights
        # FIXME: In Scipy version 1.1.0, the function scipy.spatial.distance.canberra and
        # scipy.spatial.distance.sokalmichener will accept the weights, and the calculated results from
        # these functions will affected by the normalization of the weights. The following is disabled for
        # this purpose  --- wangc July 17, 2018
        # For future development, please pay attention to canberra, minkowski, and sokalmichener metrics
        #if 'w' in self.distParams.keys():
          # Normalized weights, since methods exist in Scipy are using unnormalized weights
          #self.distParams['w'] = np.asarray(self.distParams['w'])/np.sum(self.distParams['w'])
      else:
        if 'w' in self.distParams.keys():
          self.raiseAWarning("Weights will not be used, since weights provided with key word 'w' is not supported for your current version of scipy!")
          self.distParams.pop('w')
      dictTemp = utils.mergeDictionaries(kwargs, self.distParams)
      try:
        value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x, y, **dictTemp)
      except TypeError as e:
        self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type', self.metricType[1])
        self.raiseAnError(TypeError, 'Input parameters error: \n', str(e), '\n')
    else:
      self.raiseAnError(IOError, "Input data type is not correct!")

    return value
