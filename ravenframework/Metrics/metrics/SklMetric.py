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
Created on August 20 2016

@author: mandd
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import ast
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import utils
from .MetricInterface import MetricInterface
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class SKL(MetricInterface):
  """
    Scikit-learn metrics
  """
  availMetrics ={}

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
    inputSpecification.addSub(InputData.parameterInputFactory("sample_weight",contentType=InputTypes.FloatListType),quantity=InputData.Quantity.zero_to_one)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    if len(self.availMetrics) == 0:
      import sklearn
      import sklearn.metrics
      # FIXME: median_absolute_error only accepts 1-D numpy array, and if we want to use this metric, it should
      # be handled differently.
      #from sklearn.metrics import median_absolute_error

      # regression metrics
      self.availMetrics['regression'] = {}
      self.availMetrics['regression']['explained_variance_score'] = sklearn.metrics.explained_variance_score
      self.availMetrics['regression']['mean_absolute_error']      = sklearn.metrics.mean_absolute_error
      self.availMetrics['regression']['r2_score']                 = sklearn.metrics.r2_score
      self.availMetrics['regression']['mean_squared_error']       = sklearn.metrics.mean_squared_error
      # paired distance metrics, no weights
      self.availMetrics['paired_distance'] = {}
      self.availMetrics['paired_distance']['euclidean']         = sklearn.metrics.pairwise.paired_euclidean_distances
      self.availMetrics['paired_distance']['manhattan']         = sklearn.metrics.pairwise.paired_manhattan_distances
      self.availMetrics['paired_distance']['cosine']            = sklearn.metrics.pairwise.paired_cosine_distances
      # TODO: add more metrics here
      # metric from scipy.spatial.distance, for example mahalanobis, minkowski

    # The type of given metric, None or List of two elements, first element should be in availMetrics.keys()
    # and sencond element should be in availMetrics.values()[firstElement].keys()
    self.metricType = None
    # True indicates the metric needs to be able to handle dynamic data
    self._dynamicHandling = True

  def handleInput(self, paramInput):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
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
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1) for paired_distance, shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_outputs)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_outputs)
      @ In, weights, array_like (numpy.array or list), optional, weights associated
        with input, shape (n_samples) if axis = 0, otherwise shape (n_outputs)
      @ In, axis, integer, optional, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, numpy.ndarray, metric result, shape (n_outputs) if axis = 0, otherwise
        shape (n_samples), we assume the dimension of input numpy.ndarray is no more than 2.
    """
    #######################################################################################
    # The inputs of regression metric, i.e. x, y should have shape (n_samples, n_outputs),
    # and the outputs will have the shape (n_outputs).
    # However, the inputs of paired metric, i.e. x, y should convert the shape to
    # (n_outputs, n_samples), and the outputs will have the shape (n_outputs).
    #######################################################################################
    assert(isinstance(x,np.ndarray)) # NOTE these assertions will not show up for non-debug runs!
    assert(isinstance(y,np.ndarray))
    assert(x.shape == y.shape), "Input data x, y should have the same shape"
    if weights is not None and self.metricType[0] == 'regression' and 'sample_weight' not in self.distParams.keys():
      self.distParams['sample_weight'] = weights
    if self.metricType[0] == 'regression':
      self.distParams['multioutput'] = 'raw_values'
    dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
    if self.metricType[0] == 'paired_distance':
      if len(x.shape) == 1:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
      else:
        # Transpose is needed, since paired_distance is operated on the 'row'
        x = x.T
        y = y.T
    if axis == 1:
      x = x.T
      y = y.T
      # check the dimension of weights
      assert(x.shape[0] == len(weights)), "'weights' should have the same length of the first dimension of input data"
    elif axis != 0:
      self.raiseAnError(IOError, "Valid axis value should be '0' or '1' for the evaluate method of metric", self. name, "value", axis, "is provided!")
    try:
      value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x, y, **dictTemp)
    except TypeError as e:
      self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType[1], '"!')
      self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')

    return value
