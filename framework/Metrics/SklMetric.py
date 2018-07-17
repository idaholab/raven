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
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import ast
from utils import utils
import sklearn
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# FIXME: median_absolute_error only accepts 1-D numpy array, and if we want to use this metric, it should
# be handled differently.
#from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
#Internal Modules End--------------------------------------------------------------------------------


class SKL(Metric):
  """
    Scikit-learn metrics
  """
  availMetrics ={}
  # regression metrics
  availMetrics['regression'] = {}
  availMetrics['regression']['explained_variance_score'] = explained_variance_score
  availMetrics['regression']['mean_absolute_error']      = mean_absolute_error
  availMetrics['regression']['r2_score']                 = r2_score
  availMetrics['regression']['mean_squared_error']       = mean_squared_error
  # paired distance metrics, no weights
  if int(sklearn.__version__.split(".")[1]) > 17:
    availMetrics['paired_distance'] = {}
    availMetrics['paired_distance']['euclidean']         = pairwise.paired_euclidean_distances
    availMetrics['paired_distance']['manhattan']         = pairwise.paired_manhattan_distances
    availMetrics['paired_distance']['cosine']            = pairwise.paired_cosine_distances
  # TODO: add more metrics here
  # metric from scipy.spatial.distance, for example mahalanobis, minkowski

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Metric.__init__(self)
    # The type of given metric, None or List of two elements, first element should be in availMetrics.keys()
    # and sencond element should be in availMetrics.values()[firstElement].keys()
    self.metricType = None
    # True indicates the metric needs to be able to handle dynamic data
    self._dynamicHandling = True

  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.distParams = {}
    for child in xmlNode:
      if child.tag == 'metricType':
        self.metricType = list(elem.strip() for elem in child.text.split('|'))
        if len(self.metricType) != 2:
          self.raiseAnError(IOError, "Metric type: '", child.text, "' is not correct, please check the user manual for the correct metric type!")
      else:
        self.distParams[str(child.tag)] = utils.tryParse(child.text)

    if self.metricType[0] not in self.__class__.availMetrics.keys() or self.metricType[1] not in self.__class__.availMetrics[self.metricType[0]].keys():
      self.raiseAnError(IOError, "Metric '", self.name, "' with metricType '", self.metricType[0], "|", self.metricType[1], "' is not valid!")

    if self.metricType[0] == 'paired_distance' and int(sklearn.__version__.split(".")[1]) < 18:
      self.raiseAnError(IOError, "paired_distance is not supported in your SciKit-Learn version, if you want to use this metric, please make sure your SciKit-Learn version >= 18!")

    for key, value in self.distParams.items():
      try:
        newValue = ast.literal_eval(value)
        if type(newValue) == list:
          newValue = np.asarray(newValue)
        self.distParams[key] = newValue
      except:
        self.distParams[key] = value

  def __evaluateLocal__(self, x, y, weights = None, axis = 0, **kwargs):
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
      @ In, axis, integer, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dictionary of parameters characteristic of each metric
      @ Out, value, numpy.ndarray, metric result, shape (n_outputs) if axis = 0, otherwise
        shape (n_samples), we assume the dimension of input numpy.ndarray is no more than 2.
    """
    #######################################################################################
    # The inputs of regression metric, i.e. x, y should have shape (n_samples, n_outputs),
    # and the outputs will have the shape (n_outputs).
    # However, the inputs of paired metric, i.e. x, y should convert the shape to
    # (n_outputs, n_samples), and the outputs will have the shape (n_outputs).
    #######################################################################################
    assert(isinstance(x,np.ndarray))
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
