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
#from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
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
  availMetrics['regression']['mean_absolute_error'] = mean_absolute_error
  availMetrics['regression']['r2_score'] = r2_score
  availMetrics['regression']['mean_squared_error'] = mean_squared_error
  availMetrics['regression']['mean_squared_log_error'] = mean_squared_log_error
  #availMetrics['regression']['median_absolute_error'] = mean_absolute_error # disabled because this metric only accept 1D array, and not weights associated
  # paired distance metrics, no weights
  if int(sklearn.__version__.split(".")[1]) > 17:
    availMetrics['paired_distance'] = {}
    availMetrics['paired_distance']['euclidean'] = pairwise.paired_euclidean_distances
    availMetrics['paired_distance']['manhattan'] = pairwise.paired_manhattan_distances
    availMetrics['paired_distance']['cosine'] = pairwise.paired_cosine_distances
  # TODO: add more metrics here
  # metric from scipy.spatial.distance, for example mahalanobis, minkowski

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, NOne
    """
    Metric.__init__(self)
    self.metricType = None
    self._dynamicHandling = True
    self.acceptsProbability = True

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
          self.raiseAnError(IOError, "Metric type: '", child.tag, "' is not correct, please check the user manual for the correct metric type!")
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

  def __evaluateLocal__(self, x, y, weights = None, **kwargs):
    """
      This method compute difference between two points x and y based on given metric
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided, the array will be reshaped via x.reshape(1,-1)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided, the array will be reshaped via y.reshape(1,-1)
      @ In, weights, None or array_like (numpy.array or list), weights associated with input
      @ Out, value, numpy.ndarray, metric result
    """
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      assert(x.shape == y.shape, "Input data x, y should have the same shape")
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
          x = x.T
          y = y.T
      try:
        value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x, y, **dictTemp)
      except TypeError as e:
        self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType[1], '"!')
        self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')
    else:
      self.raiseAnError(IOError,'Input data type is not correct!')

    return value
