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
import sklearn.metrics.pairwise as pairwise
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
#Internal Modules End--------------------------------------------------------------------------------

scores = {'explained_variance_score':explained_variance_score,
          'mean_absolute_error':mean_absolute_error,
          'r2_score':r2_score,
          'mean_squared_error':mean_squared_error,
          'median_absolute_error':median_absolute_error}

class SKL(Metric):
  """
    Scikit-learn metrics which can be employed only for PointSets
  """
  def initialize(self,inputDict):
    """
      This method initializes the metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self.metricType = None

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
        self.metricType = child.text
      else:
        self.distParams[str(child.tag)] = utils.tryParse(child.text)
    availableMetrics = pairwise.kernel_metrics().keys() + pairwise.distance_metrics().keys() + scores.keys()
    if self.metricType not in availableMetrics:
      metricList = ', '.join(availableMetrics[:-1]) + ', or ' + availableMetrics[-1]
      self.raiseAnError(IOError,'Metric SKL error: metricType ' + str(self.metricType) + ' is not available. Available metrics are: ' + metricList + '.')

    for key, value in self.distParams.items():
      try:
        newValue = ast.literal_eval(value)
        if type(newValue) == list:
          newValue = np.asarray(newValue)
        self.distParams[key] = newValue
      except:
        self.distParams[key] = value

  def distance(self, x, y=None, **kwargs):
    """
      This method returns the distance between two points x and y. If y is not provided then x is a pointSet and a distance matrix is returned
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided, the array will be reshaped via x.reshape(1,-1)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided, the array will be reshaped via y.reshape(1,-1)
      @ Out, value, numpy.ndarray, distance between x and y (if y is provided) or a square distance matrix if y is None
    """
    if y is not None:
      if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
        if len(x.shape) == 1 and self.metricType not in scores.keys():
          x = x.reshape(1,-1)
          #self.raiseAWarning(self, "1D array is provided. For consistence, this array is reshaped via x.reshape(1,-1) ")
        if len(y.shape) == 1 and self.metricType not in scores.keys():
          y = y.reshape(1,-1)
          #self.raiseAWarning(self, "1D array is provided. For consistence, this array is reshaped via y.reshape(1,-1) ")
        dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
        try:
          if self.metricType in pairwise.kernel_metrics().keys():
            value = pairwise.pairwise_kernels(X=x, Y=y, metric=self.metricType, **dictTemp)
          elif self.metricType in pairwise.distance_metrics():
            value = pairwise.pairwise_distances(X=x, Y=y, metric=self.metricType, **dictTemp)
          elif self.metricType in scores.keys():
            value = np.zeros((1,1))
            value[:,:] = scores[self.metricType](x,y,**dictTemp)
        except TypeError as e:
          self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType, '"!')
          self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')
        if value.shape == (1,1):
          return value[0]
        else:
          return value
      else:
        self.raiseAnError(IOError,'Metric SKL error: SKL metrics support only PointSets and not HistorySets')
    else:
      if self.metricType == 'mahalanobis':
        covMAtrix = np.cov(x.T)
        kwargs['VI'] = np.linalg.inv(covMAtrix)
      dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
      try:
        if self.metricType in pairwise.kernel_metrics().keys():
          value = pairwise.pairwise_kernels(X=x, metric=self.metricType, **dictTemp)
        elif self.metricType in pairwise.distance_metrics().keys():
          value = pairwise.pairwise_distances(X=x, metric=self.metricType, **dictTemp)
      except TypeError as e:
        self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType, '"!')
        self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')
      if value.shape == (1,1):
        return value[0]
      else:
        return value
