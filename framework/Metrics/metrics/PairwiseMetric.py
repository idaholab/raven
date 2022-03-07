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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import scipy
import ast
import scipy.spatial.distance as spatialDistance
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import utils
from .MetricInterface import MetricInterface
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


class PairwiseMetric(MetricInterface):
  """
    Scikit-learn pairwise metrics
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
    inputSpecification.addSub(InputData.parameterInputFactory("degree",contentType=InputTypes.IntegerType),quantity=InputData.Quantity.zero_to_one)
    inputSpecification.addSub(InputData.parameterInputFactory("gamma",contentType=InputTypes.FloatType),quantity=InputData.Quantity.zero_to_one)
    inputSpecification.addSub(InputData.parameterInputFactory("coef0",contentType=InputTypes.IntegerType),quantity=InputData.Quantity.zero_to_one)
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
      import sklearn.metrics.pairwise

      # pairwised kernel metrics
      self.availMetrics['kernel'] = {}
      self.availMetrics['kernel']['additive_chi2']       = sklearn.metrics.pairwise.additive_chi2_kernel
      self.availMetrics['kernel']['chi2']                = sklearn.metrics.pairwise.chi2_kernel
      self.availMetrics['kernel']['cosine_similarity']   = sklearn.metrics.pairwise.cosine_similarity
      self.availMetrics['kernel']['laplacian']           = sklearn.metrics.pairwise.laplacian_kernel
      self.availMetrics['kernel']['linear']              = sklearn.metrics.pairwise.linear_kernel
      self.availMetrics['kernel']['polynomial']          = sklearn.metrics.pairwise.polynomial_kernel
      self.availMetrics['kernel']['rbf']                 = sklearn.metrics.pairwise.rbf_kernel
      self.availMetrics['kernel']['sigmoid']             = sklearn.metrics.pairwise.sigmoid_kernel
      # pairwised distance metrices
      self.availMetrics['pairwise'] = {}
      self.availMetrics['pairwise']['euclidean']         = sklearn.metrics.pairwise.euclidean_distances
      self.availMetrics['pairwise']['manhattan']         = sklearn.metrics.pairwise.manhattan_distances
      # pairwised metrics from scipy
      self.availMetrics['pairwise']['minkowski']         = None
      self.availMetrics['pairwise']['mahalanobis']       = None
      self.availMetrics['pairwise']['braycurtis']        = None
      self.availMetrics['pairwise']['canberra']          = None
      self.availMetrics['pairwise']['chebyshev']         = None
      self.availMetrics['pairwise']['correlation']       = None
      self.availMetrics['pairwise']['dice']              = None
      self.availMetrics['pairwise']['hamming']           = None
      self.availMetrics['pairwise']['jaccard']           = None
      self.availMetrics['pairwise']['kulsinki']          = None
      self.availMetrics['pairwise']['matching']          = None
      self.availMetrics['pairwise']['rogerstanimoto']    = None
      self.availMetrics['pairwise']['russellrao']        = None
      self.availMetrics['pairwise']['sokalmichener']     = None
      self.availMetrics['pairwise']['sokalsneath']       = None
      self.availMetrics['pairwise']['yule']              = None

    self.metricType = None
    # True indicates the metric needs to be able to handle pairwise data
    self._pairwiseHandling = True

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

  def run(self, x, y=None, axis=0, weights=None, **kwargs):
    """
      This method computes difference between two points x and y based on given metric
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1)
      @ In, y, optional, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1)
      @ In, axis, integer, 0 or 1, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows"). If axis = 1,
        then x = x.T, y = y.T
      @ In, weights, array_like (numpy.ndarray or list), optional, unused in this method
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, 2D numpy.ndarray, metric result, shape (numRowsInX, numRowsInY) if y is not
        None, otherwise shape (numRowsInX, numRowsInX)
    """
    if axis != 0 and axis != 1:
      self.raiseAnError(IOError, "Acceptable values for axis are 0 or 1, but the provided value is", axis)
    assert(isinstance(x,np.ndarray)), "Input data x should be numpy.array"
    if len(x.shape) == 1:
      x = x.reshape(-1,1)
    if axis == 1:
      x = x.T
    if y != None:
      assert(isinstance(y,np.ndarray)), "Input data y should be numpy.array"
      if len(y.shape) == 1:
        y = y.reshape(-1,1)
      if axis == 1:
        y = y.T
      assert(x.shape[1] == y.shape[1]), "The number of columns in x should be the same as the number of columns in y"
    dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
    # set up the metric engine if it is None
    if self.__class__.availMetrics[self.metricType[0]][self.metricType[1]] == None:
      if y == None:
        self.__class__.availMetrics[self.metricType[0]][self.metricType[1]] = spatialDistance.pdist
        try:
          value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x,metric=self.metricType[1], **dictTemp)
        except TypeError as e:
          self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType[1], '"!')
          self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')
      else:
        self.__class__.availMetrics[self.metricType[0]][self.metricType[1]] = spatialDistance.cdist
        try:
          value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x,y,metric=self.metricType[1], **dictTemp)
        except TypeError as e:
          self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType[1], '"!')
          self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')
    else:
      try:
        value = self.__class__.availMetrics[self.metricType[0]][self.metricType[1]](x, Y=y, **dictTemp)
      except TypeError as e:
        self.raiseAWarning('There are some unexpected keyword arguments found in Metric with type "', self.metricType[1], '"!')
        self.raiseAnError(TypeError,'Input parameters error:\n', str(e), '\n')

    return value
