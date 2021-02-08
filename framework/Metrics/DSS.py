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
Created on December 20 2020

@author: yoshrk
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
#import scipy.spatial.distance as spatialDistance
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class DSS(Metric):
  """
    Dynamical System Scaling Metric
    Class for measuring the metric between two instantaneous data points at the same (or approximate) process time
    @ Data Synthesis or Engineering Scaling.
    @ Data Synthesis is the act to measure the metric between data sets that are presumed to be equivalent in phenomena,
    in reference time, and initial/boundary conditions. Essentially, both data sets are expected to be equivalent.
    For the sake of data comparison with no intentions to match data but only to measure the metric, this model may be
    used as well.
    @ For engineering scaling, although the phenomena are equivalent, the reference time and initial/boundary conditions
    are different. Both sets are tied by process time only.
    @ In either case, if the to be compared data sets are are of little relevance, it is most likely DSS will fail to
    measure the metric distance accurately.
  """
  availScaling ={}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(DSS, cls).getInputSpecification()
    actionTypeInput = InputData.parameterInputFactory("actionType", contentType=InputTypes.StringType)
    inputSpecification.addSub(actionTypeInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    Metric.__init__(self)
    # The type of given analysis
    self.actionType                      = None
    # True indicates the metric needs to be able to handle dynamic data
    self._dynamicHandling = True
    # True indicates the metric needs to be able to handle pairwise data
    self._pairwiseHandling = False

  def _localReadMoreXML(self, xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = DSS.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == "actionType":
        self.order = child.value
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")

  def __evaluateLocal__(self, x, y, weights = None, axis = 0, **kwargs):
    """
      This method computes DSS distance between two inputs x and y based on given metric
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_time_steps)
      @ In, weights, array_like (numpy.array or list), optional, weights associated
        with input, shape (n_samples) if axis = 0, otherwise shape (n_time_steps)
      @ In, axis, integer, optional, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result
    """
    assert (isinstance(x, np.ndarray))
    assert (isinstance(y, np.ndarray))
    tempX = x
    tempY = y
    omegaNormTarget = tempX[0]
    omegaNormScaledFeature = tempY[0]
    pTime = tempX[1]
    D = tempY[1]
    betaFeature = tempX[2]
    betaTarget = tempX[2]
    distance = np.zeros((pTime.shape))
    distanceSum = np.zeros((pTime.shape[0]))
    sigma = np.zeros((pTime.shape[0]))
    for cnt in range(len(pTime)):
      distanceSquaredSum = 0
      for cnt2 in range(len(pTime[cnt])):
        distance[cnt][cnt2] = betaTarget[cnt][cnt2]*abs(D[cnt][cnt2])**0.5*(1/omegaNormTarget[cnt][cnt2]-1/omegaNormScaledFeature[cnt][cnt2])
        distanceSum[cnt] += abs(distance[cnt][cnt2])
        distanceSquaredSum += distance[cnt][cnt2]**2
      sigma[cnt] = (1/len(sigma)*distanceSquaredSum)**0.5
    value = distanceSum
    #value = [pTime,betaTarget,omegaNormScaledFeature,omegaNormTarget,D,distance,distanceSum,sigma]
    return value