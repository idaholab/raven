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
Created on April 29 2021

@author: Mohammad Abdo (@Jimmy-INL)
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

class RepresentativityFactors(Metric):
  """
    RepresntativityFactors is the metric class used to quantitatively
    assess the relativeness of a mock experiment to the target plant.
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
    inputSpecification = super(RepresntativityFactors, cls).getInputSpecification()
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
    paramInput = Metric.getInputSpecification()()
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
    # assert (isinstance(x, np.ndarray))
    # assert (isinstance(y, np.ndarray))
    senMeasurables = kwargs['senMeasurables']
    senFOMs = kwargs['senFOMs']
    covParameters = kwargs['covParameters']
    r = (senFOMs @ covParameters @ senMeasurables.T)/\
        np.sqrt(senFOMs @ covParameters @ senFOMs.T)/\
        np.sqrt(senMeasurables @ covParameters @ senMeasurables.T)
    return r