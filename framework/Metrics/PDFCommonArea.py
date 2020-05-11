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
Created on 2017 September 12

@author: Joshua Cogliati
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
import Metrics.MetricUtilities
#Internal Modules End--------------------------------------------------------------------------------

class PDFCommonArea(Metric):
  """
    Metric to compare two datasets using the PDF Common Area.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(PDFCommonArea, cls).getInputSpecification()

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, NOne
    """
    Metric.__init__(self)
    # If True the metric needs to be able to handle (value,probability) where value and probability are lists
    self.acceptsProbability = True
    # If True the metric needs to be able to handle a passed in Distribution
    self.acceptsDistribution = True

  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    pass

  def __evaluateLocal__(self, x, y, weights = None, axis = 0, **kwargs):
    """
      This method computes difference between two points x and y based on given metric
      @ In, x, instance of Distributions.Distribution, tuple or list, array containing data of x,
        or given distribution.
      @ In, y, instance of Distributions.Distribution, tuple or list, array containing data of y,
        or given distribution.
      @ In, weights, array_like (numpy.array or list), optional, not used in this metric
      @ In, axis, integer, default is 0, not used for this metric.
      @ In, kwargs, dict, dictionary of parameters characteristic of each metric
      @ Out, value, float, metric result, PDF common area
    """
    value = Metrics.MetricUtilities._getPDFCommonArea(x,y)
    return float(value)
