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
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric
import Metrics.MetricUtilities
#Internal Modules End--------------------------------------------------------------------------------

class PDFCommonArea(Metric):
  """
    Metric to compare two datasets using the PDF Common Area.
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, NOne
    """
    Metric.__init__(self)
    self.acceptsProbability = True
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
      @ In, x, numpy.ndarray, array containing data of x, if 1D array is provided,
        the array will be reshaped via x.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_outputs)
      @ In, y, numpy.ndarray, array containing data of y, if 1D array is provided,
        the array will be reshaped via y.reshape(-1,1), shape (n_samples, ), if 2D
        array is provided, shape (n_samples, n_outputs)
      @ In, weights, None or array_like (numpy.array or list), optional weights associated
        with input, shape (n_samples) if axis = 0, otherwise shape (n_outputs)
      @ In, axis, integer, axis along which a metric is performed, default is 0,
        i.e. the metric will performed along the first dimension (the "rows").
        If metric postprocessor is used, the first dimension is the RAVEN_sample_ID,
        and the second dimension is the pivotParameter if HistorySet is provided.
      @ In, kwargs, dictionary of parameters characteristic of each metric
      @ Out, value, numpy.ndarray, metric result, shape (n_outputs) if axis = 0, otherwise
        shape (n_samples), we assume the dimension of input numpy.ndarray is no more than 2.
    """
    """
      Calculates the PDF Common Area between two datasets.
      @ In, x, something that can be converted into a PDF
      @ In, y, something that can be converted into a PDF
      @ In, kwargs, ignored.
      @ Out, value, float, CDF Area Difference
    """
    value = Metrics.MetricUtilities._getPDFCommonArea(x,y)
    return float(value)
