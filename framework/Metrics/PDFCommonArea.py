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
  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.acceptsProbability = True
    self.acceptsDistribution = True

  def distance(self,x,y,**kwargs):
    """
      Calculates the PDF Common Area between two datasets.
      @ In, x, something that can be converted into a PDF
      @ In, y, something that can be converted into a PDF
      @ In, kwargs, ignored.
      @ Out, value, float, CDF Area Difference
    """
    value = Metrics.MetricUtilities._getPDFCommonArea(x,y)
    return float(value)
