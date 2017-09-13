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
import PostProcessors.ComparisonStatisticsModule
#Internal Modules End--------------------------------------------------------------------------------

class CDFAreaDifference(Metric):
  """
    Metric to compare two datasets using the CDF Area Difference.
  """
  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.acceptsProbability = True

  def distance(self,x,y,**kwargs):
    """
      Calculates the CDF Area Difference between two datasets.
      @ In, x, something that can be converted into a CDF
      @ In, y, something that can be converted into a CDF
      @ Out, value, float, CDF Area Difference
    """
    value = PostProcessors.ComparisonStatisticsModule._getCDFAreaDifference(x,y)
    return float(value)
