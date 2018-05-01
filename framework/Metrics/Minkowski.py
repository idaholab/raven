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
Created on Jul 18 2016

@author: mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import math
import numpy as np
import scipy.spatial.distance as spDist
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric

#Internal Modules End--------------------------------------------------------------------------------


class Minkowski(Metric):
  """
    Minkowski metrics which can be employed for both pointSets and historySets
  """

  def initialize(self, inputDict):
    """
      This method initialize the metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self.p = None
    self.pivotParameter = None

  def _localReadMoreXML(self, xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'p':
        self.p = float(child.text)
      if child.tag == 'pivotParameter':
        self.pivotParameter = child.text

  def distance(self, x, y):
    """
      This method actually calculates the distance between two dataObects x and y
      @ In, x, dict, dictionary containing data of x
      @ In, y, dict, dictionary containing data of y
      @ Out, value, float, distance between x and y
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
      value = spDist.minkowski(x, y, self.p)
      return value
    elif isinstance(x, dict) and isinstance(y, dict):
      if self.pivotParameter == None:
        self.raiseAnError(
            IOError,
            'The Minkowski metrics is being used on a historySet without the parameter pivotParameter being specified'
        )
      if x.keys() == y.keys():
        value = 0
        for key in x.keys():
          if x[key].size == y[key].size:
            if key != self.pivotParameter:
              value += spDist.minkowski(x[key], y[key], self.p)
            value = math.pow(value, 1.0 / self.p)
            return value
          else:
            print('Metric Minkowski error: the length of the variable array ' + str(key) +
                  ' is not consistent among the two data sets')
      else:
        print('Metric Minkowski error: the two data sets do not contain the same variables')
    else:
      print('Metric Minkowski error: the structures of the two data sets are different')
