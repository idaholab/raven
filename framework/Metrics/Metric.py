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

@author: mandd, wangc
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

class Metric(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
    This is the general interface to any RAVEN metric object.
    It contains an initialize, a _readMoreXML, and an evaluation (i.e., distance) methods
  """
  def __init__(self):
    """
      This is the basic method initialize the metric object
      @ In, none
      @ Out, none
    """
    BaseType.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__
    self.acceptsProbability  = False #If True the metric needs to be able to handle (value,probability) where value and probability are lists
    self.acceptsDistribution = False #If True the metric needs to be able to handle a passed in Distribution
    self._dynamicHandling    = False #If True the metric needs to be able to handle dynamic data

  def initialize(self, inputDict):
    """
      This method initialize each metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    pass

  def _readMoreXML(self, xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self._localReadMoreXML(xmlNode)

  def evaluate(self, x, y, weights = None, **kwargs):
    """
      This method compute the metric between x and y
      @ In, x, numpy.array, array containing data of x
      @ In, y, numpy.array, array containing data of y
      @ In, weights, None or numpy.array, an array of weights associated with x
      @ In, kwargs, dictionary of parameters characteristic of each metric
      @ Out, value, float or numpy.array, metric results between x and y
    """
    value = self.__evaluateLocal__(x, y, weights=weights, **kwargs)

    return value

  def isDynamic(self):
    """
      This method is utility function that tells if the metric is able to
      treat dynamic data on its own or not
      @ In, None
      @ Out, isDynamic, bool, True if the metric is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  @abc.abstractmethod
  def __evaluateLocal__(self, x, y, weights = None, **kwargs):
    """
      This method compute the metric between x and y
      @ In, x, numpy.array, array containing data of x
      @ In, y, numpy.array, array containing data of y
      @ In, weights, None or numpy.array, an array of weights associated with x
      @ In, kwargs, dictionary of parameters characteristic of each metric
      @ Out, value, float or numpy.array, metric results between x and y
    """
