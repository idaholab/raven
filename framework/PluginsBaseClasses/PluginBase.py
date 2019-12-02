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
Created on November 6, 2017

@author: alfoa
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
#Internal Modules End-----------------------------------------------------------

class PluginBase(object):
  """
    This class represents an abstract class each specialized plugin class should inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _methodsToCheck = []
  entityType = None  # the RAVEN entity fulfilled by this object

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """

  @classmethod
  def isAValidPlugin(cls):
    """
      Method to check if this plugin is a valid one (i.e. contains all the expected API method)
      @ In, None
      @ Out, validPlugIn, bool, is this plugin a valid one?
    """
    classMethods = [method for method in dir(cls) if callable(getattr(cls, method))]
    validPlugIn = set(cls._methodsToCheck) <= set(classMethods)
    return validPlugIn

