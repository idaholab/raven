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

#External Modules---------------------------------------------------------------
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PluginBase import PluginBase
#Internal Modules End-----------------------------------------------------------

class DummyFactory:
  """ passthrough until ExternalModels work like other plugins """
  def registerType(self, *args):
    """
      dummy
      @ In, args, list, arguments
      @ Out, None
    """
    pass

class ExternalModelPluginBase(PluginBase):
  """
    This class represents a specialized class from which each ExternalModel plugins must inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _interfaceFactory = DummyFactory()
  _methodsToCheck = ['run','initialize']
  entityType = 'ExternalModel' # should this just be Model?

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    PluginBase.__init__(self)
    # FIXME the ExternalModelPluginBase acts strangely compared to most classes, since it was the
    # first one we tried. It should be modified to work like the others.

