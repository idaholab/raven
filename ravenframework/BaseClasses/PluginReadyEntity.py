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
Extends BaseEntity for commonly-used mechanics in plugins. Some of these may be generalizable to all
of RAVEN Entities when they are converted to the Entity/Interface model.

Created May 10, 2021
@author: talbpaul
"""

from abc import abstractmethod

import numpy as np

from ..utils import InputData, InputTypes
from .BaseEntity import BaseEntity

class PluginReadyEntity(BaseEntity):
  """
    Extends BaseEntity for Plugin-Ready Entities including:
    - Entity/Interface format
    - use of "subType" to specify specs (including plugins specs)
    - etc
  """
  # class members
  interfaceFactory = None # NOTE each PluginEntity must define their interfaceFactory here at the class level
  defaultInterface = None # If this Entity has a default Interface, its string name should go here
  strictInput = True      # True if this Entity is ready for strict checking of its input
  # -> strictInput should only be False for legacy purposes; full inputdata checking recommended

  @classmethod
  def getInputSpecification(cls, xml=None):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, xml, xml.etree.ElementTree.Element, optional, if given then only get specs for
          corresponding subType requested by the node
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
          specifying input of cls.
    """
    # OVERLOADED in order to take XML as optional input, for loading specific types instead of
    # all possible types
    assert cls.interfaceFactory is not None
    spec = super().getInputSpecification()
    subTypeIsRequired = bool(cls.defaultInterface is None)
    if xml is None:
      # if specific loading XML is not available, load all options (usually for manual)
      okTypes = list(cls.interfaceFactory.knownTypes())
      okEnum = InputTypes.makeEnumType(cls.__name__, cls.__name__, okTypes)
      spec.addParam('subType', required=subTypeIsRequired, param_type=okEnum,
          descr=rf"""Type of {cls.__name__} to generate""")
      spec.strictMode = cls.strictInput
    else:
      # otherwise, load only specific interface related to XML request
      itfName = xml.attrib.get('subType', cls.defaultInterface)
      itf = cls.interfaceFactory.returnClass(itfName)
      spec.addParam('subType', required=subTypeIsRequired, param_type=InputTypes.StringType)
      itfSpecs = itf.getInputSpecification()
      spec.mergeSub(itfSpecs)
    return spec

  def parseXML(self, xml):
    """
      Parse XML into input parameters
      Overloaded to pass XML to getInputSpecifications
      @ In, xml, xml.etree.ElementTree.Element, XML element node
      @ Out, InputData.ParameterInput, the parsed input
    """
    # OVERLOADED in order to provide XML argument to getInputSpecification, for specific type
    paramInput = self.getInputSpecification(xml=xml)()
    paramInput.parseNode(xml)
    return paramInput

  @abstractmethod
  def _getInterface(self):
    """
      Return the interface associated with this entity.
      @ In, None
      @ Out, _getInterface, object, interface object
    """

  ########
  #
  # Utilities
  #
  def isInstanceString(self, toCheck):
    """
      Use string type names to check if associated metric is one of those whose names are provided.
      @ In, toCheck, list, list of names of viable types
      @ Out, isInstanceString, bool, True if interface name matches one of the provided options
    """
    viable = tuple([self.interfaceFactory.returnClass(check) for check in np.atleast_1d(toCheck)])
    return isinstance(self._getInterface(), viable)

  @property
  def interfaceKind(self):
    """
      Provide the "type" of the interface for this Entity.
      @ In, None
      @ Out, interfaceType, str, name of type for this Entity's Interface
    """
    return self._getInterface().__class__.__name__
