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
Created on April 2, 2021

@author: talbpaul
"""

from ..utils import InputTypes
from .OutStreamEntity import OutStreamEntity
from .PrintInterfaces import factory as PrintFactory


class Print(OutStreamEntity):
  """
    Handler for Plot implementations
  """
  interfaceFactory = PrintFactory
  defaultInterface = 'FilePrint'

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'PrintEntity'
    self._printer = None            # implemention, inheriting from interface

  def _readMoreXML(self, xml):
    """
      Legacy passthrough.
      @ In, xml, xml.etree.ElementTree.Element, input
      @ Out, None
    """
    spec = self.parseXML(xml)
    self.handleInput(spec)

  def _handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super()._handleInput(spec)
    reqType = spec.parameterValues.get('subType', 'FilePrint')
    self._printer = self.interfaceFactory.returnInstance(reqType)
    self._printer.handleInput(spec)

  def _getInterface(self):
    """
      Return the interface associated with this entity.
      @ In, None
      @ Out, _getInterface, object, interface object
    """
    return self._printer

  def initialize(self, stepEntities):
    """
      Initialize the OutStream. Initialize interfaces and pass references to sources.
      @ In, stepEntities, dict, the Entities used in the current Step. Sources are taken from this.
      @ Out, None
    """
    super().initialize(stepEntities)
    self._printer.initialize(stepEntities)

  def addOutput(self):
    """
      Function to add a new output source (for example a CSV file or a HDF5
      object)
      @ In, None
      @ Out, None
    """
    self._printer.run()

  ################
  # Utility
  def getInitParams(self):
    """
      This function is called from the base class to print some of the
      information inside the class. Whatever is permanent in the class and not
      inherited from the parent class should be mentioned here. The information
      is passed back in the dictionary. No information about values that change
      during the simulation are allowed.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = super().getInitParams()
    paramDict.update(self._printer.getInitParams())
    return paramDict
