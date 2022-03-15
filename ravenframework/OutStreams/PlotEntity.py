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
Created on April 1, 2021

@author: talbpaul
"""
import os
import matplotlib

from ..utils import utils, InputTypes
from .OutStreamEntity import OutStreamEntity
from .PlotInterfaces import factory as PlotFactory

# initialize display settings
display = utils.displayAvailable()
if not display:
  matplotlib.use('Agg')



class Plot(OutStreamEntity):
  """
    Handler for Plot implementations
  """
  interfaceFactory = PlotFactory
  defaultInterface = 'GeneralPlot'
  strictInput = False # GeneralPlot is not checked yet

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'PlotEntity'
    self._plotter = None            # implemention, inheriting from interface

  def _readMoreXML(self, xml):
    """
      Legacy passthrough.
      @ In, xml, xml.etree.ElementTree.Element, input
      @ Out, None
    """
    # TODO remove this when GeneralPlot conforms to inputParams
    subType = xml.attrib.get('subType', 'GeneralPlot').strip()
    if subType == 'GeneralPlot':
      self._plotter = self.interfaceFactory.returnInstance(subType)
      self._plotter.handleInput(xml)
    else:
      spec = self.parseXML(xml)
      self.handleInput(spec)

  def _handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super()._handleInput(spec)
    # we specialized out the GeneralPlot in _readMoreXML, so here we have a real inputParams user
    reqType = spec.parameterValues['subType']
    self._plotter = self.interfaceFactory.returnInstance(reqType)
    self._plotter.handleInput(spec)

  def _getInterface(self):
    """
      Return the interface associated with this entity.
      @ In, None
      @ Out, _getInterface, object, interface object
    """
    return self._plotter

  def initialize(self, stepEntities):
    """
      Initialize the OutStream. Initialize interfaces and pass references to sources.
      @ In, stepEntities, dict, the Entities used in the current Step. Sources are taken from this.
      @ Out, None
    """
    super().initialize(stepEntities)
    self._plotter.initialize(stepEntities)

  def addOutput(self):
    """
      Function to add a new output source (for example a CSV file or a HDF5
      object)
      @ In, None
      @ Out, None
    """
    self._plotter.run()

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
    paramDict.update(self._plotter.getInitParams())
    return paramDict

  ################
  # Plot Interactive
  def endInstructions(self, instructionString):
    """
      Method to execute instructions at end of a step (this is applied when an
      interactive mode is activated)
      @ In, instructionString, string, the instruction to execute
      @ Out, None
    """
    self._plotter.endInstructions(instructionString)
