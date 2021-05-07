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

from utils import utils, InputTypes
from .OutStreamEntity import OutStreamEntity
from .PlotInterfaces import factory as interfaceFactory

# initialize display settings
display = utils.displayAvailable()
if not display:
  matplotlib.use('Agg')



class Plot(OutStreamEntity):
  """
    Handler for Plot implementations
  """
  @classmethod
  def getInputSpecification(cls, xml=None):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, xml, xml.etree.ElementTree.Element, optional, if given then only get specs for
          corresponding subType requested by the node
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification() # TODO add xml arg when generalizing
    if xml is None:
      # generic definition; collect all known options
      # this is used for e.g. documentation
      okTypes = list(interfaceFactory.knownTypes())
      okEnum = InputTypes.makeEnumType('OutStreamPlot', 'OutStreamPlotType', okTypes)
      spec.addParam('subType', required=False, param_type=okEnum, descr=r"""Type of OutStream Plot to generate.""")
      # TODO add specs depending on the one chosen, not all of them!
      # FIXME the GeneralPlot has a vast need for converting to input specs. Until then,
      #       we cannot strictly check anything related to it.
      spec.strictMode = False
      for name in okTypes:
        plotter = interfaceFactory.returnClass(name)
        subSpecs = plotter.getInputSpecification()
        spec.mergeSub(subSpecs)
    else:
      # this is used when the subType has already been specified
      # e.g. when reading an XML file
      itfName = xml.attrib.get('subType', 'GeneralPlot')
      itf = interfaceFactory.returnClass(itfName)
      spec.addParam('subType', required=False, param_type=InputTypes.StringType)
      itfSpecs = itf.getInputSpecification()
      spec.mergeSub(itfSpecs)
    return spec

  def parseXML(self, xml):
    """
      Parse XML into input parameters
      Overloaded to pass XML to getInputSpecifications
      -> this should be commonly done among Entities, probably.
      @ In, xml, xml.etree.ElementTree.Element, XML element node
      @ Out, InputData.ParameterInput, the parsed input
    """
    paramInput = self.getInputSpecification(xml=xml)()
    paramInput.parseNode(xml)
    return paramInput


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
      self._plotter = interfaceFactory.returnInstance(subType)
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
    self._plotter = interfaceFactory.returnInstance(reqType)
    self._plotter.handleInput(spec)

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
