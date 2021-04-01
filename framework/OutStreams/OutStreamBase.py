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
Created on Nov 14, 2013

@author: alfoa
"""
import os

from BaseClasses import BaseEntity
from utils import InputTypes

class OutStreamBase(BaseEntity):
  """
    OUTSTREAM CLASS
    This class is a general base class for outstream action classes
    For example, a matplotlib interface class or Print class, etc.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addParam('dir', param_type=InputTypes.StringType, required=False)
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._interface = None    # interface implementation handled by this Entity
    self.type = 'Base'        # identifying type

    # outstreaming options
    self.options = {}
    # counter
    self.counter = 0
    # overwrite outstream?
    self.overwrite = True
    # outstream types available
    self.availableOutStreamType = []
    # number of agregated outstreams
    self.numberAggregatedOS = 1
    # optional sub directory for printing and plotting
    self.subDirectory = None
    self.printTag = 'OUTSTREAM MANAGER'
    self.filename = ''

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the input that belongs to this
      specialized class and initialize based on the inputs received
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node
      @ Out, None
    """
    from .GeneralPlot import GeneralPlot # can't import earlier without circular reference
    # legacy interface still uses raw xml
    if isinstance(self._interface, GeneralPlot):
      spec = xmlNode
    else:
      spec = self.getInputSpecification()()
      spec.parseNode(xmlNode)
    self._interface.handleInput(spec)

  def _handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    pass

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
    paramDict = {}
    paramDict['Global Class Type                 '] = 'OutStreamManager'
    paramDict['Specialized Class Type            '] = self.type
    if self.overwrite:
      paramDict['Overwrite output everytime called '] = 'True'
    else:
      paramDict['Overwrite output everytime called '] = 'False'
    for index in range(len((self.availableOutStreamType))):
      paramDict['OutStream Available #' + str(index + 1) + '   :'] = self.availableOutStreamType[index]
    paramDict.update(self.localGetInitParams())
    return paramDict

  def addOutput(self):
    """
      Function to add a new output source (for example a CSV file or a HDF5
      object)
      @ In, None
      @ Out, None
    """
    self.raiseAnError(NotImplementedError, 'method addOutput must be implemented by derived classes!!!!')

  def initialize(self, inDict):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, inDict, dict, contains all the Objects are going to be used in the
      current step. The sources are searched into this.
      @ Out, None
    """
    if self.subDirectory is not None:
      if not os.path.exists(self.subDirectory):
        os.makedirs(self.subDirectory)

    self.sourceData = []
    for agrosindex in range(self.numberAggregatedOS):
      foundData = False
      for output in inDict['Output']:
        if output.name.strip() == self.sourceName[agrosindex] and output.type in DataObjects.factory.knownTypes():
          self.sourceData.append(output)
          foundData = True
      if not foundData:
        for inp in inDict['Input']:
          if not type(inp) == type(""):
            if inp.name.strip() == self.sourceName[agrosindex] and inp.type in DataObjects.factory.knownTypes():
              self.sourceData.append(inp)
              foundData = True
            elif type(inp) == Models.ROM:
              self.sourceData.append(inp)
              foundData = True  # good enough
      if not foundData and 'TargetEvaluation' in inDict.keys():
        if inDict['TargetEvaluation'].name.strip() == self.sourceName[agrosindex] and inDict['TargetEvaluation'].type in DataObjects.factory.knownTypes():
          self.sourceData.append(inDict['TargetEvaluation'])
          foundData = True
      if not foundData and 'SolutionExport' in inDict.keys():
        if inDict['SolutionExport'].name.strip() == self.sourceName[agrosindex] and inDict['SolutionExport'].type in DataObjects.factory.knownTypes():
          self.sourceData.append(inDict['SolutionExport'])
          foundData = True
      if not foundData:
        self.raiseAnError(IOError, 'the DataObject "{data}" was not found among the "<Input>" nodes for this step!'.format(data = self.sourceName[agrosindex]))
