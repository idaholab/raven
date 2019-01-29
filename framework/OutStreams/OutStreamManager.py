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
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3-------------------------------------------

#External Modules---------------------------------------------------------------
import os
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from BaseClasses import BaseType
import DataObjects
import Models
from utils import utils
#Internal Modules End-----------------------------------------------------------

class OutStreamManager(BaseType):
  """
    OUTSTREAM CLASS
    This class is a general base class for outstream action classes
    For example, a matplotlib interface class or Print class, etc.
  """
  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    BaseType.__init__(self)

    ## Use the class name as the type, so as we extend this class, this is
    ## automatically updated to be the correct value. Honestly, we shouldn't
    ## need this as we can just reference the class name wherever this is used.
    ## Otherwise, if you don't agree with that sentiment, then this should at
    ## least propogate itself up the hierarchy
    self.type = self.__class__.__name__

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
      Function to read the portion of the xml input that belongs to this
      specialized class and initialize some stuff based on the inputs received
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
      The text is supposed to contain the info where and which variable to change.
      In case of a code the syntax is specified by the code interface itself
    """
    if 'overwrite' in xmlNode.attrib.keys():
      if xmlNode.attrib['overwrite'].lower() in utils.stringsThatMeanTrue():
        self.overwrite = True
      else:
        self.overwrite = False
    if 'dir' in xmlNode.attrib:
      self.subDirectory =  xmlNode.attrib['dir']
      if '~' in self.subDirectory:
        self.subDirectory= os.path.expanduser(self.subDirectory)

    self.localReadXML(xmlNode)


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
        if output.name.strip() == self.sourceName[agrosindex] and output.type in DataObjects.knownTypes():
          self.sourceData.append(output)
          foundData = True
      if not foundData:
        for inp in inDict['Input']:
          if not type(inp) == type(""):
            if inp.name.strip() == self.sourceName[agrosindex] and inp.type in DataObjects.knownTypes():
              self.sourceData.append(inp)
              foundData = True
            elif type(inp) == Models.ROM:
              self.sourceData.append(inp)
              foundData = True  # good enough
      if not foundData and 'TargetEvaluation' in inDict.keys():
        if inDict['TargetEvaluation'].name.strip() == self.sourceName[agrosindex] and inDict['TargetEvaluation'].type in DataObjects.knownTypes():
          self.sourceData.append(inDict['TargetEvaluation'])
          foundData = True
      if not foundData and 'SolutionExport' in inDict.keys():
        if inDict['SolutionExport'].name.strip() == self.sourceName[agrosindex] and inDict['SolutionExport'].type in DataObjects.knownTypes():
          self.sourceData.append(inDict['SolutionExport'])
          foundData = True
      if not foundData:
        self.raiseAnError(IOError, 'the DataObject "{data}" was not found among the "<Input>" nodes for this step!'.format(data = self.sourceName[agrosindex]))
