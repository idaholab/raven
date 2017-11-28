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
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

#Internal Modules---------------------------------------------------------------
import DataObjects
from .OutStreamManager import OutStreamManager
from ClassProperty import ClassProperty
#Internal Modules End-----------------------------------------------------------

class OutStreamPrint(OutStreamManager):
  """
    Class for managing the printing of files as an output stream.
  """

  ## Promoting these to static class variables, since they will not alter from
  ## object to object. The use of the @ClassProperty with only a getter makes
  ## the variables immutable (so long as no one touches the internally stored
  ## "_"-prefixed), so other objects don't accidentally modify them.

  _availableOutStreamTypes = ['csv', 'xml']

  @ClassProperty
  def availableOutStreamTypes(cls):
    """
        A class level constant that tells developers what outstreams are
        available from this class
        @ In, cls, the OutStreamPrint class of which this object will be a type
    """
    return cls._availableOutStreamTypes

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    OutStreamManager.__init__(self)
    self.printTag = 'OUTSTREAM PRINT'
    OutStreamManager.__init__(self)
    self.sourceName = []
    self.sourceData = None
    self.what = None

  def localGetInitParams(self):
    """
      This method is called from the base function. It retrieves the initial
      characteristic params that need to be seen by the whole enviroment
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for index in range(len(self.sourceName)):
      paramDict['Source Name ' + str(index) + ' :'] = self.sourceName[index]
    if self.what:
      for index in range(len(self.what)):
        paramDict['Variable Name ' + str(index) + ' :'] = self.what[index]
    return paramDict

  def initialize(self, inDict):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system
      @ In, inDict, dict, It contains all the Object are going to be used in the
        current step. The sources are searched into this.
      @ Out, None
    """
    # the linking to the source is performed in the base class initialize method
    OutStreamManager.initialize(self, inDict)

  def localReadXML(self, xmlNode):
    """
      This Function is called from the base class, It reads the parameters that
      belong to a plot block (outputs by filling data structure (self))
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    self.type = 'OutStreamPrint'
    for subnode in xmlNode:
      if subnode.tag not in ['type','source','what','filename','target']:
        self.raiseAnError(IOError, ' Print Outstream object ' + str(self.name) + ' contains the following unknown node: ' + str(subnode.tag))
      if subnode.tag == 'source':
        self.sourceName = subnode.text.split(',')
      elif subnode.tag == 'filename':
        self.filename = subnode.text
      else:
        self.options[subnode.tag] = subnode.text
    if 'type' not in self.options.keys():
      self.raiseAnError(IOError, 'type tag not present in Print block called ' + self.name)
    if self.options['type'] not in self.availableOutStreamTypes:
      self.raiseAnError(TypeError, 'Print type ' + self.options['type'] + ' not available yet. ')
    if 'what' in self.options.keys():
      self.what = self.options['what']

  def addOutput(self):
    """
      Calls output functions on desired instances in order to print out the
      linked dataObjects
      @ In, None
      @ Out, None
    """
    dictOptions = {}
    dictOptions['filenameroot'] = self.name
    if len(self.filename) > 0:
      dictOptions['filenameroot'] = self.filename

    if self.what:
      dictOptions['what'] = self.what

    if 'target' in self.options.keys():
      dictOptions['target'] = self.options['target']

    for index in range(len(self.sourceName)):
      try:
        empty = len(self.sourceData[index]) == 0
      except TypeError:
        empty = False
      if not empty:
        try:
          if self.options['type'] == 'csv':
            self.sourceData[index].write(dictOptions['filenameroot'],style='CSV',**dictOptions)
          elif self.options['type'] == 'xml':
            self.sourceData[index].printXML(dictOptions)
        except AttributeError:
          self.raiseAnError(IOError, 'No implementation for source type', self.sourceData[index].type, 'and output type "'+str(self.options['type'].strip())+'"!')

