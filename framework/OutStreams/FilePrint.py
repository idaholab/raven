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

from utils import InputData, InputTypes
import DataObjects
from .OutStreamBase import OutStreamBase
from ClassProperty import ClassProperty

class FilePrint(OutStreamBase):
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
        @ In, cls, the OutStreams class of which this object will be a type
    """
    return cls._availableOutStreamTypes

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = OutStreamBase.getInputSpecification()

    types = InputTypes.makeEnumType('FilePrintTypes', 'FilePrintTypes', cls._availableOutStreamTypes)
    spec.addSub(InputData.parameterInputFactory('type', contentType=types))
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringListType))
    spec.addSub(InputData.parameterInputFactory('what', contentType=InputTypes.StringListType))
    spec.addSub(InputData.parameterInputFactory('filename', contentType=InputTypes.StringType))
    spec.addSub(InputData.parameterInputFactory('clusterLabel', contentType=InputTypes.StringType))

    # these are in user manual or code, but don't appear to be used/documented ...
    # spec.addSub(InputData.parameterInputFactory('target', contentType=InputTypes.StringListType))
    # spec.addSub(InputData.parameterInputFactory('directory',
    # contentType=InputTypes.StringListType))
    return spec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    OutStreamBase.__init__(self)
    self.type = 'OutStreamFilePrint'
    self.printTag = 'OUTSTREAM PRINT'
    self.sourceName = []
    self.sourceData = None
    self.what = None
    # dictionary of what indices have already been printed, so we don't duplicate writing efforts
    self.indexPrinted = {} # keys are filenames, which should be reset at the end of every step
    self.subDirectory = None # subdirectory where to store the outputs

  def _handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    typ = spec.findFirst('type')
    if typ is None:
      self.raiseAnError(IOError, f'OutStream.Print "{self.name}" is missing the "type" node!')
    self.options['type'] = typ.value

    source = spec.findFirst('source')
    if source is None:
      self.raiseAnError(IOError, f'OutStream.Print "{self.name}" is missing the "source" node!')
    self.sourceName = source.value

    # FIXME this is a terrible name
    what = spec.findFirst('what')
    if what is not None:
      self.what = what.value # [x.lower() for x in what.value]

    fname = spec.findFirst('filename')
    if fname is not None:
      self.filename = fname.value

    cluster = spec.findFirst('clusterLabel')
    if cluster is not None:
      self.options['clusterLabel'] = cluster.value

    # checks
    if self.options['type'] == 'csv' and self.what is not None:
      for target in [x.lower() for x in self.what]:
        if not target.startswith(('input', 'output', 'metadata')):
          self.raiseAnError(IOError, f'<what> requests must start with "input", "output", or "metadata"! See OutStream.Print "{self.name}"')

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
      for index, var in enumerate(self.what):
        paramDict['Variable Name ' + str(index) + ' :'] = var
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
    OutStreamBase.initialize(self, inDict)

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
    if self.subDirectory is not None:
      dictOptions['filenameroot'] = os.path.join(self.subDirectory,dictOptions['filenameroot'])

    if self.what:
      dictOptions['what'] = self.what

    if 'target' in self.options.keys():
      dictOptions['target'] = self.options['target']

    for index in range(len(self.sourceName)):
      try:
        empty = self.sourceData[index].isEmpty
      except AttributeError:
        empty = False
      if self.options['type'] == 'csv':
        filename = dictOptions['filenameroot']
        rlzIndex = self.indexPrinted.get(filename,0)
        dictOptions['firstIndex'] = rlzIndex
        # clusterLabel lets the user print a point set as if it were a history, with input decided by clusterLabel
        if 'clusterLabel' in self.options:
          if type(self.sourceData[index]).__name__ != 'PointSet':
            self.raiseAWarning('Label clustering currently only works for PointSet data objects!  Skipping for',self.sourceData[index].name)
          else:
            dictOptions['clusterLabel'] = self.options['clusterLabel']
        try:
          rlzIndex = self.sourceData[index].write(filename,style='CSV',**dictOptions)
        except AttributeError:
          self.raiseAnError(NotImplementedError, 'No implementation for source type', self.sourceData[index].type, 'and output type "'+str(self.options['type'].strip())+'"!')
        finally:
          self.indexPrinted[filename] = rlzIndex
      elif self.options['type'] == 'xml':
        try:
          self.sourceData[index].printXML(dictOptions)
        except AttributeError:
          self.raiseAnError(NotImplementedError, 'No implementation for source type', self.sourceData[index].type, 'and output type "'+str(self.options['type'].strip())+'"!')



  def finalize(self):
    """
      End-of-step operations for cleanup.
      @ In, None
      @ Out, None
    """
    # clear history of printed realizations; start fresh for next step
    self.indexPrinted = {}
