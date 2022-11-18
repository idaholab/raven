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

@author: alfoa, talbpaul
"""
import os

from ...utils import InputData, InputTypes
from .PrintInterface import PrintInterface

class FilePrint(PrintInterface):
  """
    Class for managing the printing of files as an output stream.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()

    types = InputTypes.makeEnumType('FilePrintTypes', 'FilePrintTypes', ['csv', 'xml'])
    spec.addSub(InputData.parameterInputFactory('type', contentType=types))
    spec.addSub(InputData.parameterInputFactory('source', contentType=InputTypes.StringListType))
    spec.addSub(InputData.parameterInputFactory('what', contentType=InputTypes.StringListType))
    spec.addSub(InputData.parameterInputFactory('clusterLabel', contentType=InputTypes.StringType))

    return spec

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type = 'OutStreamFilePrint'
    self.printTag = 'OUTSTREAM PRINT'
    self.sourceName = []
    self.sourceData = []
    self.what = None
    self.options = {}    # outstreaming options # no addl info from original developer
    # dictionary of what indices have already been printed, so we don't duplicate writing efforts
    self.indexPrinted = {} # keys are filenames, which should be reset at the end of every step
    self.subDirectory = None # subdirectory where to store the outputs

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)

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

  def getInitParams(self):
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

  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system
      @ In, stepEntities, dict, entities from the current Step
      @ Out, None
    """
    self.legacyCollectSources(stepEntities)
    super().initialize(stepEntities)

  def run(self):
    """
      Calls output functions on desired instances in order to print out the
      linked dataObjects
      @ In, None
      @ Out, None
    """
    super().run()
    dictOptions = {}
    dictOptions['filenameroot'] = self.name
    if len(self.filename) > 0:
      dictOptions['filenameroot'] = self.filename
    if self.subDirectory is not None:
      dictOptions['filenameroot'] = os.path.join(self.subDirectory,dictOptions['filenameroot'])

    if self.what:
      dictOptions['what'] = self.what

    if 'target' in self.options:
      dictOptions['target'] = self.options['target']

    for index in range(len(self.sourceName)):
      if self.options['type'] == 'csv':
        filename = dictOptions['filenameroot']
        rlzIndex = self.indexPrinted.get(filename,0)
        dictOptions['firstIndex'] = rlzIndex
        # clusterLabel lets the user print a point set as if it were a history, with input decided by clusterLabel
        if 'clusterLabel' in self.options:
          if type(self.sourceData[index]).__name__ != 'PointSet':
            self.raiseAWarning(f'Label clustering currently only works for PointSet data objects!  Skipping for {self.sourceData[index].name}')
          else:
            dictOptions['clusterLabel'] = self.options['clusterLabel']
        try:
          rlzIndex = self.sourceData[index].write(filename,style='CSV',**dictOptions)
        except AttributeError:
          self.raiseAnError(NotImplementedError, f'No implementation for source type {self.sourceData[index].type} and output type "{self.options["type"].strip()}"!')
        finally:
          self.indexPrinted[filename] = rlzIndex
      elif self.options['type'] == 'xml':
        try:
          self.sourceData[index].printXML(dictOptions)
        except AttributeError:
          self.raiseAnError(NotImplementedError, f'No implementation for source type {self.sourceData[index].type} and output type "{self.options["type"].strip()}"!')

  def finalize(self):
    """
      End-of-step operations to enable re-running workflows.
      @ In, None
      @ Out, None
    """
    # clear history of printed realizations
    self.indexPrinted = {}
    self.sourceData = []
