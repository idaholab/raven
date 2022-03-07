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

from .. import DataObjects
from .. import Models
from ..BaseClasses import BaseInterface
from ..utils import InputTypes, InputData

class OutStreamInterface(BaseInterface):
  """
    Base class for other OutStream Interfaces (Print, Plot).
    Not meant to be an interface itself.
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    spec.addParam('dir', descr=r"""Designates the directory in which files should be saved. If not
        an absolute directory, assumes relative to the WorkingDir. """)
    spec.addParam('overwrite', param_type=InputTypes.BoolType, descr=r"""If True, then any conflicting
        existing files will be overwritten.""")
    spec.addSub(InputData.parameterInputFactory('filename', contentType=InputTypes.StringType,
        descr=r"""Sets the name to use for the saved file."""))
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'OutStreamInterface'
    self.overwrite = True       # overwrite existing creations?
    self.subDirectory = None    # directory to save generated files to
    self.filename = ''          # target file name
    self.numberAggregatedOS = 1 # number of aggregated outstreams # no addl info from original OutStream

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)
    self.subDirectory = spec.parameterValues.get('dir')
    self.overwrite = spec.parameterValues.get('overwrite')
    fname = spec.findFirst('filename')
    if fname is not None:
      self.filename = fname.value

  def initialize(self, stepEntities):
    """
      Initialize for a new Step
      @ In, stepEntities, dict, Entities from the Step
      @ Out, None
    """
    super().initialize()
    if self.subDirectory is not None:
      if not os.path.exists(self.subDirectory):
        os.makedirs(self.subDirectory)

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
    return paramDict

  def legacyCollectSources(self, inDict):
    """
      Collect the usable sources in the format this plotter expects.
      This is a legacy method; it is used by FilePrint and GeneralPlot (the original two OutStreams),
      but appears to be specific to particular approaches so should not necessarily be adopted generally.
      @ In, inDict, dict, Step entities
      @ Out, None
    """
    self.sourceData = []
    for outIndex in range(self.numberAggregatedOS):
      foundData = False
      for output in inDict['Output']:
        if output.name.strip() == self.sourceName[outIndex] and output.type in DataObjects.factory.knownTypes():
          self.sourceData.append(output)
          foundData = True
          break
      if not foundData:
        for inp in inDict['Input']:
          if not isinstance(inp, str):
            if inp.name.strip() == self.sourceName[outIndex] and inp.type in DataObjects.factory.knownTypes():
              self.sourceData.append(inp)
              foundData = True
            elif type(inp) == Models.ROM:
              self.sourceData.append(inp)
              foundData = True  # good enough
      if not foundData and 'TargetEvaluation' in inDict.keys():
        if inDict['TargetEvaluation'].name.strip() == self.sourceName[outIndex] and inDict['TargetEvaluation'].type in DataObjects.factory.knownTypes():
          self.sourceData.append(inDict['TargetEvaluation'])
          foundData = True
      if not foundData and 'SolutionExport' in inDict.keys():
        if inDict['SolutionExport'].name.strip() == self.sourceName[outIndex] \
            and inDict['SolutionExport'].type in DataObjects.factory.knownTypes():
          self.sourceData.append(inDict['SolutionExport'])
          foundData = True
      if not foundData:
        self.raiseAnError(IOError, 'the DataObject "{data}" was not found among the "<Input>" nodes for this step!'.format(data = self.sourceName[outIndex]))
