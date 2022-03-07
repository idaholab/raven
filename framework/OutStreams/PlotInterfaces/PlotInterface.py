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

from abc import abstractmethod
from .. import OutStreamInterface, OutStreamEntity
from ...utils.utils import displayAvailable

class PlotInterface(OutStreamInterface):
  """
    Archetype for Plot implementations
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class "cls".
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying the input of cls.
    """
    spec = super().getInputSpecification()
    return spec

  def __init__(self):
    """
      Init of Base class
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'PlotInterface'

  def handleInput(self, spec):
    """
      Loads the input specs for this object.
      @ In, spec, InputData.ParameterInput, input specifications
      @ Out, None
    """
    super().handleInput(spec)

  def initialize(self, stepEntities):
    """
      Function to initialize the OutStream. It basically looks for the "data"
      object and links it to the system.
      @ In, stepEntities, dict, contains all the Objects are going to be used in the
      current step. The sources are searched into this.
      @ Out, None
    """
    super().initialize(stepEntities)

  @abstractmethod
  def run(self):
    """
      Main run method.
      Generally, the sources from which data should be taken for plots has been established by now,
      often though the "initialize" method. This method should generate plots, and probably
      store them to file, depending on the strategy of this plotter. See examples in other plotters.
      @ In, None
      @ Out, None
    """

  def endInstructions(self, instructionString):
    """
      Finalize plotter. Called if "pauseAtEnd" is in the Step attributes.
      @ In, instructionString, string, instructions to execute
      @ Out, None
    """
    if instructionString == 'interactive' and displayAvailable():
      import matplotlib.pyplot as plt
      for i in plt.get_fignums():
        fig = plt.figure(i)
        try:
          fig.ginput(n=-1, timeout=0, show_clicks=False)
        except Exception as e:
          self.raiseAWarning('There was an error with figure.ginput. Continuing anyway ...')

  ##################
  # Utility
  def findSource(self, name, stepEntities):
    """
      Find a source from the potential step sources.
      @ In, name, str, name of the source
      @ In, stepEntities, dict, entities from the Step
      @ Out, findSource, object, discovered object or None
    """
    for out in stepEntities['Output']:
      if isinstance(out, OutStreamEntity):
        continue
      if out.name == name:
        return out
    for inp in stepEntities['Input']:
      if inp.name == name:
        return inp
    for other in ['TargetEvaluation', 'SolutionExport']:
      if other in stepEntities:
        if stepEntities[other].name == name:
          return stepEntities[other]
    return None

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
    paramDict['Global Class Type                  '] = 'Plotter'
    paramDict['Specialized Class Type             '] = self.type
    if self.overwrite:
      paramDict['Overwrite output everytime called'] = 'True'
    else:
      paramDict['Overwrite output everytime called'] = 'False'
    return paramDict
