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
Created on April 30, 2018

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from PostProcessors.FTStructure import FTStructure
#Internal Modules End-----------------------------------------------------------


class FTModel(ExternalModelPluginBase):
  """
    This class is designed to create a Fault-Tree model
  """

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the Fault-Tree model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.mapping    = {}
    container.InvMapping = {}

    for child in xmlNode:
      if child.tag == 'topEvents':
        container.topEventID = child.text.strip()
      elif child.tag == 'map':
        container.mapping[child.get('var')]      = child.text.strip()
        container.InvMapping[child.text.strip()] = child.get('var')
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      else:
        raise IOError("FTModel: xml node " + str (child.tag) + " is not allowed")

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    pass

  def createNewInput(self, container, inputs, samplerType, **Kwargs):
    """
      This function has been added for this model in order to be able to create a FTstructure from multiple files
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(kwargs)), tuple, return the new input in a tuple form
    """
    container.faultTreeModel = FTStructure(inputs, container.topEventID)
    container.faultTreeModel.FTsolver()
    return Kwargs

  def run(self, container, Inputs):
    """
      This method determines the status of the TopEvent of the FT provided the status of its Basic Events
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
    """
    if self.checkTypeOfAnalysis(container,Inputs):
      value = self.runTimeDep(container, Inputs)
    else:
      value = self.runStatic(container, Inputs)

    container.__dict__[container.topEventID]= value[container.topEventID]

  def checkTypeOfAnalysis(self,container,Inputs):
    """
      This method checks which type of analysis to be performed:
       - True:  dynamic (time dependent)
       - False: static
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, analysisType, bool, type of analysis to be performed

    """
    arrayValues=set()
    for key in Inputs.keys():
      if key in container.mapping.keys():
        arrayValues.add(Inputs[key])
    analysisType = None
    if arrayValues.difference({0.,1.}):
      analysisType = True
    else:
      analysisType = False
    return analysisType

  def runStatic(self, container, Inputs):
    """
      This method performs a static analysis of the FT model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, value, float, value of the Tope Event of the FT
    """

    inputForFT = {}
    for key in container.InvMapping.keys():
      inputForFT[key] = Inputs[container.InvMapping[key]]
    value = container.faultTreeModel.evaluateFT(inputForFT)
    return value

  def runTimeDep(self, container, Inputs):
    """
      This method performs a dynamic analysis of the FT model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, outcome, dict, time depedendnt value of the Tope Event of the FT
    """
    times = []
    times.append(0.)
    for key in Inputs.keys():
      if key in container.mapping.keys() and Inputs[key]!=1.:
        times.append(Inputs[key])
    times = sorted(times, key=float)

    outcome={}
    outcome[container.topEventID] = np.asarray([0.])

    for time in times:
      inputToPass=self.inputToBePassed(container,time,Inputs)
      tempOut = self.runStatic(container, inputToPass)
      for var in outcome.keys():
        if tempOut[var] == 1.:
          if time == 0.:
            outcome[var] = np.asarray([1.])
          else:
            if outcome[var][0] <= 0:
              outcome[var] = np.asarray([time])
    return outcome

  def inputToBePassed(self,container,time,Inputs):
    """
      This method return the status of the input variables at time t=time
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ In, time, float, time at which the input variables need to be evaluated
      @ Out, inputToBePassed, dict, value of the FT basic events at t=time
    """
    inputToBePassed = {}
    for key in Inputs.keys():
      if key in container.mapping.keys():
        if Inputs[key] == 0. or Inputs[key] == 1.:
          inputToBePassed[key] = Inputs[key]
        else:
          if Inputs[key] > time:
            inputToBePassed[key] = np.asarray([0.])
          else:
            inputToBePassed[key] = np.asarray([1.])
    return inputToBePassed



