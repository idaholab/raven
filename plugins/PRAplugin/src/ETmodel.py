from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from PostProcessors.ETstructure import ETstructure
#Internal Modules End-----------------------------------------------------------


class ETmodel(ExternalModelPluginBase):
  """
    This class is designed to create an Event-Tree model
  """

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the Event-Tree model
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
      elif child.tag == 'sequenceID':
        container.sequenceID = child.text.strip()
      else:
        raise IOError("ETmodel: xml node " + str (child.tag) + " is not allowed")

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize the Event-Tree model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    pass

  def createNewInput(self, container, inputs, samplerType, **Kwargs):
    """
      This function has been added for this model in order to be able to create an ETstructure from multiple files
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(kwargs)), tuple, return the new input in a tuple form
    """
    container.eventTreeModel = ETstructure(inputs=inputs, expand=True)
    return Kwargs

  def run(self, container, Inputs):
    """
      This method provides the sequence of the ET given the status of its branching conditions
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
    """
    inputForET = {}
    for key in container.InvMapping.keys():
      if Inputs[container.InvMapping[key]] > 0:
        inputForET[key] = 1.0
      else:
        inputForET[key] = 0.0

    value = container.eventTreeModel.solve(inputForET)
    container.__dict__[container.sequenceID]= value
