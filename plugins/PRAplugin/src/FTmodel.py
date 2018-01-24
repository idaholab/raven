from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from PostProcessors.FTstructure import FTstructure
#Internal Modules End-----------------------------------------------------------


class FTmodel(ExternalModelPluginBase):

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
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
        print('xml error')

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    #container.faultTreeModel = FTstructure(inputFiles, self.messageHandler, container.topEventID)
    container.faultTreeModel = FTstructure(inputFiles, container.topEventID)

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      This method takes the variables in input and computes
      oneOutputOfThisPlugin(t) = var1Coefficient*exp(var1*t)+var2Coefficient*exp(var2*t) ...
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    value = container.faultTreeModel.evaluateFT(Inputs)
    container.__dict__[container.topEventID]= value
