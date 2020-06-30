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
Created on June 24, 2018

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import itertools
import math
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
from PostProcessors.MCSimporter import MCSreader
#Internal Modules End-----------------------------------------------------------

class MCSSolver(ExternalModelPluginBase):
  """
    This class is designed to create a MCS solver model
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)
    
  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to the MCS solver model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.filename = None 
    container.mapping    = {}
    container.InvMapping = {}

    for child in xmlNode:
      if child.tag == 'topEventID':
        container.topEventID = child.text.strip()
      elif child.tag == 'solverOrder':
        self.solverOrder = int(child.text.strip())
      elif child.tag == 'map':
        container.mapping[child.get('var')]      = child.text.strip()
        container.InvMapping[child.text.strip()] = child.get('var')
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      else:
        raise IOError("MCSSolver: xml node " + str(child.tag) + " is not allowed")
  
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
    if len(inputs) > 1:
      raise IOError("MCSSolver: More than one file has been passed to the MCS solver")
    
    mcsIDs, probability, mcsList, beList = MCSreader(inputs[0]) 
    
    self.topEventTerms = {}
    
    # mcsList is supposed to be a list of lists
    # E.g., if the MCS are ABC CD and AE --> MCS1=['A','B','C'], MCS2=['D','C'], MCS3=['A','E']
    #       then mcsList = [MCS1,MCS2,MCS3] = 
    #                    = [['A', 'B', 'C'], ['D', 'C'], ['A', 'E']]
    # Top event should be:   ABC + CD + AE +
    #                      - ABCD - ABCE - ACDE
    #                      + ABCDE
    
    for order in range(1,self.solverOrder+1):
      self.topEventTerms[order]=[]
      terms = list(itertools.combinations(mcsList,order))
      # terms is a list of tuples   
      # E.g., for order=2: [ (['A', 'B', 'C'], ['D', 'C']), 
      #                      (['A', 'B', 'C'], ['A', 'E']), 
      #                      (['D', 'C'], ['A', 'E']) ]   
      self.topEventTerms[order] = []
      for term in terms:
        basicEventCombined = set(itertools.chain(*term))
        # E.g. if                 term=(['A', 'B', 'C'], ['D', 'C'])
        #      then basicEventCombined={'A', 'B', 'D', 'C'}
        self.topEventTerms[order].append(basicEventCombined)
    return Kwargs
  
  def run(self, container, Inputs):
    """
      This method determines the status of the TopEvent of the FT provided the status of its Basic Events
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
    """
    inputForSolver = {}
    for key in container.InvMapping.keys():
      inputForSolver[key] = Inputs[container.InvMapping[key]]
    value = self.MCSsolver(**inputForSolver)
    return value
  
  def MCSsolver(self,**kwargs): 
    TEprobability = 0.0 
    for order in range(1,self.solverOrder+1):
      orderProbability=0
      for term in self.topEventTerms[order]:
        print(term)
        orderProbability = orderProbability + np.prod(term)
      TEprobability = TEprobability + orderProbability   
    return TEprobability


  