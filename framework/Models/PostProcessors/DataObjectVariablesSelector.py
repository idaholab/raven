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
Created on July 2, 2019

@author: alfoa
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from utils import utils
from utils import InputData, InputTypes
#Internal Modules End-----------------------------------------------------------

class DataObjectVariablesSelector(PostProcessorInterface):
  """
    Does the average of multiple realizations along the RAVEN_sampleID dimension
    ONLY, leaving the other dimensions as they are.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inSpec= super(DataObjectVariablesSelector, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('variableDataObject', contentType=InputTypes.StringType))
    inSpec.addSub(InputData.parameterInputFactory('target', contentType=InputTypes.StringType))
    return inSpec

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    super().__init__(self, messageHandler)
    self.dynamic = True # from base class, indicates time-dependence is handled internally
    self.variableDataObject = None # string, variables to apply postprocessor to
    self.variablesToExtract = []

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      tag = child.getName()
      if tag == 'variableDataObject':
        self.variableDataObject = child.value
      if tag == 'target':
        self.target = child.value

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) != 2:
      self.raiseAnError(IOError, 'Expected 2 input DataObject, but received {} inputs!'.format(len(currentInp)))
    # find feature variable dataobject
    found = False
    for index, currInp in enumerate(currentInp):
      if currInp.name in self.variableDataObject:
        found = True
        if currInp.type != 'PointSet':
          self.raiseAnError(IOError, '<variableDataObject> DataObject must be a PointSet, but received {}!'.format(currInp.type))
    if not found:
      self.raiseAnError(IOError, '<variableDataObject> DataObject has not be found among the inputs!!')
    return currentInp

  def run(self, inputs):
    """
      This method executes the postprocessor action.
      @ In, inputs, list(object), objects containing the data to process.
      @ Out, realizations, list, list of realizations obtained
    """
    for index, currInp in enumerate(inputs):
      if currInp.name in self.variableDataObject:
         #  get the variables names
        if not set([self.target]) <= set(currInp.getVars()):
          self.raiseAnError(KeyError, 'The requested target were not all found in the variableDataObject data! ' +
                            'Unused: {}. '.format(set(inputs[0].getVars()) - set(self.target)) +
                            'Missing: {}.'.format(set(self.target) - set(inputs[0].getVars())))        
        variableNames = currInp.asDataset()[self.target].values.tolist()
        break
    originalDataset =  inputs[index-1]
    if not set(variableNames) <= set(originalDataset.getVars()):
      self.raiseAnError(KeyError, 'The requested variableNames were not all found in the variableDataObject data! ' +
                        'Unused: {}. '.format(set(originalDataset.getVars()) - set(variableNames)) +
                        'Missing: {}.'.format(set(variableNames) - set(originalDataset.getVars())))       
    return originalDataset.asDataset()[variableNames+originalDataset.getVars('output')+originalDataset.getVars('meta')]


  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    result = evaluation[1]
    self.raiseADebug('Sending output to DataSet "{name}"'.format(name=output.name))
    output.load(result, style='dataset')
