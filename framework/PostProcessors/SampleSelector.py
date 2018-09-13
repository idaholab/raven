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
Created on August 28, 2018

@author: giovannimaronati
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
import Runners
#Internal Modules End-----------------------------------------------------------

class SampleSelector(PostProcessor):
  """
    This postprocessor selects the row in which the minimum or the maximum
    of a target is found.The postprocessor can  act on DataObject, and
    generates a DataObject in return.
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
    inSpec= super(SampleSelector, cls).getInputSpecification()
    inSpec.addSub(InputData.parameterInputFactory('target',
                                                  contentType=InputData.StringType,
                                                  strictMode=True))
    inSpec.addSub(InputData.parameterInputFactory('criterion',
                                                  contentType=InputData.StringType,
                                                  strictMode=True))
    return inSpec

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.dynamic = True # from base class, indicates time-dependence is handled internally
    self.target = None # string, variable to apply postprocessor to

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
      if tag == 'target':
        self.target = child.value
      if tag == 'criterion':
        self.criterion = child.value
      elif tag == 'number':
        self.numBins = child.value

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'Expected 1 input DataObject, but received {} inputs!'.format(len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['PointSet','HistorySet','DataSet']:
      self.raiseAnError(IOError, 'SampleSelector postprocessor "{}" requires a DataObject input! Got "{}".'
                                 .format(self.name, currentInput.type))
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, realizations, list, list of realizations obtained
    """
    inData = self.inputToInternal(inputIn)
    realizations = []
    # actual method: pick min/max target
    d = inData.asDataset()
    i = d[self.target].argmin()
    pick = inData.realization(index = i)

    return pick

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect!")

    pick = evaluation[1]
    for key,value in pick.items():
      pick[key] = np.atleast_1d(value)
    output.addRealization(pick)