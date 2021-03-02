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
Created on Dec 21, 2017

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
import copy
import itertools
from collections import OrderedDict
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData, InputTypes
from utils import xmlUtils as xmlU
from utils import utils
from .FTStructure import FTStructure
import Files
#Internal Modules End-----------------------------------------------------------

class FTImporter(PostProcessor):
  """
    This is the base class of the postprocessor that imports Fault-Trees (FTs) into RAVEN as a PointSet
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR FT IMPORTER'
    self.FTFormat = None # chosen format of the FT file
    self.topEventID = None

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(FTImporter, cls).getInputSpecification()
    fileAllowedFormats = InputTypes.makeEnumType("FTFileFormat", "FTFileFormatType", ["OpenPSA"])
    inputSpecification.addSub(InputData.parameterInputFactory("fileFormat", contentType=fileAllowedFormats))
    inputSpecification.addSub(InputData.parameterInputFactory("topEventID", contentType=InputTypes.StringType))
    return inputSpecification

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    fileFormat = paramInput.findFirst('fileFormat')
    self.fileFormat = fileFormat.value
    topEventID = paramInput.findFirst('topEventID')
    self.topEventID = topEventID.value

  def run(self, inputs):
    """
      This method executes the postprocessor action.
      @ In,  inputs, list, list of file objects
      @ Out, out, dict, dict containing the processed FT
    """
    faultTreeModel = FTStructure(inputs, self.topEventID)
    return faultTreeModel.returnDict()


  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object, (DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict ={}
    outputDict['data'] = evaluation[1]

    outputDict['dims'] = {}
    for key in outputDict['data'].keys():
      outputDict['dims'][key] = []
    if output.type in ['PointSet']:
      output.load(outputDict['data'], style='dict', dims=outputDict['dims'])
    else:
      self.raiseAnError(RuntimeError, 'FTImporter failed: Output type ' + str(output.type) + ' is not supported.')
