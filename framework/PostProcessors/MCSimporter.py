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
Created on Nov 1, 2019

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import pandas as pd
import numpy as np
import csv
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData, InputTypes
from utils import xmlUtils as xmlU
from utils import utils
#Internal Modules End-----------------------------------------------------------


class MCSImporter(PostProcessor):
  """
    This is the base class of the PostProcessor that imports Minimal Cut Sets (MCSs) into RAVEN as a PointSet
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag  = 'POSTPROCESSOR MCS IMPORTER'
    self.expand    = None  # option that controls the structure of the ET. If True, the tree is expanded so that
                           # all possible sequences are generated. Sequence label is maintained according to the
                           # original tree

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(MCSImporter, cls).getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("expand",       contentType=InputTypes.BoolType))
    inputSpecification.addSub(InputData.parameterInputFactory("BElistColumn", contentType=InputTypes.StringType))
    return inputSpecification

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the PostProcessor
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

    expand = paramInput.findFirst('expand')
    self.expand = expand.value

    if self.expand:
      beListColumn = paramInput.findFirst('BElistColumn')
      self.beListColumn = beListColumn.value

  def run(self, inputs):
    """
      This method executes the PostProcessor action.
      @ In,  inputs, list, list of file objects
      @ Out, None
    """

    mcsFileFound = False
    beFileFound  = False

    for file in inputs:
      if file.getType()=="MCSlist":
        if mcsFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=MCSlist have been found')
        else:
          MCSlistFile = file
          mcsFileFound = True
      if file.getType()=="BElist":
        if self.expand==False:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', A file with type=BElist has been found but expand is set to False')
        if beFileFound:
          self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Multiple files with type=BElist have been found')
        else:
          BElistFile = file
          beFileFound  = True

    if beFileFound==False and self.expand==True:
      self.raiseAnError(IOError, 'MCSImporterPostProcessor Post-Processor ' + self.name + ', Expand is set to False but no file with type=BElist has been found')

    self.mcsList=[]
    self.beList=set()
    self.probability = np.zeros((0))
    self.mcsIDs = np.zeros((0))

    # construct the list of MCSs and the list of BE
    counter=0
    with open(MCSlistFile.getFilename(), 'r') as file:
      next(file) # skip header
      lines = file.read().splitlines()
      for l in lines:
        elementsList = l.split(',')

        self.mcsIDs=np.append(self.mcsIDs,elementsList[0])
        elementsList.pop(0)

        self.probability=np.append(self.probability,elementsList[0])
        elementsList.pop(0)

        for element in elementsList:
          element.rstrip('\n')
        self.mcsList.append(elementsList)
        counter = counter+1
        if self.expand==False:
          self.beList.update(elementsList)
    if self.expand:
      beData = pd.read_csv(BElistFile.getFilename())
      self.beList = beData[self.beListColumn].values.tolist()

    mcsPointSet = {}

    # MCS Input variables
    mcsPointSet['probability'] = self.probability
    mcsPointSet['MCS_ID']      = self.mcsIDs
    mcsPointSet['out']         = np.ones((counter))

    # MCS Output variables
    for be in self.beList:
      mcsPointSet[be]= np.zeros(counter)
    counter=0
    for mcs in self.mcsList:
      for be in mcs:
        mcsPointSet[be][counter] = 1.0
      counter = counter+1

    return mcsPointSet

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object, (DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this PostProcessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict ={}
    outputDict['data'] = evaluation[1]

    if output.type in ['PointSet']:
      outputDict['dims'] = {}
      for key in outputDict.keys():
        outputDict['dims'][key] = []
      output.load(outputDict['data'], style='dict', dims=outputDict['dims'])
    else:
        self.raiseAnError(RuntimeError, 'MCSImporter failed: Output type ' + str(output.type) + ' is not supported.')
