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
Created on July 10, 2013

@author: alfoa
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy import spatial, interpolate
import os
from glob import glob
import copy
import math
from collections import OrderedDict, defaultdict
import time
from sklearn.linear_model import LinearRegression
import importlib
import abc
import six
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import mathUtils
from utils import xmlUtils
from utils.RAVENiterators import ravenArrayIterator
import DataObjects
from Assembler import Assembler
import LearningGate
import MessageHandler
import GridEntities
import Files
import Models
import unSupervisedLearning
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class InterfacedPostProcessor(PostProcessor):
  """
    This class allows to interface a general-purpose post-processor created ad-hoc by the user.
    While the ExternalPostProcessor is designed for analysis-dependent cases, the InterfacedPostProcessor is designed more generic cases
    The InterfacedPostProcessor parses (see PostProcessorInterfaces.py) and uses only the functions contained in the raven/framework/PostProcessorFunctions folder
    The base class for the InterfacedPostProcessor that the user has to inherit to develop its own InterfacedPostProcessor is specified
    in PostProcessorInterfaceBase.py
  """

  PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super(RavenOutput, cls).getInputSpecification()

    ## TODO: Fill this in with the appropriate tags

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.methodToRun = None

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the Interfaced processor
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """

    # paramInput = InterfacedPostProcessor.getInputSpecification()()
    # paramInput.parseNode(xmlNode)

    for child in xmlNode:
      if child.tag == 'method':
        self.methodToRun = child.text
    self.postProcessor = InterfacedPostProcessor.PostProcessorInterfaces.returnPostProcessorInterface(self.methodToRun,self)
    if not isinstance(self.postProcessor,PostProcessorInterfaceBase):
      self.raiseAnError(IOError, 'InterfacedPostProcessor Post-Processor '+ self.name +' : not correctly coded; it must inherit the PostProcessorInterfaceBase class')

    self.postProcessor.initialize()
    self.postProcessor.readMoreXML(xmlNode)
    if self.postProcessor.inputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.inputFormat not correctly initialized')
    if self.postProcessor.outputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.outputFormat not correctly initialized')


  def run(self, inputIn):
    """
      This method executes the interfaced  post-processor action.
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDic, dict, dict containing the post-processed results
    """
    if self.postProcessor.inputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.inputFormat not correctly initialized')
    if self.postProcessor.outputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.outputFormat not correctly initialized')
    inputDic= self.inputToInternal(inputIn)

    outputDic = self.postProcessor.run(inputDic)
    if self.postProcessor.checkGeneratedDicts(outputDic):
      return outputDic
    else:
      self.raiseAnError(RuntimeError,'InterfacedPostProcessor Post-Processor '+ self.name +' : function has generated a not valid output dictionary')

  def _inverse(self, inputIn):
    outputDic = self.postProcessor._inverse(inputIn)
    return outputDic

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluations = finishedJob.getEvaluation()
    if isinstance(evaluations, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    evaluation = evaluations[1]

    exportDict = {'inputSpaceParams':evaluation['data']['input'],'outputSpaceParams':evaluation['data']['output'],'metadata':evaluation['metadata']}

    listInputParms   = output.getParaKeys('inputs')
    listOutputParams = output.getParaKeys('outputs')

    if output.type == 'HistorySet':
      for hist in exportDict['inputSpaceParams']:
        if type(exportDict['inputSpaceParams'].values()[0]).__name__ == "dict":
          for key in listInputParms:
            output.updateInputValue(key,exportDict['inputSpaceParams'][hist][str(key)])
          for key in listOutputParams:
            output.updateOutputValue(key,exportDict['outputSpaceParams'][hist][str(key)])
        else:
          for key in exportDict['inputSpaceParams']:
            if key in output.getParaKeys('inputs'):
              output.updateInputValue(key,exportDict['inputSpaceParams'][key])
          for key in exportDict['outputSpaceParams']:
            if key in output.getParaKeys('outputs'):
              output.updateOutputValue(key,exportDict['outputSpaceParams'][str(key)])
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key])
    else:
      # output.type == 'PointSet':
      for key in exportDict['inputSpaceParams']:
        if key in output.getParaKeys('inputs'):
          for value in exportDict['inputSpaceParams'][key]:
            output.updateInputValue(str(key),value)
      for key in exportDict['outputSpaceParams']:
        if str(key) in output.getParaKeys('outputs'):
          for value in exportDict['outputSpaceParams'][key]:
            output.updateOutputValue(str(key),value)
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key])


  def inputToInternal(self,input):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, input, list, list of dataObjects handed to the post-processor
      @ Out, inputDict, list, list of dictionaries this object can process
    """
    inputDict = []
    for inp in input:
      if type(inp) == dict:
        return [inp]
      else:
        inputDictTemp = {'data':{}, 'metadata':{}}
        inputDictTemp['data']['input']  = copy.deepcopy(inp.getInpParametersValues())
        inputDictTemp['data']['output'] = copy.deepcopy(inp.getOutParametersValues())
        inputDictTemp['metadata']       = copy.deepcopy(inp.getAllMetadata())
        inputDictTemp['name'] = inp.whoAreYou()['Name']
        inputDictTemp['type'] = str(inp.type)
        inputDict.append(inputDictTemp)
    return inputDict
