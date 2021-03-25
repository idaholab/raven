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
Created on December 1st, 2015

"""
from __future__ import division, print_function, unicode_literals, absolute_import

#External Modules------------------------------------------------------------------------------------
import abc
import os
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils.cached_ndarray import c1darray
from utils import utils, InputData
from BaseClasses import MessageUser
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class CheckInterfacePP(InputData.CheckClass):
  """
  Checks that this is an Interface Post Processor of a given type
  """
  def __init__(self, name):
    """
      Creates a CheckInterfacePP class
      @ In, name, string, the method name
      @ Out, None
    """
    self.name = name
    self.__reason = ""

  def check(self, node):
    """
      Checks the node to see if it matches the checkDict
      @ In, node, xml node to check
      @ Out, bool, true if matches
    """
    self.__reason = ""
    passed = "subType" in node.attrib and node.attrib["subType"] == "InterfacedPostProcessor"
    if not passed:
      self.__reason = "subType=InterfacedPostProcessor not in attribs"
    methods = node.findall("method")
    if len(methods) == 1:
      match = methods[0].text == self.name
      if not match:
        self.__reason += ""+repr(methods[0].text)+ "!=" + self.name + " "
      passed = passed and match
    else:
      self.__reason += "wrong number of method blocks "+str(len(methods))+" "
      passed = False
    return passed

  def failCheckReason(self, node):
    """
      returns a string about why the check failed
      @ In, node, xml node to check
      @ Out, string, message for user about why check failed.
    """
    return self.__reason

class PostProcessorInterfaceBase(utils.metaclass_insert(abc.ABCMeta,object), MessageUser):
  """
    This class is the base interfaced post-processor class
    It contains the three methods that need to be implemented:
      - initialize
      - run
      - readMoreXML
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
    inputSpecification = InputData.parameterInputFactory(cls.__name__, ordered=False)
    inputSpecification.setCheckClass(CheckInterfacePP("PostProcessorInterfaceBaseClass"))
    inputSpecification.addParam("subType", InputTypes.StringType)
    inputSpecification.addParam("name", InputTypes.StringType)

    return inputSpecification

  def __init__(self, **kwargs):
    """
      Constructor
      @ In, messageHandler, MessageHandler, optional, message handler object
      @ Out, None
    """
    super().__init__(**kwargs)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__

    self.transformationSettings = {}   # this dictionary is used to store all the setting required to back transform the data into its original format
                                       # it gets filled in the run method and used in the inverse method


  def initialize(self):
    """
      Method to initialize the Interfaced Post-processor. Note that the user needs to specify two mandatory variables:
       - self.inputFormat:  dataObject that the PP is supposed to receive in input
       - self.outputFormat: dataObject that the PP is supposed to generate in output
      These two variables check that the input and output dictionaries match what PP is supposed to receive and generate
      Refer to the manual on the format of these two dictionaries
      @ In, None
      @ Out, None
    """
    self.inputFormat  = None
    self.outputFormat = None

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    pass

  def run(self,inputDic):
    """
     Method to post-process the dataObjects
     @ In, inputDic, dict, dictionary which contains the data inside the input DataObject
     @ Out, None
    """
    pass

  def _inverse(self,inputDic):
    """
      Method to perform the inverse of the post-process action
      @ In, inputDic, dict, dictionary which contains the data to be back pre-processed
      @ Out, None
    """
    pass

  def checkGeneratedDicts(self,outputDic):
    """
      Method to check that dictionary generated in def run(self, inputDic) is consistent
      @ In, outputDic, dict, dictionary generated by the run method
      @ Out, True/False, bool, outcome of the outputDic check
    """
    checkInp = self.checkInputFormat(outputDic['data']['input'])
    checkOut = self.checkOutputFormat(outputDic['data']['output'])
    if checkInp and checkOut:
      return True
    else:
      if not checkInp:
        self.raiseAWarning('PP Generation check on Inputs failed!')
      if not checkOut:
        self.raiseAWarning('PP Generation check on Outputs failed!')
      return False

  def checkOutputFormat(self,outputDic):
    """
      This method checks that the generated output part of the generated dictionary is built accordingly to outputFormat
      @ In, outputDic, dict, dictionary generated by the run method
      @ Out, outcome, bool, outcome of the outputDic check (True/False)
    """
    outcome = True
    if isinstance(outputDic,dict):
      if self.outputFormat == 'HistorySet':
        for key in outputDic:
          if isinstance(outputDic[key],dict):
            outcome = outcome and True
          else:
            self.raiseAWarning('Bad PP output type for key:',key,':',type(outputDic[key]),'; should be dict!')
            outcome = False
          for keys in outputDic[key]:
            if isinstance(outputDic[key][keys],(np.ndarray,c1darray)):
              outcome = outcome and True
            else:
              self.raiseAWarning('Bad PP output type for key:',key,keys,':',type(outputDic[key][keys]),'; should be np.ndarray or c1darray!')
              outcome = False
      else:  # self.outputFormat == 'PointSet':
        for key in outputDic:
          if isinstance(outputDic[key],(np.ndarray,c1darray)):
            outcome = outcome and True
          else:
            self.raiseAWarning('Bad PP output type for key:',key,':',type(outputDic[key]),'; should be np.ndarray or c1darray!')
            outcome = False
    else:
      self.raiseAWarning('Bad PP output dict:',type(outputDic),'is not a dict!')
      outcome = False
    return outcome

  def checkInputFormat(self,outputDic):
    """
      This method checks that the generated input part of the generated dictionary is built accordingly to outputFormat
      @ In, outputDic, dict, dictionary generated by the run method
      @ Out, outcome, bool, outcome of the outputDic check (True/False)
    """
    outcome = True
    if isinstance(outputDic,dict):
        for key in outputDic:
          if isinstance(outputDic[key],(np.ndarray,c1darray)):
            outcome = outcome and True
          else:
            self.raiseAWarning('Bad PP output type for key:',key,':',type(outputDic[key]),'; should be np.ndarray or c1darray!')
            outcome = False
    else:
      self.raiseAWarning('Bad PP output dict:',type(outputDic),'is not a dict!')
      outcome = False
    return outcome

  def checkArrayMonotonicity(time):
    """
      This method checks that an array is increasing monotonically
      @ In, time, numpy array, array to be checked
      @ Out, outcome, bool, outcome of the monotonicity check
    """
    outcome = True
    for t in time:
      if t != 0:
        if time[t] > time[t-1]:
          outcome = outcome and True
        else:
          outcome = outcome and False
    return outcome
