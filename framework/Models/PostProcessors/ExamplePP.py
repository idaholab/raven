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

#External Modules---------------------------------------------------------------
import copy
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from BaseClasses import BaseType
from utils import InputData
from Assembler import Assembler
import MessageHandler
#Internal Modules End-----------------------------------------------------------

class ExamplePostProcessor(PostProcessorBase):
  """
    This is the base class for ExamplePostProcessor
  """
  @classmethod
  def getInputSpecification(cls): # --> An interface should created for this method
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ExamplePostProcessor, cls).getInputSpecification()
    return inputSpecification

  def _readMoreXML(self, xmlNode): # --> use this method instead of getInputSpecification
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    pass

  def initialize(self, runInfo, inputs, initDict=None) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    self.inputs = inputs
    self._workingDir = runInfo['WorkingDir']

  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process.
      Should avoid to use (inputToInternal output), and passing xarray directly/dataset
      Possible inputs include: dict, xarray.Dataset, pd.DataFrame
      @ Out, dict, xarray.Dataset, pd.DataFrame --> I think we can avoid collectoutput in the plugin pp
    """
    pass

  def addMetaKeys(self,args,params={}): # Do we want to expose this to users? Do we want to create a local method?
    """
      Adds keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    PostProcessorBase.addMetaKeys(args, params)
