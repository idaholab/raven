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

class PostProcessor(Assembler):
  """
    This is the base class for postprocessors
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    Assembler.__init__(self)
    self.type = self.__class__.__name__  # pp type
    self.name = self.__class__.__name__  # pp name
    self.messageHandler = messageHandler
    self.metadataKeys = set()            # list of registered metadata keys to expected in this postprocessor
    self.metadataParams = {}             # dictionary of registered metadata keys with respect to their indexes, i.e. {key:list(indexes)}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ######## Temporary until this class inherits from the BaseType
    inputSpecification = InputData.parameterInputFactory('PostProcessor', ordered=False, baseNode=InputData.RavenBase)
    inputSpecification.addParam("name", InputData.StringType, True)
    ######## End Temporary until this class inherits from the BaseType

    ## This will replace the lines above
    # inputSpecification = super(PostProcessor, cls).getInputSpecification()
    inputSpecification.addParam("subType", InputData.StringType, True)

    return inputSpecification

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against
        the available options specific to this Sampler
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
    # if 'externalFunction' in initDict.keys(): self.externalFunction = initDict['externalFunction']
    self.inputs = inputs
    self._workingDir = runInfo['WorkingDir']

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, list of current inputs
    """
    return [(copy.deepcopy(currentInput))]

  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process. (inputToInternal output)
      @ Out, None
    """
    pass

  ## TODO FIXME ##
  # These two methods (addMetaKeys, provideExpectedMetaKeys) are made to be consistent with the BaseClasses.BaseType, and in
  # that glorious day when the PostProcessors inherit from the BaseType, these implementations should be removed.
  def addMetaKeys(self,args,params={}):
    """
      Adds keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    self.metadataKeys = self.metadataKeys.union(set(args))
    self.metadataParams.update(params)

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta,tuple, (list(str),dict), expected keys (empty if none) and expected indexes related to expected keys
    """
    return self.metadataKeys,self.metadataParams
