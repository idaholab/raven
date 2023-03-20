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
Created on Mar 16, 2013
@author: crisr
"""
from abc import ABCMeta, abstractmethod
import copy

from ..utils import mathUtils
from ..utils.utils import metaclass_insert
from ..BaseClasses import BaseType
from ..BaseClasses import Assembler

class BaseInterface(metaclass_insert(ABCMeta, Assembler, BaseType)):
  """
    Archetype for "interface" classes, including implementations/strategies/algorithms to execute
    the intention of BaseEntity types. For example, SupervisedLearning Engines are an Interface
    to the Models.ROM class. Base interfaces define APIs for adding new algorithm classes.

    Entities in RAVEN request a specific Interface via the subType input attribute. Generally,
    <Entity name="myName" subType="requestedInterface">
      ...
    </Entity>
    such as
    <Plot name="my_line" subType="GeneralPlot">
      ...
    </Plot>
  """
  ################################
  # Core API (confirmed)
  def __init__(self):
    """
      Construct.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.name             = ''                   # name of this istance (alias)
    self.type             = type(self).__name__  # specific type within this class
    self.verbosity        = None                 # verbosity level (see message handler)
    self.globalAttributes = {}                   # this is a dictionary that contains parameters that are set at
                                                 # the level of the base classes defining the types
    self.printTag         = 'BaseType'           # the tag that refers to this class in all the specific printing
    self.variableGroups   = {}                   # the variables this class needs to be aware of
    self.metadataKeys     = set()                # list of registered metadata keys to expect from this entity
    self.metadataParams   = {}                   # dictionary of registered metadata keys with repect to their indexes
    self.saveParams       = False                # this variable controls if the input parameters should be saved (handleInput)
    self.paramInput       = None                 # if self.saveParams == True, this variable contains the paramInput of this interface

  def handleInput(self, paramInput, variableGroups=None, globalAttributes=None):
    """
      provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _handleInput
      that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag),
      verbosity (xml attribute)
      @ In, paramInput, utils.InputData.parameterInput, input data from xml
      @ In, variableGroups, dict{str:VariableGroup}, optional, variable groups container
      @ In, globalAttributes, dict{str:object}, optional, global attributes
      @ Out, None
    """
    self.variableGroups = variableGroups if variableGroups is not None else {}
    if 'name' in paramInput.parameterValues:
      self.name = paramInput.parameterValues['name']
    else:
      self.raiseAnError(IOError, 'not found name for a '+self.__class__.__name__)
    #self.type = paramInput.getName() -> we set this in __init__, setting it here to the spec type seems sketchy
    if self.globalAttributes is not None:
      self.globalAttributes = globalAttributes
    if 'verbosity' in paramInput.parameterValues:
      self.verbosity = paramInput.parameterValues['verbosity'].lower()
      self.raiseADebug('Set verbosity for '+str(self)+' to '+str(self.verbosity))
    self._handleInput(paramInput)
    self.raiseADebug('------Reading Completed for:')
    self.printMe()

  def initialize(self, *args, **kwargs):
    """
      Set up this interface for a particular activity
      @ In, args, list, positional arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """
    pass

  @abstractmethod
  def run(self, *args, **kwargs):
    """
      Main method to "do what you do".
      @ In, args, list, positional arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """

  ################################
  # Utility API
  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, (self.metadataKeys, self.metadataParams), tuple, (set(str),dict),
             expected keys (empty if none) and indexes/dimensions corresponding to expected keys
    """
    return self.metadataKeys, self.metadataParams

  def addMetaKeys(self,args, params=None):
    """
      Adds keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    if params is None:
      params = {}
    if any(not mathUtils.isAString(a) for a in args):
      self.raiseAnError('Arguments to addMetaKeys were not all strings:',args)
    self.metadataKeys = self.metadataKeys.union(set(args))
    self.metadataParams.update(params)

  def removeMetaKeys(self, args):
    """
      Removes keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to de-register
      @ Out, None
    """
    self.metadataKeys = self.metadataKeys - set(args)
    for arg in set(args):
      self.metadataParams.pop(arg, None)

  ################################
  # API (legacy) - these should go away as we convert existing systems
  def readXML(self, xmlNode, variableGroups=None, globalAttributes=None, workingDir=None):
    """
      provide a basic reading capability from the xml input file for what is common to all types in
      the simulation than calls _readMoreXML that needs to be overloaded and used as API. Each type
      supported by the simulation should have: name (xml attribute), type (xml tag),
      verbosity (xml attribute)
      @ In, xmlNode, ET.Element, input xml
      @ In, variableGroups, dict{str:VariableGroup}, optional, variable groups container
      @ In, globalAttributes, dict{str:object}, optional, global attributes
      @ In, workingDir, str, optional, the path of the working directory
      @ Out, None
    """
    if 'name' in xmlNode.attrib.keys():
      self.name = xmlNode.attrib['name']
    else:
      self.raiseAnError(IOError,'not found name for a '+self.__class__.__name__)
    self.type = xmlNode.tag
    if globalAttributes is not None:
      self.globalAttributes = globalAttributes
    if 'verbosity' in xmlNode.attrib.keys() or 'verbosity' in self.globalAttributes:
      verbGlobal = None if self.globalAttributes is None else self.globalAttributes.get('verbosity')
      verbLocal = xmlNode.attrib.get('verbosity')
      self.verbosity = verbLocal if verbLocal is not None else verbGlobal
      self.raiseADebug('Set verbosity for '+str(self)+' to '+str(self.verbosity))
    self._readMoreXML(xmlNode)
    self.raiseADebug('------Reading Completed for:')
    self.printMe()

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some variables based on the inputs got.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    super()._readMoreXML(xmlNode)

  def _handleInput(self, paramInput):
    """
      Function to handle the input parameters that belong to this specialized
      and initialize variables based on the input.
      @ In, paramInput, InputData.Parameters
      @ Out, None
    """
    if self.saveParams:
      self.paramInput = copy.deepcopy(paramInput)

  ################################
  # undecided; are these still useful?
  def whoAreYou(self):
    """
      This is a generic interface that will return the type and name of any class that inherits this base class plus all the inherited classes
      @ In, None
      @ Out, tempDict, dict, dictionary containing the Type, Class and Name of this instanciated object
    """
    tempDict          = {}
    tempDict['Class'] = '{0:15}'.format(self.__class__.__name__) +' from '+' '.join([str(base) for base in self.__class__.__bases__])
    tempDict['Type' ] = self.type
    tempDict['Name' ] = self.name
    return tempDict

  def getInitParams(self):
    """
      Function to be overloaded to get a dictionary of the name and values of the initial parameters associated with any class
      @ In, None
      @ Out, getInitParams, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
    """
    return {}

  def myCurrentSetting(self):
    """
      This is a generic interface that will return the name and value of the parameters that change during the simulation of any class that inherits this base class.
      In reality it is just empty and will fill the dictionary calling getCurrentSetting that is the function to be overloaded used as API
      @ In, None
      @ Out, paramDict, dict, dictionary containing the current parameters of this instantiated object
    """
    paramDict = self.getCurrentSetting()
    return paramDict

  def getCurrentSetting(self):
    """
      Function to be overloaded to inject the name and values of the parameters that might change during the simulation
      @ In, None
      @ Out, getCurrentSetting, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
    """
    return {}

  def printMe(self):
    """
      This is a generic interface that will print all the info for
      the instance of an object that inherit this class
      @ In, None
      @ Out, None
    """
    tempDict = self.whoAreYou()
    for key in tempDict.keys():
      self.raiseADebug('       {0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.getInitParams()
    self.raiseADebug('       Initialization Parameters:')
    for key in tempDict.keys():
      self.raiseADebug('       {0:15}: {1}'.format(key,str(tempDict[key])))
    tempDict = self.myCurrentSetting()
    self.raiseADebug('       Current Setting:')
    for key in tempDict.keys():
      self.raiseADebug('       {0:15}: {1}'.format(key,str(tempDict[key])))

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, acceptable, set, modified set of acceptable variables with all formatting complete
    """
    return acceptable

  def _validateSolutionExportVariables(self, solutionExport):
    """
      Validates entries in the SolutionExport against the list of acceptable ones.
      Overload to write custom checking.
      @ In, solutionExport, DataObjects.DataSet, target evaluation data object
      @ Out, None
    """
    # don't validate non-requests
    if solutionExport is None:
      return
    # dynamic list of unfound but requested variables
    requested = set(solutionExport.getVars())
    # get acceptable names
    fromSolnExport = set(self.getSolutionExportVariableNames())
    acceptable = set(self._formatSolutionExportVariableNames(fromSolnExport))

    # remove registered solution export names first
    remaining = requested - acceptable
    # anything remaining is unknown!
    if remaining:
      err = 'Some requested SolutionExport variables are not generated as part ' +\
            'of this entity: {}'.format(remaining)
      err += '\n-> Valid unused options include: {}'.format(acceptable - requested)
      self.raiseAnError(IOError, err)
