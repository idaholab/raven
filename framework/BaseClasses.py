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
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#External Modules------------------------------------------------------------------------------------
import inspect
import sys
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
import MessageHandler
from utils import InputData
#Internal Modules End--------------------------------------------------------------------------------

class BaseType(MessageHandler.MessageUser):
  """
    this is the base class for each general type used by the simulation
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
    inputSpecification = InputData.parameterInputFactory(cls.__name__, ordered=False, baseNode=InputData.RavenBase)
    inputSpecification.addParam("name", InputData.StringType, True)

    return inputSpecification

  def __init__(self):
    self.name             = ''                                                          # name of this istance (alias)
    self.type             = type(self).__name__                                         # specific type within this class
    self.verbosity        = None                                                        # verbosity level (see message handler)
    self.globalAttributes = {}                                                          # this is a dictionary that contains parameters that are set at the level of the base classes defining the types
    self._knownAttribute  = []                                                          # this is a list of strings representing the allowed attribute in the xml input for the class
    self._knownAttribute += ['name','verbosity']                                        # attributes that are known
    self.printTag         = 'BaseType'                                                  # the tag that refers to this class in all the specific printing
    self.messageHandler   = None                                                        # message handling object
    self.variableGroups   = {}                                                          # the variables this class needs to be aware of
    self.metadataKeys     = set()                                                       # list of registered metadata keys to expect from this entity
    self.metadataParams   = {}                                                          # dictionary of registered metadata keys with repect to their indexes
    self.mods             = utils.returnImportModuleString(inspect.getmodule(BaseType)) #list of modules this class depends on (needed for automatic parallel python)
    for baseClass in self.__class__.__mro__:
      self.mods.extend(utils.returnImportModuleString(inspect.getmodule(baseClass),True))
    self.mods.extend(utils.returnImportModuleString(inspect.getmodule(self),True))

  def readXML(self,xmlNode,messageHandler,variableGroups={},globalAttributes=None):
    """
      provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _readMoreXML
      that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag),
      verbosity (xml attribute)
      @ In, xmlNode, ET.Element, input xml
      @ In, messageHandler, MessageHandler object, message handler
      @ In, variableGroups, dict{str:VariableGroup}, optional, variable groups container
      @ In, globalAttributes, dict{str:object}, optional, global attributes
      @ Out, None
    """
    self.setMessageHandler(messageHandler)
    self.variableGroups = variableGroups
    if 'name' in xmlNode.attrib.keys():
      self.name = xmlNode.attrib['name']
    else:
      self.raiseAnError(IOError,'not found name for a '+self.__class__.__name__)
    self.type     = xmlNode.tag
    if globalAttributes is not None:
      self.globalAttributes = globalAttributes
    if 'verbosity' in xmlNode.attrib.keys() or 'verbosity' in self.globalAttributes:
      verbGlobal = None if self.globalAttributes is None else self.globalAttributes.get('verbosity')
      verbLocal = xmlNode.attrib.get('verbosity')
      self.verbosity = verbLocal if verbLocal is not None else verbGlobal
      self.raiseADebug('Set verbosity for '+str(self)+' to '+str(self.verbosity))
    #search and replace variableGroups where found in texts
    def replaceVariableGroups(node):
      """
        Replaces variables groups with variable entries in text of nodes
        @ In, node, xml.etree.ElementTree.Element, the node to search for replacement
        @ Out, None
      """
      if node.text is not None and node.text.strip() != '':
        textEntries = list(t.strip() for t in node.text.split(','))
        for t,text in enumerate(textEntries):
          if text in variableGroups.keys():
            textEntries[t] = variableGroups[text].getVarsString()
            self.raiseADebug('Replaced text in <%s> with variable group "%s"' %(node.tag,text))
        #note: if we don't explicitly convert to string, scikitlearn chokes on unicode type
        node.text = str(','.join(textEntries))
      for child in node:
        replaceVariableGroups(child)
    replaceVariableGroups(xmlNode)
    self._readMoreXML(xmlNode)
    self.raiseADebug('------Reading Completed for:')
    self.printMe()

  def handleInput(self,paramInput,messageHandler,variableGroups={},globalAttributes=None):
    """
      provide a basic reading capability from the xml input file for what is common to all types in the simulation than calls _handleInput
      that needs to be overloaded and used as API. Each type supported by the simulation should have: name (xml attribute), type (xml tag),
      verbosity (xml attribute)
      @ In, paramInput, InputParameter, input data from xml
      @ In, messageHandler, MessageHandler object, message handler
      @ In, variableGroups, dict{str:VariableGroup}, optional, variable groups container
      @ In, globalAttributes, dict{str:object}, optional, global attributes
      @ Out, None
    """
    self.setMessageHandler(messageHandler)
    self.variableGroups = variableGroups
    if 'name' in paramInput.parameterValues:
      self.name = paramInput.parameterValues['name']
    else:
      self.raiseAnError(IOError,'not found name for a '+self.__class__.__name__)
    self.type     = paramInput.getName()
    if self.globalAttributes!= None:
      self.globalAttributes = globalAttributes
    if 'verbosity' in paramInput.parameterValues:
      self.verbosity = paramInput.parameterValues['verbosity'].lower()
      self.raiseADebug('Set verbosity for '+str(self)+' to '+str(self.verbosity))
    #TODO fix replacing Variable Groups.
    #search and replace variableGroups where found in texts
    # def replaceVariableGroups(node):
    #   """
    #     Replaces variables groups with variable entries in text of nodes
    #     @ In, node, xml.etree.ElementTree.Element, the node to search for replacement
    #     @ Out, None
    #   """
    #   if node.text is not None and node.text.strip() != '':
    #     textEntries = list(t.strip() for t in node.text.split(','))
    #     for t,text in enumerate(textEntries):
    #       if text in variableGroups.keys():
    #         textEntries[t] = variableGroups[text].getVarsString()
    #         self.raiseADebug('Replaced text in <%s> with variable group "%s"' %(node.tag,text))
    #     #note: if we don't explicitly convert to string, scikitlearn chokes on unicode type
    #     node.text = str(','.join(textEntries))
    #   for child in node:
    #     replaceVariableGroups(child)
    #replaceVariableGroups(xmlNode)
    self._handleInput(paramInput)
    self.raiseADebug('------Reading Completed for:')
    self.printMe()

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some variables based on the inputs got.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    pass

  def _handleInput(self, paramInput):
    """
      Function to handle the input parameters that belong to this specialized
      and initialize variables based on the input.
      @ In, paramInput, InputData.Parameters
      @ Out, None
    """
    pass

  def setMessageHandler(self,handler):
    """
      Function to set up the link to the the common Message Handler
      @ In, handler, MessageHandler object, message handler
      @ Out, None
    """
    if not isinstance(handler,MessageHandler.MessageHandler):
      e=IOError('Attempted to set the message handler for '+str(self)+' to '+str(handler))
      print('\nERROR! Setting MessageHandler in BaseClass,',e,'\n')
      sys.exit(1)
    self.messageHandler = handler

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
      @ Out, paramDict, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
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
      @ Out, paramDict, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
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

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and indexes/dimensions corresponding to expected keys
    """
    return self.metadataKeys, self.metadataParams

  def addMetaKeys(self,args, params={}):
    """
      Adds keywords to a list of expected metadata keys.
      @ In, args, list(str), keywords to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    if any(not utils.isAString(a) for a in args):
      self.raiseAnError('Arguments to addMetaKeys were not all strings:',args)
    self.metadataKeys = self.metadataKeys.union(set(args))
    self.metadataParams.update(params)
