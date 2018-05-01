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
Created on Oct 20, 2014

@author: alfoa

This module contains interfaces to import external functions
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, absolute_import
# WARNING if you import unicode_literals here, we fail tests (e.g. framework.testFactorials).  This may be a future-proofing problem. 2015-04.
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import utils
from CustomCommandExecuter import execCommand

#Internal Modules End--------------------------------------------------------------------------------


class Function(BaseType):
  """
    This class is the base class for different wrappers for external functions
    providing the tools to solve F(x)=0 where at least one of the components of
    the output space of a model (F is a vector as x) plus a series of
    inequalities in the input space that are considered by the
    supportBoundingTest
  """

  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    BaseType.__init__(self)
    self.workingDir = runInfoDict['WorkingDir']
    self.__functionFile = ''  # function file name
    self.__actionDictionary = {}  # action dictionary
    # dictionary of implemented actions
    self.__actionImplemented = {
        'residuumSign': False,
        'supportBoundingTest': False,
        'residuum': False,
        'gradient': False
    }
    self.__inputVariables = []  # list of variables' names' given in input (xml)
    self.__inputFromWhat = {}  # dictionary of input data type
    self.__inputFromWhat['dict'] = self.__inputFromDict
    #self.__inputFromWhat['Data']         = self.__inputFromData
    self.printTag = 'FUNCTIONS'

  def _readMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this
      specialized class and initialize some variables based on the inputs
      received.
      @ In, xmlNode, xml.etree.ElementTree.Element, XML element node that represents the portion of the input that belongs to this class
      @ Out, None
    """
    if 'file' in xmlNode.attrib.keys():
      self.functionFile = xmlNode.attrib['file']
      # get the module to load and the filename without path
      moduleToLoadString, self.functionFile = utils.identifyIfExternalModelExists(
          self, self.functionFile, self.workingDir)
      # import the external function
      importedModule = utils.importFromPath(moduleToLoadString,
                                            self.messageHandler.getDesiredVerbosity(self) > 1)
      if not importedModule:
        self.raiseAnError(IOError, 'Failed to import the module ' + moduleToLoadString +
                          ' supposed to contain the function: ' + self.name)
      #here the methods in the imported file are brought inside the class
      for method in importedModule.__dict__.keys():
        if method in [
            '__residuumSign__', '__residuumSign', 'residuumSign', '__supportBoundingTest__',
            '__supportBoundingTest', 'supportBoundingTest', '__residuum__', '__residuum',
            'residuum', '__gradient__', '__gradient', 'gradient'
        ]:
          if method in ['__residuumSign__', '__residuumSign', 'residuumSign']:
            self.__residuumSign = importedModule.__dict__[method]
            self.__actionDictionary['residuumSign'] = self.__residuumSign
            self.__actionImplemented['residuumSign'] = True
          if method in ['__supportBoundingTest__', '__supportBoundingTest', 'supportBoundingTest']:
            self.__supportBoundingTest = importedModule.__dict__[method]
            self.__actionDictionary['supportBoundingTest'] = self.__supportBoundingTest
            self.__actionImplemented['supportBoundingTest'] = True
          if method in ['__residuum__', '__residuum', 'residuum']:
            self.__residuum = importedModule.__dict__[method]
            self.__actionDictionary['residuum'] = self.__residuum
            self.__actionImplemented['residuum'] = True
          if method in ['__gradient__', '__gradient', 'gradient']:
            self.__gradient = importedModule.__dict__[method]
            self.__actionDictionary['gradient'] = self.__gradient
            self.__actionImplemented['gradient'] = True
        else:
          #custom
          self.__actionDictionary[method] = importedModule.__dict__[method]
          self.__actionImplemented[method] = True
    else:
      self.raiseAnError(
          IOError,
          'No file name for the external function has been provided for external function ' +
          self.name + ' of type ' + self.type)
    cnt = 0
    for child in xmlNode:
      if child.tag == 'variable':
        execCommand('self.' + child.text + ' = None', self=self)
        self.__inputVariables.append(child.text)
        cnt += 1
        if len(child.attrib.keys()) > 0:
          self.raiseAnError(IOError, 'variable block in the definition of the function ' +
                            self.name + ' should not have any attribute!')
    if cnt == 0:
      self.raiseAnError(IOError,
                        'not variable found in the definition of the function ' + self.name)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['Module file name'] = self.functionFile
    paramDict['The residuum is provided'] = self.__actionImplemented['residuum']
    paramDict['The sign of the residuum is provided'] = self.__actionImplemented['residuumSign']
    paramDict['The gradient is provided'] = self.__actionImplemented['gradient']
    paramDict['The support bonding is provided'] = self.__actionImplemented['supportBoundingTest']
    for key, value in enumerate(self.__actionImplemented):
      if key not in ['residualSign', 'supportBoundingTest', 'residual', 'gradient']:
        paramDict['Custom Function'] = value
    return paramDict

  def getCurrentSetting(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is a temporary value in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary
      Function adds the current settings in a temporary dictionary
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    for key in self.__inputVariables:
      execCommand(
          "object['variable " + str(key) + " has value']=self." + key, self=self, object=paramDict)
    return paramDict

  def __importValues(self, myInput):
    """
      This method makes available the variable values sent in as self.key
      @ In, myInput, object (dataObjects,dict), object from which the data need to be imported
      @ Out, None
    """
    if type(myInput) == dict:
      self.__inputFromWhat['dict'](myInput)
    else:
      self.raiseAnError(IOError, 'Unknown type of input provided to the function ' + str(self.name))

  def __inputFromDict(self, myInputDict):
    """
      This is meant to be used to collect the input directly from a sampler generated input or simply from a generic dictionary
      In case the input comes from a sampler the expected structure is myInputDict['SampledVars'][variable name] = value
      In case it is a generic dictionary the expected structure is myInputDict[variable name] = value
      @ In, myInputDict, dict, dict from which the data need to be imported
      @ Out, None
    """
    if 'SampledVars' in myInputDict.keys():
      inDict = myInputDict['SampledVars']
    else:
      inDict = myInputDict
    for name in self.__inputVariables:
      if name in inDict.keys():
        execCommand('self.' + name + '=object["' + name + '"]', self=self, object=inDict)
      else:
        self.raiseAnError(
            IOError,
            'The input variable ' + name + ' in external function seems not to be passed in')

  def evaluate(self, what, myInput):
    """
      Method that returns the result of the type of action described by 'what'
      @ In, what, string, what action needs to be performed
      @ In, myInput, object (dataObjects,dict), object from which the data need to be imported
      @ Out, response, object, the response of the action defined in what
    """
    self.__importValues(myInput)
    if what not in self.__actionDictionary:
      self.raiseAnError(IOError, 'Method ' + what + ' not defined in ' + self.name)
    response = self.__actionDictionary[what](self)
    return response

  def availableMethods(self):
    """
      Get a list of the callable methods this interface provides
      @ In, None
      @ Out, keys, list, list of available methods in this Function module (imported module)
    """
    return self.__actionDictionary.keys()

  def parameterNames(self):
    """
      Get a list of the variables this function needs
      @ In, None
      @ Out, __inputVariables, list, the parameter names
    """
    return self.__inputVariables[:]


"""
 Interface Dictionary (factory) (private)
"""
__base = 'function'
__interFaceDict = {}
__interFaceDict['External'] = Function
__knownTypes = __interFaceDict.keys()


def knownTypes():
  """
    Returns known types.
    @ In, None
    @ Out, __knownTypes, list, list of known types
  """
  return __knownTypes


needsRunInfo = True


def returnInstance(Type, runInfoDict, caller):
  """
    Returns an object construction pointer from this module.
    @ In, Type, string, requested object
    @ In, caller, object, requesting object
    @ Out, __interFaceDict, instance, instance of the object
  """
  if Type in knownTypes():
    return __interFaceDict[Type](runInfoDict)
  else:
    caller.raiseAnError(NameError, 'FUNCTIONS', 'not known ' + __base + ' type ' + Type)
