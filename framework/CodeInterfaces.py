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
Created on April 14, 2014

@author: alfoa

comment: The CodeInterface Module is an Handler.
         It inquires all the modules contained in the folder './CodeInterfaces'
         and load them, constructing a '__interFaceDict' on the fly
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

__moduleInterfaceList = []
startDir = os.path.join(os.path.dirname(__file__), 'CodeInterfaces')
for dirr, _, _ in os.walk(startDir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr, "*.py")))
  utils.add_path(dirr)
__moduleImportedList = []
"""
 Interface Dictionary (factory) (private)
"""
__base = 'Code'
__interFaceDict = {}
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(
        utils.importFromPath(__moduleInterfaceList[moduleIndex], False))
    for key, modClass in inspect.getmembers(__moduleImportedList[-1],
                                            inspect.isclass):
      # in this way we can get all the class methods
      classMethods = [
          method for method in dir(modClass)
          if callable(getattr(modClass, method))
      ]
      if 'createNewInput' in classMethods:
        __interFaceDict[key.replace("Interface", "")] = modClass
__knownTypes = list(__interFaceDict.keys())


def knownTypes():
  """
    Method to return the list of known code interfaces' type
    @ In, None
    @ Out, __knownTypes, list, the list of known types
  """
  return __knownTypes


def returnCodeInterface(Type, caller):
  """
    this allows the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module
    @ In, Type, string, the type of code interface to instanciate
    @ In, caller, instance, instance of the caller
  """
  if Type not in knownTypes():
    caller.raiseAnError(NameError, 'not known ' + __base + ' type ' + Type)
  return __interFaceDict[Type]()
