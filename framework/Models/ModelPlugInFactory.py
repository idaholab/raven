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
Created on November 6, 2017

@author: alfoa

comment: The ModelPlugIn Module is an Handler.
         It inquires all the modules contained in the ./raven/contrib/plugins
         and load the ones that refer to a model, constructing a '__interFaceDict' on the fly
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import inspect
from collections import defaultdict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
#Internal Modules End--------------------------------------------------------------------------------

__moduleInterfaceList = []
startDir = os.path.join(os.path.dirname(__file__),'../../plugins')
for dirr,_,_ in os.walk(startDir):
  __moduleInterfaceList.extend(glob(os.path.join(dirr,"*.py")))
  utils.add_path(dirr)
__moduleImportedList = []
__basePluginClasses  = {'ExternalModel':'ExternalModelPluginBase'}

"""
 Interface Dictionary (factory) (private)
"""
__base                          = 'ModelPlugins'
__interFaceDict                 = defaultdict(dict)
for moduleIndex in range(len(__moduleInterfaceList)):
  if 'class' in open(__moduleInterfaceList[moduleIndex]).read():
    __moduleImportedList.append(utils.importFromPath(__moduleInterfaceList[moduleIndex],False))
    for key,modClass in inspect.getmembers(__moduleImportedList[-1], inspect.isclass):
      for base in modClass.__bases__:
        for ravenEntityName, baseClassName in __basePluginClasses.items():
          if base.__name__ == baseClassName:
            __interFaceDict[ravenEntityName][key] = modClass
            # check the validity of the plugin
            if not modClass.isAvalidPlugin():
              print(modClass)
              raise IOError("The plugin based on the class "+ravenEntityName.strip()+" is not valid. Please check with the Plugin developer!")
__knownTypes = [item for sublist in __interFaceDict.values() for item in sublist]

def knownTypes():
  """
    Method to return the list of known model plugins
    @ In, None
    @ Out, __knownTypes, list, the list of known types
  """
  return __knownTypes

def returnPlugin(Type,subType,caller):
  """
    this allows the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module
    @ In, Type, string, the type of plugin main class (e.g. ExternalModel)
    @ In, subType, string, the subType of the plugin specialized class (e.g. CashFlow)
    @ In, caller, instance, instance of the caller
  """
  if subType not in knownTypes():
    caller.raiseAnError(NameError,'not known '+__base+' type '+Type+' subType '+Type)
  return __interFaceDict[Type][subType]()
