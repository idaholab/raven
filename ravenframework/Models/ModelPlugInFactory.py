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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
from glob import glob
import inspect
from collections import defaultdict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
from .. import PluginManager
#Internal Modules End--------------------------------------------------------------------------------


# load the plugin_directory.xml and use that to establish the paths to the plugins
# for each of the plugin paths, load up the Entities that have models in them
## -> should we do this through some kind of a PluginLoader that registers the objects
##    in other factories?

__basePluginClasses = {'ExternalModel':'ExternalModelPluginBase',
                      'Code': 'CodePluginBase',
                      'PostProcessor': 'PostProcessorPluginBase',
                      'SupervisedLearning': 'SupervisedLearningPlugin',
                      }
__interfaceDict = defaultdict(dict)
__knownTypes = [] # populate through registration

def registerSubtypes(typeName, typeDict):
  """
    Registers new subtypes.
    @ In, typeName, str, name of entity subtype
    @ In, typeDict, dict, dictionary mapping names to models
  """
  for name, subtype in typeDict.items():
    __interfaceDict[typeName][name] = subtype
    __knownTypes.append(name)

def _registerAllPlugins():
  """
    Method to register all plugins
    @ In, None
    @ Out, None
  """
  for baseType, baseName in __basePluginClasses.items():
    plugins = PluginManager.getEntities(baseType)
    registerSubtypes(baseType, plugins)

_registerAllPlugins()

def knownTypes():
  """
    Method to return the list of known model plugins
    @ In, None
    @ Out, __knownTypes, list, the list of known types
  """
  return __knownTypes

def loadPlugin(Type, subType):
  """
    Tries to load the subType to make it available
    @ In, Type, string, the type of plugin main class (e.g. ExternalModel)
    @ In, subType, string, the subType of the plugin specialized class (e.g. CashFlow)
    @ Out, None
  """
  name = subType.split(".")[0]
  PluginManager.finishLoadPlugin(name)
  _registerAllPlugins()

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
  return __interfaceDict[Type][subType]()
