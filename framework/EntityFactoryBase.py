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
Created March 15, 2020

@author: talbpaul
"""
import abc

from utils import utils

class EntityFactory(object):
  """
    Provides structure for entity factory
  """
  #############
  # API
  def __init__(self, name=None, needsRunInfo=False, returnInputParameter=False):
    """
      Constructor.
      @ In, name, str, optional, base entity name (e.g. Sampler)
      @ In, returnInputParameter, bool, optional, whether this entity can use inputParams (otherwise xml)
      @ Out, None
    """
    self.name = None                          # name of entity, e.g. Sampler
    self.needsRunInfo = needsRunInfo                 # whether entity needs run info
    self.returnInputParameter = returnInputParameter # use xml or inputParams
    self._registeredTypes = {}                       # registered types for this entity
    self._pluginFactory = None                       # plugin factory, if any; provided by Simulation

  def registerPluginFactory(self, factory):
    """
      Creates link to plugin factory
      @ In, factory, module, PluginFactory
      @ Out, None
    """
    self._pluginFactory = factory

  def registerType(self, name, obj):
    """
      Registers class as type of this entity
      @ In, name, str, name by which entity should be known
      @ In, obj, object, class definition
      @ Out, None
    """
    # TODO check for duplicates?
    # if name in self._registeredTypes:
    #   raise RuntimeError(f'Duplicate entries in "{self.name}" Factory type "{name}": '+
    #                      f'{self._registeredTypes[name]}, {obj}')
    self._registeredTypes[name] = obj

  def registerAllSubtypes(self, baseType, alias=None):
    """
      Registers all inheritors of the baseType as types by classname for this entity.
      @ In, baseType, object, base class type (e.g. Sampler.Sampler)
      @ In, alias, dict, optional, alias names to use for registration names
      @ Out, None
    """
    if alias is None:
      alias = {}
    for obj in utils.getAllSubclasses(baseType):
      name = alias.get(obj.__name__, obj.__name__)
      self.registerType(name, obj)

  def unregisterSubtype(self, name):
    """
      Remove type from registry.
      @ In, name, str, name of subtype
      @ Out, None
    """
    self._registeredTypes.pop(name, None)

  def knownTypes(self):
    """
      Returns known types.
      @ Out, __knownTypes, list, list of known types
    """
    # NOTE: plugins might not be listed if they haven't been loaded yet!
    return self._registeredTypes.keys()

  def returnClass(self, Type, caller):
    """
      Returns an object construction pointer from this module.
      @ In, Type, string, requested object
      @ In, caller, object, requesting object
      @ Out, __interFaceDict, instance, instance of the object
    """
    # is this from an unloaded plugin?
    # return class from known types
    try:
      return self._registeredTypes[Type]
    except KeyError:
      # otherwise, error
      msg = f'"{self.name}" module does not recognize type "{Type}"; '
      msg += f'known types are: {self.knownTypes()}'
      caller.raiseAnError(NameError, msg)

  def returnInstance(self, Type, runInfo, caller, **kwargs):
    """
      Returns an instance pointer from this module.
      @ In, Type, string, requested object
      @ In, caller, object, requesting object
      @ In, kwargs, dict, additional keyword arguments to constructor
      @ Out, __interFaceDict, instance, instance of the object
    """
    if self.needsRunInfo:
      return self.returnClass(Type, caller)(runInfo, **kwargs)
    else:
      return self.returnClass(Type, caller)(**kwargs)

  #############
  # UTILITIES
  # NOTE: FUTURE, load plugins
  # def _checkPluginLoaded(self, typeName):
  #   """
  #     Checks if the requested entity is from a plugin (has '.' in type name), and if so loads plugin if it isn't already
  #     @ In, typeName, str, name of entity to check (e.g. MonteCarlo or MyPlugin.MySampler)
  #     @ Out, None
  #   """
  #   if self._pluginFactory is not None and '.' in typeName:
  #     pluginName, remainder = typeName.split('.', maxsplit=1)
  #     new = self._pluginFactory.finishLoadPlugin(pluginName)