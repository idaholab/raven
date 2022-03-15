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

from .BaseClasses import MessageUser
from .BaseClasses import InputDataUser
from . import PluginManager
from .utils import utils

class EntityFactory(MessageUser):
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
    super().__init__()
    self.name = name                                 # name of entity, e.g. Sampler
    self.needsRunInfo = needsRunInfo                 # whether entity needs run info
    self.returnInputParameter = returnInputParameter # use xml or inputParams
    self._registeredTypes = {}                       # registered types for this entity
    self._pluginFactory = PluginManager              # plugin factory, if any; provided by Simulation

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
      @ In, alias, dict, optional, alias names to use for registration names as {"ObjectName": "AliasName"}
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

  def returnClass(self, Type):
    """
      Returns an object construction pointer from this module.
      @ In, Type, string, requested object
      @ Out, returnClass, object, class of the object
    """
    # is this from an unloaded plugin?
    # return class from known types
    try:
      return self._registeredTypes[Type]
    except KeyError:
      # is this a request from an unloaded plugin?
      obj = self._checkInUnloadedPlugin(Type)
      if obj is None:
        # otherwise, error
        msg = f'"{self.name}" module does not recognize type "{Type}"; '
        msg += f'known types are: {", ".join(list(self.knownTypes()))}'
        self.raiseAnError(NameError, msg)
      else:
        return obj

  def returnInstance(self, Type, **kwargs):
    """
      Returns an instance pointer from this module.
      @ In, Type, string, requested object
      @ In, kwargs, dict, additional keyword arguments to constructor
      @ Out, returnInstance, instance, instance of the object
    """
    cls = self.returnClass(Type)
    instance = cls(**kwargs)
    return instance

  def collectInputSpecs(self, base):
    """
      Extends "base" to include all specs for all objects known by this factory as children of "base"
      @ In, base, InputData.ParameterInput, starting spec
      @ Out, None
    """
    for name in self.knownTypes():
      cls = self.returnClass(name, None)
      if isinstance(cls, InputDataUser):
        base.addSub(cls.getInputSpecifications())

  def instanceFromXML(self, xml):
    """
      Using the provided XML, return the required instance
      @ In, xml, xml.etree.ElementTree.Element, head element for instance
      @ In, runInfo, dict, info from runInfo
      @ Out, kind, str, name of type of entity
      @ Out, name, str, identifying name of entity
      @ Out, entity, instance, object from factory
    """
    kind = xml.tag
    name = xml.attrib['name']
    entity = self.returnInstance(kind)
    return kind, name, entity


  #############
  # UTILITIES

  def _checkInUnloadedPlugin(self, typeName):
    """
      Checks if the requested entity is from a plugin (has '.' in type name), and if so loads plugin if it isn't already
      @ In, typeName, str, name of entity to check (e.g. MonteCarlo or MyPlugin.MySampler)
      @ Out, _checkInUnloadedPlugin, object, requested object if found or None if not.
    """
    if self._pluginFactory is not None and '.' in typeName:
      pluginName, remainder = typeName.split('.', maxsplit=1)
      loadedNew = self._pluginFactory.finishLoadPlugin(pluginName)
      if not loadedNew:
        return None
      else:
        return self._registeredTypes.get(typeName, None)
