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
Created on October 23, 2019

@author: talbpaul

comment: Superseding the ModelPluginFactory, this factory collects the entities from plugins
         and makes them available to the various entity factories in RAVEN.
"""
from __future__ import absolute_import
import os
import sys
import inspect
import importlib
from collections import defaultdict

from .PluginBaseClasses import PluginBase
from .utils import xmlUtils

_filePath = os.path.dirname(__file__)
## custom errors
class PluginError(RuntimeError):
  """
    Custom error for run-time issues with plugins.
  """
  pass

## Method definitions
def loadPlugins(path, catalogue):
  """
    Loads plugins and their entities into factory storage.
    @ In, path, str, location of plugins folder
    @ In, catalogue, str, location of plugins directory xml file
    @ Out, None
  """
  pluginsRoot, _ = xmlUtils.loadToTree(catalogue)
  for pluginNode in pluginsRoot:
    name = pluginNode.find('name').text
    location = pluginNode.find('location').text
    if location is None:
      raise PluginError('Installation is corrupted for plugin "{}". Check raven/plugins/plugin_directory.xml and try reinstalling using raven/scripts/intall_plugin')
    if name is None:
      name = os.path.basename(location)
    print('Loading plugin "{}" at {}'.format(name, location))
    module = loadPluginModule(name, location)
    loadEntities(name, module)

#Stores the name of the plugin, and the (spec, plugin, loaded)
_delayedPlugins = {}

def loadPluginModule(name, location):
  """
    Loads the plugin as a package
    @ In, name, str, name of plugin package
    @ In, location, str, path to plugin main directory
    @ Out, plugin, module instance, root module of package
  """
  # TODO for now assume all entities are named in the package
  # load the package loading specs
  spec = importlib.util.spec_from_file_location(name, os.path.join(location, '__init__.py'))
  if spec is None:
    raise PluginError('Plugin "{}" does not appear to be set up as a package!'.format(name))
  # instance of the module
  plugin = importlib.util.module_from_spec(spec)
  # add it to the namespace TODO is this a good idea?
  sys.modules[spec.name] = plugin
  # load the module
  #Save plugin so spec.loader.exec_module(plugin) can be called later
  _delayedPlugins[name] = (spec, plugin, False)
  print(' ... successfully imported "{}" ...'.format(name))
  return plugin

def finishLoadPlugin(name):
  """
    Finishes loading the plugin. This takes time, so should
    only be used when the plugin is needed (such as when it is encountered in
    the input file.
    @ In, name, str, name of the plugin package
    @ Out, available, bool, if True then plugin is available to use
  """
  if name not in _delayedPlugins:
    #Try to find module in system (such as from pip install)
    #Note that loadEntities (below) checks if there is a subclass of
    # PluginBase.PluginBase before using it, so this is safe.
    spec = importlib.util.find_spec(name)
    if spec is None:
      return False
    plugin = importlib.util.module_from_spec(spec)
    _delayedPlugins[name] = (spec, plugin, False)
  spec, plugin, loaded = _delayedPlugins[name]
  if not loaded:
    spec.loader.exec_module(plugin)
    loadEntities(name, plugin)
    #Set loaded flag to true to prevent reloading
    _delayedPlugins[name] =  (spec, plugin, True)
  return True

def loadEntities(name, plugin):
  """
    Loads the RAVEN entities in a package.
    Uses inheritance to determine what plugin entities belong where
    @ In, name, str, name of plugin package
    @ In, plugin, module instance, root module of package
    @ Out, None
  """
  # get the modules that are part of this package
  for pluginMemberName, pluginMember in inspect.getmembers(plugin):
    if inspect.ismodule(pluginMember):
      # get the classes that are part of the module
      ## only get classes that inherit from the PluginBase class
      for candidateName, candidate in inspect.getmembers(pluginMember):
        if inspect.isclass(candidate) and inspect.getmodule(candidate) == pluginMember \
                                      and issubclass(candidate, PluginBase.PluginBase):
          # find the first parent class that is a PluginBase class
          ## NOTE [0] is the class itself
          for parent in inspect.getmro(candidate)[1:]:
            if issubclass(parent, PluginBase.PluginBase):
              break
            # guaranteed to be found, I think, so no else
          # check validity
          if not candidate.isAValidPlugin():
            raise PluginError('Invalid plugin entity: "{}" from plugin "{}"'.format(candidateName, name))
          registerName = '{}.{}'.format(name, candidateName)
          # register locally # TODO need?
          _pluginEntities[candidate.entityType][registerName] = candidate
          # register with corresponding factory
          candidate.getInterfaceFactory().registerType(registerName, candidate)
          print(' ... registered "{}" as a "{}" RAVEN entity.'.format(registerName, candidate.entityType))

def getEntities(entityType=None):
  """
    Provides loaded entities.
    @ In, entityType, str, optional, class of entity to load (e.g. ExternalModel)
    @ Out, _pluginEntities, dict, name: class of plugin entities
  """
  if entityType is None:
    return _pluginEntities
  return _pluginEntities.get(entityType, {})

## factory loading

# storage for available entities
# structure is _pluginEntities[raven entity name] = {name: import command} (TODO check this is good)
_pluginEntities = defaultdict(dict)

# load plugins directory and collect plugins
pluginsPath = os.path.join(_filePath, '..', 'plugins')
if os.path.isdir(pluginsPath): #If false, then probably running as pip package
  # use "catalogue" to differentiate between "path" and "directory"
  pluginsCatalogue = os.path.abspath(os.path.join(pluginsPath, 'plugin_directory.xml'))
  # if no installed plugins, report and finish; otherwise, load plugins
  if os.path.isfile(pluginsCatalogue):
    loadPlugins(pluginsPath, pluginsCatalogue)
  else:
    print('PluginFactory: No installed plugins detected.')
