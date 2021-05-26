#!/usr/bin/env python
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

## TODO update!
# This is a utility script to install a plugin in the RAVEN plugin directory
# It takes the following command line arguments
# -s, the plugin directory that needs to be installed
# -f, force the copy if the directory in the destination location already exists
# -z, print the framework directory
# to run the script use the following command:
#  python install_plugins -s path/to/plugin -f
import os
import sys
import time
import argparse
frameworkDir = os.path.join(os.path.dirname(__file__), '..', 'framework')
sys.path.append(frameworkDir)
from utils import xmlUtils

# python changed the import error in 3.6
if sys.version_info[0] == 3 and sys.version_info[1] >= 6:
  impErr = ModuleNotFoundError
else:
  impErr = ImportError

# python 2.X uses a different capitalization for configparser
try:
  import configparser
except impErr:
  import ConfigParser as configparser

# globals
ravenConfigName = '.ravenconfig.xml'
requiredDirs = ['src', 'doc', 'tests']
pluginTreeFile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plugins', 'plugin_directory.xml'))

def checkValidPlugin(rawLoc):
  """
    Checks that the plugin at "loc" is a valid one.
    @ In, rawLoc, str, path to plugin (possibly relative)
    @ Out, okay, bool, whether the plugin looks fine
    @ Out, msgs, list(str), error messages collected during check.
    @ Out, loc, str, absolute path to plugin
  """
  okay = True
  msgs = []
  loc = os.path.abspath(os.path.expanduser(rawLoc))
  # check existence
  if not os.path.isdir(loc):
    okay = False
    msgs.append('Not a valid directory: {}'.format(loc))
  # only check structure if directory found
  if okay:
    # check for source, doc, tests dirs
    missing = []
    for needDir in requiredDirs:
      fullDir = os.path.join(loc, needDir)
      if not os.path.isdir(fullDir):
        missing.append(needDir)
    if missing:
      okay = False
      for m in missing:
        msgs.append('Required directory missing in {}: {}'.format(fullDir, m))

  return okay, msgs, loc

def writeNewPluginXML(name, location):
  """
    Writes plugin information to XML
    @ In, name, str, name of plugin
    @ In, location, str, directory of plugin
    @ Out, new, xml.etree.ElementTree.Element, new plugin information in xml
  """
  new = xmlUtils.newNode('plugin')
  new.append(xmlUtils.newNode('name', text=name))
  new.append(xmlUtils.newNode('location', text=location))
  # TODO read a config file IN THE PLUGIN to determine what nodes it should include
  ## for example, executable, default excluded dirs, etc
  new.append(xmlUtils.newNode('exclude'))
  return new

def updatePluginXML(root, name, location):
  """
    Update an existing plugin entry with new information
    @ In, root, xml.etree.ElementTree.Element, root of plugin tree
    @ In, name, str, name of plugin
    @ In, location, str, location of plugin on disk
    @ Out, match, xml.etree.ElementTree.Element, updated element
  """
  match = root.findall('./plugin/[name=\'{}\']'.format(name))[0]
  oldPath = match.find('location').text
  # nothing to do if old path and new path are the same!
  if oldPath != location:
    # TODO overwrite or not as an option?
    print('Updating existing location of plugin "{}" from "{}" to "{}"'.format(name, oldPath, location))
    match.find('location').text = location
  return match

def tellPluginAboutRaven(loc):
  """
    Informs plugin about raven framework location
    @ In, loc, str, location of plugin
    @ Out, ravenLoc, str, location of raven
  """
  # check for config file; load up a root element either way
  configFile = os.path.join(loc, ravenConfigName)
  if os.path.isfile(configFile):
    root, _ = xmlUtils.loadToTree(configFile)
  else:
    root = xmlUtils.newNode('RavenConfig')
  # add raven information
  ravenLoc = root.find('FrameworkLocation')
  if ravenLoc is None:
    ravenLoc = xmlUtils.newNode('FrameworkLocation')
    root.append(ravenLoc)
  ravenFrameworkLoc = os.path.abspath(os.path.expanduser(frameworkDir))
  if ravenLoc.text != ravenFrameworkLoc:
    ravenLoc.text = ravenFrameworkLoc
    xmlUtils.toFile(configFile, root)
  return ravenLoc.text

def loadPluginTree():
  """
    Loads the plugin information XML tree
    @ In, None
    @ Out, pluginTreeFile, str, location and name of tree file
    @ Out, root, xml.etree.ElementTree.Element, root of plugin info tree
  """
  # load sources
  if os.path.isfile(pluginTreeFile):
    root, _ = xmlUtils.loadToTree(pluginTreeFile)
  else:
    root = xmlUtils.newNode('plugins')
  return pluginTreeFile, root

def writePluginTree(destination, root):
  """
    Writes plugin info tree to file
    @ In, destination, string, where to write file to
    @ In, root, xml.etree.ElementTree.Element, element to write
    @ Out, None
  """
  xmlUtils.toFile(destination, root, pretty=True)

def getInstalledPlugins():
  """
    Provide a list of installed plugins.
    @ In, None
    @ Out, list((str, loc)), list of installed plugin names and locations
  """
  _, root = loadPluginTree()
  return [(x.find('name').text.strip(), x.find('location').text.strip()) for x in root.findall('./plugin')]

def getPluginLocation(name):
  """
    Return location of named plugin.
    @ In, name, str, name of plugin
    @ Out, loc, str, location of plugin
  """
  _, root = loadPluginTree()
  plugin = root.find('./plugin/[name=\'{}\']'.format(name))
  if plugin is not None:
    return plugin.find('location').text
  return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="RAVEN Plugin Handler",
                                   description="Plugin management tool for RAVEN")
  # for now, the only use is to request a location, so we make that the arugment
  # -> this is used in the run_tests script to find test dirs
  #parser.add_argument('loc', dest='pluginLocReq', required=True, default=None, metavar='plugin_name',
  parser.add_argument('-f', '--find', dest='loc', default=None, metavar='plugin_name',
                      help='provides location of requested plugin')
  parser.add_argument('-l', '--list', dest='list', action='store_true',
                      help='lists installed plugins')
  parser.add_argument('-z', '--framework-dir', dest='framework_dir',
                      action='store_true', help='prints framework directory')

  # no arguments? get some help!
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  if args.framework_dir:
    print(os.path.abspath(frameworkDir))
  # plugins list
  doList = args.list
  if doList:
    plugins = getInstalledPlugins()
    print(' '.join(p[0] for p in plugins))
  # plugin location
  requested = args.loc
  if args.loc:
    loc = getPluginLocation(requested)
    if loc is None:
      raise KeyError('Plugin "{}" not installed!'.format(requested))
    print(loc)
