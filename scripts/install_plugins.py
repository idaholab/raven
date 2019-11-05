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
# to run the script use the following command:
#  python install_plugins -s path/to/plugin -f
import os
import sys
import shutil
import argparse
frameworkDir = os.path.join(os.path.dirname(__file__), '..', 'framework')
sys.path.append(frameworkDir)
from utils import xmlUtils

# set up command-line arguments
parser = argparse.ArgumentParser(prog="RAVEN Plugin Installer",
                                 description="Used to install external plugins to the RAVEN repository. " +\
                                             "By default creates a path link (use -c to override).")
parser.add_argument('-s', '--source', dest='source_dir', action='append', required=True,
                    help='designate the folder where the plugin is located (e.g. ~/projects/CashFlow). '+\
                         'May be specified multiple times as -s path1 -s path2 -s path3, etc.')
parser.add_argument('-c', '--copy', dest='full_copy', action='store_true',
                    help='fully copy the plugin, do not create a path link')
parser.add_argument('-e', '--exclude', dest='exclude', action='append',
                    help='exclude designated folders from loading in RAVEN.')
args = parser.parse_args()

requiredDirs = ['src', 'doc', 'tests']

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

def _writeNewPluginXML(name, location):
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

def _updatePluginXML(root, name, location):
  """
    Update an existing plugin entry with new information
    @ In, root, xml.etree.ElementTree.Element, root of plugin tree
    @ In, name, str, name of plugin
    @ In, location, str, location of plugin on disk
    @ Out, match, xml.etree.ElementTree.Element, updated element
  """
  match = root.findall('./plugin/name[.=\'{}\']/..'.format(name))[0]
  oldPath = match.find('location').text
  # nothing to do if old path and new path are the same!
  if oldPath != location:
    # TODO overwrite or not as an option?
    print('Updating existing location of plugin "{}" from "{}" to "{}"'.format(name, oldPath, location))
    match.find('location').text = location
  return match

def _tellPluginAboutRaven(name, loc):
  """
    Informs plugin about raven framework location
    @ In, name, str, name of plugin
    @ In, loc, location of plugin
    @ Out, None
  """
  # check for config file; load up a root element either way
  configFile = os.path.join(loc, '.ravenconfig')
  if os.path.isfile(configFile):
    root, _ = xmlUtils.loadToTree(configFile)
  else:
    root = xmlUtils.newNode('RavenConfig')
  # add raven information
  ravenLoc = root.find('FrameworkLocation')
  if ravenLoc is None:
    ravenLoc = xmlUtils.newNode('FrameworkLocation')
    root.append(ravenLoc)
  ravenLoc.text = os.path.abspath(os.path.expanduser(frameworkDir))
  xmlUtils.toFile(configFile, root)

if __name__ == '__main__':
  ### Design notes
  # "Installing" is actually just the process of registering the location of the plugin
  # so that it can be loaded during RAVEN entity loading.
  #
  # In the event a full copy is done, we still register the location to assure all plugins
  # follow the same mechanics.
  #
  sources = []
  failedSources = []
  # check legitimacy of plugin directories
  for s, sourceDir in enumerate(args.source_dir):
    loc = os.path.abspath(sourceDir)
    okay, msgs, newLoc = checkValidPlugin(loc)
    if okay:
      sources.append(newLoc)
    else:
      print('ERROR: Plugin at "{}" had the following error(s):')
      for e in msgs:
        print('          ', e)
      print('       Skipping plugin installation.')
      failedSources.append(sourceDir)

  if args.full_copy:
    # TODO: copy the files
    #       register the plugin in the plugin_directory as being in the plugins folder
    raise NotImplementedError

  # load sources
  pluginTreeFile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plugins', 'plugin_directory.xml'))
  if os.path.isfile(pluginTreeFile):
    root, _ = xmlUtils.loadToTree(pluginTreeFile)
  else:
    root = xmlUtils.newNode('plugins')

  # add or update plugins from sources
  existing = [x.text.strip() for x in root.findall('./plugin/name')]
  for plugDir in sources:
    name = os.path.basename(plugDir)
    if name in existing:
      match = _updatePluginXML(root, name, plugDir)
    else:
      # create a new entry
      new = _writeNewPluginXML(name, plugDir)
      root.append(new)
      match = new
    if args.exclude:
      match.find('exclude').text = ','.join(x for x in args.exclude)
    # tell plugin about RAVEN
    _tellPluginAboutRaven(name, plugDir)
  xmlUtils.toFile(pluginTreeFile, root, pretty=True)




