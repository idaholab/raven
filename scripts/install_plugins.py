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
import plugin_handler as pluginHandler

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
    okay, msgs, newLoc = pluginHandler.checkValidPlugin(loc)
    if okay:
      print(' ... plugin located at "{}" ...'.format(newLoc))
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

  infoFile, root = pluginHandler.loadPluginTree()

  # add or update plugins from sources
  existing, _ = zip(*pluginHandler.getInstalledPlugins())
  for plugDir in sources:
    name = os.path.basename(plugDir)
    if name in existing:
      match = pluginHandler.updatePluginXML(root, name, plugDir)
      print(' ... plugin "{}" path updated to "{}" ...'.format(name, plugDir))
    else:
      # create a new entry
      new = pluginHandler.writeNewPluginXML(name, plugDir)
      print(' ... plugin "{}" path created as "{}" ...'.format(name, plugDir))
      root.append(new)
      match = new
    if args.exclude:
      match.find('exclude').text = ','.join(x for x in args.exclude)
    # tell plugin about RAVEN
    rLoc = pluginHandler.tellPluginAboutRaven(plugDir)
    print(' ... plugin "{}" informed of RAVEN at "{}"...'.format(name, rLoc))
    ## TODO testing?
    print(' ... plugin "{}" succesfully installed!'.format(name))
  pluginHandler.writePluginTree(infoFile, root)

  # XXX DEBUGG testing
  print(pluginHandler.getPluginLocation('CashFlow'))



