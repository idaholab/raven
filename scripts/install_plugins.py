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

# This is a utility script to install a plugin in the RAVEN plugin directory
# It takes the following command line arguments
# -s, the plugin directory that needs to be installed
# -f, force the copy if the directory in the destination location already exists
# to run the script use the following command:
#  python install_plugins -s path/to/plugin -f

import sys, os, shutil
# get the location of this script
app_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ".."))
plugins_directory = os.path.abspath(os.path.join(app_path, "plugins"))
found_plugin_argument = False
force_copy = False
exclude_dirs = ['.git', 'raven']
for cnt, commarg in enumerate(sys.argv):
  if commarg == "-s":
    plugin_dir = os.path.abspath(sys.argv[cnt+1])
    found_plugin_argument = True
  if commarg == "-f":
    force_copy = True
  if commarg == "-e":
    exclude_str = sys.argv[cnt+1]
    # strip out whitespace
    exclude_str = "".join(exclude_str.split())
    exclude_dirs = exclude_str.split(',')
if not found_plugin_argument:
  raise IOError('Source directory for plugin installation not found! USE the syntax "-s path/to/plugin/orginal/location"!')

plugin_name = os.path.basename(plugin_dir)

#check if the plugin directory topology is correct
src_dir = os.path.join(plugin_dir,"src")
if not os.path.exists(src_dir):
  raise IOError('The plugin destination folder "'+plugin_dir+'" does not contain a "src" directory!')
if not os.path.isdir(src_dir):
  raise IOError('In the plugin destination folder "'+plugin_dir+'" the "src" target is not a directory!')
tests_dir = os.path.join(plugin_dir,"tests")
if not os.path.exists(tests_dir):
  raise IOError('The plugin destination folder "'+plugin_dir+'" does not contain a "tests" directory!')
if not os.path.isdir(tests_dir):
  raise IOError('In the plugin destination folder "'+plugin_dir+'" the "tests" target is not a directory!')
doc_dir = os.path.join(plugin_dir,"doc")
if not os.path.exists(doc_dir):
  raise IOError('The plugin destination folder "'+plugin_dir+'" does not contain a "doc" directory!')
if not os.path.isdir(doc_dir):
  raise IOError('In the plugin destination folder "'+plugin_dir+'" the "doc" target is not a directory!')
# check if plugin exists
destination_plugin = os.path.join(plugins_directory,plugin_name)
if os.path.exists(destination_plugin) and not force_copy:
  raise IOError('The destination plugin folder already exists:"'+destination_plugin+'". If you want to force the installation, use the command line argument "-f"')
if os.path.exists(destination_plugin):
  shutil.rmtree(destination_plugin)
# start copying
shutil.copytree(
  plugin_dir,
  destination_plugin,
  ignore=shutil.ignore_patterns(*exclude_dirs)
)
print('Installation of plugin "'+plugin_name+'" performed successfully!')

