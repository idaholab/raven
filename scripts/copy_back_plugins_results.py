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

import sys, os, shutil
from distutils import dir_util
# get the location of this script
app_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ".."))
plugins_directory = os.path.abspath(os.path.join(app_path, "plugins"))
plugins_test_dir  = os.path.abspath(os.path.join(app_path, "tests","plugins"))
plugins = [name for name in os.listdir(plugins_directory) if os.path.isdir(os.path.join(plugins_directory, name))]

#check if the plugin directory topology is correct
for plugin in plugins:
  source_plugin = os.path.join(plugins_test_dir,plugin)
  plugin_dir = os.path.join(plugins_directory,plugin)
  destination_folder = os.path.join(plugins_directory,plugin,"tests")
  if not os.path.exists(source_plugin):
    raise IOError('The plugin tests for plugin "'+plugin+'" has not been run!')
  # check if plugin exists
  # start copying
  dir_util.copy_tree(source_plugin,destination_folder)
  # remove source
  shutil.rmtree(source_plugin)

