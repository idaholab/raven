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

# This is a utility script to copy the plugins' tests directory
# from raven/plugins/pluginName/tests to raven/tests/plugins/pluginName/tests location
# it is a temporary script till the regression system in MOOSE does not allow the possibility
# to specify folders where look for additional tests

import sys, os, shutil, warnings
# get the location of this script
app_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ".."))
plugins_directory = os.path.abspath(os.path.join(app_path, "plugins"))
plugins_test_dir  = os.path.abspath(os.path.join(app_path, "tests","plugins"))
plugins = [name for name in os.listdir(plugins_directory) if os.path.isdir(os.path.join(plugins_directory, name))]

#check if the plugin directory topology is correct
for plugin in plugins:
  plugin_dir = os.path.join(plugins_directory,plugin)
  tests_dir = os.path.join(plugin_dir,"tests")
  # check if plugin exists
  if not os.path.exists(tests_dir):
    warnings.warn('The plugin "'+plugin_dir+'" does not contain a "tests" directory!')
  else:
    if not os.path.isdir(tests_dir):
      raise IOError('In the plugin folder "'+plugin_dir+'" the "tests" target is not a directory!')
  if os.path.exists(tests_dir):
    destination_plugin = os.path.join(plugins_test_dir,plugin)
    if os.path.exists(destination_plugin):
      shutil.rmtree(destination_plugin)
    # start copying
    shutil.copytree(tests_dir , destination_plugin ,ignore=shutil.ignore_patterns(".git"))

