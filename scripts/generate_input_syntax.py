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
import sys, os

# get the location of this script
app_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), ".."))

# Set the name of the application here and moose directory relative to the application
app_name = 'RAVEN'

MOOSE_DIR = os.path.abspath(os.path.join(app_path, '..' 'moose'))
FRAMEWORK_DIR = os.path.abspath(os.path.join(app_path, '..', 'moose', 'ravenframework'))
#### See if MOOSE_DIR is already in the environment instead
if os.environ.has_key("MOOSE_DIR"):
  MOOSE_DIR = os.environ['MOOSE_DIR']
  FRAMEWORK_DIR = os.path.join(MOOSE_DIR, 'ravenframework')
if os.environ.has_key("FRAMEWORK_DIR"):
  FRAMEWORK_DIR = os.environ['FRAMEWORK_DIR']

sys.path.append(FRAMEWORK_DIR + '/scripts/syntaxHTML')
import genInputFileSyntaxHTML

# this will automatically copy the documentation to the base directory
# in a folder named syntax
genInputFileSyntaxHTML.generateHTML(app_name, app_path, sys.argv, FRAMEWORK_DIR)
