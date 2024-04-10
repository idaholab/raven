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
  Demonstrate RAVEN running RAVEN in Python workflows.
"""

import os, sys
import shutil
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# note: we use this complicated way to find RAVEN because we don't know how RAVEN
# is installed on specific machines; it can be simplified greatly for specific applications
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(frameworkDir)

# instantiate a RAVEN instance
from ravenframework import Raven
raven = Raven()

# load workflow XML file
raven.loadWorkflowFromFile('basic.xml')

# Edit the inner workflow executable to specify the raven_framework driver to use based on how the
# framework is installed on the machine. This covers the cases of a traditional source install and a
# pip install. If "raven_framework" is found on the path, assume a pip install is being used and use
# that. Otherwise, assume a source install and use a path relative to the framework directory.
# FIXME: If we see more tests needing to be split for different RAVEN installation types, we should
# consider adding the ability to restrict tests to certain installation types. However, the python-
# raven-running-raven setup of the current test is a bit of a special case and similar situations are
# unlikely to arise in more typical workflows.
ravenInnerModel = raven.getEntity('Models', 'raven')
pipPath = shutil.which('raven_framework')
sourcePath = os.path.join(frameworkDir, 'raven_framework')
if pipPath is not None:
  ravenInnerModel.executable = pipPath
elif os.path.exists(sourcePath):
  ravenInnerModel.executable = sourcePath
else:
  raise RuntimeError('Could not find the RAVEN executable. Tried source path: ' + sourcePath +
                     ' and pip path: ' + pipPath)

# run the workflow
returnCode = raven.runWorkflow()
# check for successful run
if returnCode != 0:
  raise RuntimeError('RAVEN did not run successfully!')

# create a simple plot
results = raven.getEntity('DataObjects', 'outer_samples')
data = results.asDataset()
data.plot.scatter(x='mean_y1', y='mean_y2', hue='mean_ans')

# uncomment to see the plot
# plt.show()

"""
  <TestInfo>
    <name>framework.demo_python_raven_running_raven</name>
    <author>dgarrett622</author>
    <created>2022-04-20</created>
    <classesTested>PythonRaven</classesTested>
    <description>
       Demo of using PythonRaven to run RAVEN running RAVEN workflows.
       Different from unit tests in that this is easier to read and unerstand
    </description>
  </TestInfo>
"""
