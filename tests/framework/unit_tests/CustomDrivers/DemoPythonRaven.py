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
  Demonstrate running RAVEN in Python workflows.
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# note: we use this complicated way to find RAVEN because we don't know how RAVEN
# is installed on specific machines; it can be simplified greatly for specific applications
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(frameworkDir)

frameworkTestDir = os.path.abspath(os.path.join(frameworkDir, 'tests', 'framework'))
targetWorkflow = os.path.join(frameworkTestDir, 'basic.xml')

# instantiate a RAVEN instance
from ravenframework import Raven
raven = Raven()

# load a workflow
raven.loadWorkflowFromFile(targetWorkflow)

# run the loaded workflow
returnCode = raven.runWorkflow()
# check run completed successfully
if returnCode != 0:
  raise RuntimeError('RAVEN did not run successfully!')

# acquire and plot data from a data object
results = raven.getEntity('DataObjects', 'results')
data = results.asDataset() # see xarray docs

data.plot.scatter(x="v0", y="angle", hue="r")
# uncomment for live plotting
# plt.show()

# run workflow again to replicate results
# change csv file name to avoid overwriting the same file
printer = raven.getEntity('OutStreams', 'to_file')
printer._printer.name = 'to_file2'
returnCode2 = raven.runWorkflow()
# check that re-run was successful
if returnCode != 0:
  raise RuntimeError('RAVEN did not re-run successfully!')

# check that re-running the workflow gives the same result
runOne = pd.read_csv('to_file.csv')
runTwo = pd.read_csv('to_file2.csv')

# check that both have the same columns
runOneList = list(runOne.columns)
runTwoList = list(runTwo.columns)
allOneInTwo = all(x in runTwoList for x in runOneList)
allTwoInOne = all(x in runOneList for x in runTwoList)

if not (allOneInTwo and allTwoInOne):
  raise RuntimeError('Re-running workflow did not reproduce csv columns!')

# check that the values are the same
if not (runOne[runOne.columns].equals(runTwo[runOne.columns]) and runTwo[runTwo.columns].equals(runOne[runTwo.columns])):
  raise RuntimeError('Re-running workflow did not reproduce csv values!')

"""
  <TestInfo>
    <name>framework.demo_python_raven</name>
    <author>talbpaul</author>
    <created>2021-10-14</created>
    <classesTested>PythonRaven</classesTested>
    <description>
       Demo of using PythonRaven in RAVEN workflows.
       Different from unit tests in that this is easier to read and unerstand
    </description>
    <revisions>
      <revision author="dgarrett622" date="2022-05-31">Adding check on re-running workflow.</revision>
    </revisions>
  </TestInfo>
"""
