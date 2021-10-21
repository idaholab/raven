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
  Testing running RAVEN in Python workflows.
"""

import os
import sys

frameworkDir='/Users/mandd/projects/raven/framework/'

sys.path.append(frameworkDir)

import matplotlib.pyplot as plt
from utils import utils
import utils.TreeStructure as TS
import warnings

if not __debug__:
  warnings.filterwarnings("ignore")
else:
  warnings.simplefilter("default", DeprecationWarning)

import os
import sys
import time
import threading
import traceback
import xml.etree.ElementTree as ET

import builtins
try:
  builtins.profile
except (AttributeError,ImportError):
  # profiler not preset, so pass through
  builtins.profile = lambda f: f

#warning: this needs to be before importing h5py
os.environ["MV2_ENABLE_AFFINITY"]="0"

frameworkDir = '/Users/mandd/projects/raven/framework/'

# library handler is in scripts
sys.path.append(os.path.join(frameworkDir, '..', "scripts"))
import library_handler as LH
sys.path.pop() #remove scripts path for cleanliness

from utils import utils
import utils.TreeStructure as TS
utils.find_crow(frameworkDir)
utils.add_path(os.path.join(frameworkDir,'contrib','AMSC'))
utils.add_path(os.path.join(frameworkDir,'contrib'))
##TODO REMOVE PP3 WHEN RAY IS AVAILABLE FOR WINDOWS
utils.add_path_recursively(os.path.join(frameworkDir,'contrib','pp'))
#Internal Modules
from Simulation import Simulation
from Application import __QtAvailable
from Interaction import Interaction
#Internal Modules

targetWorkflow = 'basic.xml'

# create Simulation instance
from Simulation import Simulation
simulation = Simulation(frameworkDir)

# load a workflow
simulation.loadWorkflowFromFile(targetWorkflow)

# run the loaded workflow
returnCode = simulation.run()
# check run completed successfully
#if returnCode != 0:
#  raise RuntimeError('RAVEN did not run successfully!')

# acquire and plot data from a data object
results = simulation.getEntity('DataObjects', 'results')
data = results.asDataset() # see xarray docs

data.plot.scatter(x="v0", y="angle", hue="r")
# uncomment for live plotting
plt.show()

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
  </TestInfo>
"""