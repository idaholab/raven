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

  Created on Nov 3, 2021

  @author: aalfonsi
"""

import os
import sys

import matplotlib.pyplot as plt

# note: we use this complicated way to find RAVEN because we don't know how RAVEN
# is installed on specific machines; it can be simplified greatly for specific applications
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*3+['framework'])))
sys.path.append(frameworkDir)
thisDir = os.path.abspath(os.path.dirname(__file__))

frameworkTestDir = os.path.abspath(os.path.join(frameworkDir, '../tests', 'framework'))
targetWorkflow = os.path.join(frameworkTestDir, 'test_rom_trainer.xml')

# import Driver for now
import Driver
# import simulation
from Simulation import Simulation as sim
import utils.TreeStructure as TS
# instantiate a RAVEN simulation instance
# we simply instanciate a Simulation instance
ravenSim = sim(frameworkDir, verbosity="all")
# set input files (this can be encapsulated in a method in simulation (together with the XML reader)
ravenSim.setInputFiles([targetWorkflow])
# read tree (XML input)
tree = TS.parse(open(targetWorkflow,'r'))
# read entities (this part can be encapsulated in a simulation method)
ravenSim.XMLpreprocess(tree.getroot(),os.path.dirname(os.path.abspath(targetWorkflow)))
ravenSim.XMLread(tree.getroot(),runInfoSkip=set(["DefaultInputFile"]),xmlFilename=targetWorkflow)
# ready to initiate the simulation
ravenSim.initialize()
# now there are 2 options
# either we run the full simulation using the simulation run
# or we run a step at the time (to actually interact with the simulation)
# lets follow the second approach since the first approach would be
# simply : ravenSim.run()
# get all steps
allSteps = ravenSim.stepSequence()
 
for name in allSteps:
  inputs, step = ravenSim.initiateStep(name)
  #running a step
  ravenSim.executeStep(inputs, step)
  if name == 'test_extract_for_rom_trainer':
    print()
    # acquire and plot data from a data object while we are running the step
    ps = ravenSim.getEntity('DataObjects', 'Pointset_from_database_for_rom_trainer')
    data = ps.asDataset()# see xarray docs
    data.plot.scatter(x="DeltaTimeScramToAux", y="DG1recoveryTime", hue="CladTempThreshold")
    # these will be saved in the working directory set by RAVEN (e.g. ./tests/framework/test_rom_trainer/)
    plt.savefig(os.path.join(thisDir,"firstplot.png"))
    data.plot.scatter(x="DeltaTimeScramToAux", y="CladTempThreshold", hue="DG1recoveryTime")
    plt.savefig(os.path.join(thisDir,"secondplot.png"))
    # modify the data before going in the rom trainer
    data['DeltaTimeScramToAux']*=1.01
# finalize the simulation
ravenSim.finalizeSimulation()
 
