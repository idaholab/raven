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
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import Simulation


class PBSDSHSimulationMode(Simulation.SimulationMode):
  """Implements SimulationMode to add the new mode pbsdsh
  In this mode, the pbs pbsdsh command is used to run new commands.
  """

  def __init__(self,messageHandler):
    """Create a new PBSDSHSimulationMode instance.
    simulation: the Simulation.Simulation object
    """
    Simulation.SimulationMode.__init__(self, messageHandler)
    #Check if in pbs by seeing if environmental variable exists
    self.__in_pbs = "PBS_NODEFILE" in os.environ
    #self.printTag = returnPrintTag('PBSDSH SIMULATION MODE')

  def remoteRunCommand(self, runInfoDict):
    """
      return a command to run remotely or not.
    """
    if self.__in_pbs:
      return None
    return Simulation.createAndRunQSUB(runInfoDict)

  def modifyInfo(self, runInfoDict):
    """ Change the simulation to use pbsdsh as the precommand so that
    pbsdsh is called each time a command is done.
    """
    newRunInfo = {}
    newRunInfo['batchSize'] = runInfoDict['batchSize']
    if self.__in_pbs:
      #Figure out number of nodes and use for batchsize
      nodefile = os.environ["PBS_NODEFILE"]
      lines = open(nodefile,"r").readlines()
      #XXX this is an undocumented way to pass information back
      newRunInfo['Nodes'] = list(lines)
      oldBatchsize =  runInfoDict['batchSize']
      newBatchsize = len(lines) #the batchsize is just the number of nodes
      # of which there are one per line in the nodefile
      if newBatchsize != oldBatchsize:
        newRunInfo['batchSize'] = newBatchsize
        print("changing batchsize from "+ str(oldBatchsize)+" to " +str(newBatchsize))
      print("Using Nodefile to set batchSize:"+str(newRunInfo['batchSize']))
      #Add pbsdsh command to run.  pbsdsh runs a command remotely with pbs
      print('DEBUG precommand',runInfoDict['precommand'])
      newRunInfo['precommand'] = "pbsdsh -v -n %INDEX1% -- %BASE_WORKING_DIR%/pbsdsh_custom/raven_remote.sh out_%CURRENT_ID% %WORKING_DIR% "+ str(runInfoDict['logfileBuffer'])+" "+runInfoDict['precommand']
      newRunInfo['logfilePBS'] = 'out_%CURRENT_ID%'
      if(runInfoDict['NumThreads'] > 1):
        #Add the MOOSE --n-threads command afterwards
        newRunInfo['postcommand'] = " --n-threads=%NUM_CPUS% "+runInfoDict['postcommand']
    return newRunInfo
