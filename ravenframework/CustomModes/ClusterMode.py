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
  Shared base class for cluster (node-file based) SimulationModes such as the
  Slurm and PBS modes. This module intentionally defines NO modeName /
  modeClassName, so the CustomModes discovery mechanism skips it.
"""

from ravenframework import Simulation
from ravenframework.CustomModes import ClusterUtils


class ClusterSimulationMode(Simulation.SimulationMode):
  """
    Base class for SimulationModes that distribute runs over the nodes of a
    scheduler allocation described by a node file (one line per processor).
    Subclasses perform the scheduler-specific node discovery and remote
    submission; the batch sizing, node-file splitting and precommand assembly
    live here (previously triplicated across the Slurm/PBS/MPI modes).
  """

  def _modifyInfoForCluster(self, runInfoDict, nodeFileName, mpiParams=None,
                            createPrecommand=True, splitNodeFiles=True,
                            clusterName="the cluster"):
    """
      Shared implementation of modifyInfo for node-file based cluster modes.
      @ In, runInfoDict, dict, the original runInfo
      @ In, nodeFileName, str or None, path of the node file describing the
        allocation (None: not inside an allocation, run on the local machine)
      @ In, mpiParams, list(str), optional, extra parameters for mpiexec
      @ In, createPrecommand, bool, optional, whether to prepend the mpiexec
        precommand (False leaves the existing precommand untouched)
      @ In, splitNodeFiles, bool, optional, whether to write the per-batch
        node_%INDEX% files when batchSize > 1 (False when the launcher, e.g.
        srun, assigns resources itself)
      @ Out, newRunInfo, dict, of modified values
    """
    newRunInfo = {}
    newRunInfo['batchSize'] = runInfoDict['batchSize']
    numMPI = runInfoDict['NumMPI']
    if nodeFileName is not None:
      self.raiseADebug('Setting up remote nodes based on "{}"'.format(nodeFileName))
      lines = ClusterUtils.readNodeFile(nodeFileName)
      if len(lines) == 0:
        self.raiseAnError(IOError, 'Node file "{}" is empty! Cannot determine '
                          'the nodes available on {}.'.format(nodeFileName, clusterName))
      #XXX This is an undocumented way to pass information back
      # (JobHandler strips these lines, so keep the newline-terminated form)
      newRunInfo['Nodes'] = [line + "\n" for line in lines]
      oldBatchsize = runInfoDict['batchSize']
      newBatchsize, changed = ClusterUtils.computeBatchSize(len(lines), numMPI, oldBatchsize)
      if changed:
        newRunInfo['batchSize'] = newBatchsize
        self.raiseAWarning("changing batchsize from "+str(oldBatchsize)+" to "
                           +str(newBatchsize)+" to fit on "+str(len(lines))+" processors")
      newBatchsize = newRunInfo['batchSize']
      self.raiseADebug('Batch size is "{}"'.format(newBatchsize))
      if newBatchsize > 1 and splitNodeFiles:
        #need to split node lines so that numMPI processors are available per run,
        #then give each index a separate file
        ClusterUtils.writeNodeSubFiles(lines, newBatchsize, numMPI, runInfoDict['WorkingDir'])
        nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
      elif splitNodeFiles:
        #If only one batch just use the original node file
        nodeCommand = runInfoDict["NodeParameter"]+" "+nodeFileName
      else:
        #the launcher (e.g. srun) assigns the resources itself
        nodeCommand = " "
    else:
      #Not inside an allocation and no node file supplied in the input.
      #TODO, we don't have a way to know which machines it can run on
      # in this case, so just distribute it over the local machine:
      nodeCommand = " "

    # Create the mpiexec pre command
    # Note, with defaults the precommand is "mpiexec -f nodeFile -n numMPI"
    if createPrecommand:
      newRunInfo['precommand'] = ClusterUtils.buildMPIPrecommand(
        runInfoDict["MPIExec"], mpiParams or [], nodeCommand, numMPI,
        runInfoDict['precommand'])
    else:
      newRunInfo['precommand'] = runInfoDict['precommand']
    if runInfoDict['NumThreads'] > 1:
      newRunInfo['threadParameter'] = runInfoDict['threadParameter']
      #add number of threads to the post command.
      newRunInfo['postcommand'] = " {} {}".format(newRunInfo['threadParameter'], runInfoDict['postcommand'])
    self.raiseAMessage("precommand: "+newRunInfo['precommand']+", postcommand: "
                       +newRunInfo.get('postcommand', runInfoDict['postcommand']))
    return newRunInfo
