from __future__ import division, print_function, unicode_literals, absolute_import

import os
import Simulation


class PBSDSHSimulationMode(Simulation.SimulationMode):
  """Implements SimulationMode to add the new mode pbsdsh
  In this mode, the pbs pbsdsh command is used to run new commands.
  """

  def __init__(self,simulation):
    """Create a new PBSDSHSimulationMode instance.
    simulation: the Simulation.Simulation object
    """
    self.__simulation = simulation
    #Check if in pbs by seeing if environmental variable exists
    self.__in_pbs = "PBS_NODEFILE" in os.environ
    #self.printTag = returnPrintTag('PBSDSH SIMULATION MODE')

  def doOverrideRun(self):
    """ Check if the simulation has been run in PBS mode and
    if not the run needs to be overridden so qsub can be called.
    """
    return not self.__in_pbs

  def runOverride(self):
    """ If not in pbs mode, qsub needs to be called. """
    #Check and see if this is being accidently run
    assert self.__simulation.runInfoDict['mode'] == 'pbsdsh' and not self.__in_pbs
    Simulation.createAndRunQSUB(self.__simulation)

  def modifySimulation(self):
    """ Change the simulation to use pbsdsh as the precommand so that
    pbsdsh is called each time a command is done.
    """
    if self.__in_pbs:
      #Figure out number of nodes and use for batchsize
      nodefile = os.environ["PBS_NODEFILE"]
      lines = open(nodefile,"r").readlines()
      self.__simulation.runInfoDict['Nodes'] = list(lines)
      oldBatchsize =  self.__simulation.runInfoDict['batchSize']
      newBatchsize = len(lines) #the batchsize is just the number of nodes
      # of which there are one per line in the nodefile
      if newBatchsize != oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = newBatchsize
        print("changing batchsize from "+ str(oldBatchsize)+" to " +str(newBatchsize))
      print("Using Nodefile to set batchSize:"+str(self.__simulation.runInfoDict['batchSize']))
      #Add pbsdsh command to run.  pbsdsh runs a command remotely with pbs
      print('DEBUG precommand',self.__simulation.runInfoDict['precommand'])
      self.__simulation.runInfoDict['precommand'] = "pbsdsh -v -n %INDEX1% -- %BASE_WORKING_DIR%/pbsdsh_custom/raven_remote.sh out_%CURRENT_ID% %WORKING_DIR% "+ str(self.__simulation.runInfoDict['logfileBuffer'])+" "+self.__simulation.runInfoDict['precommand']
      self.__simulation.runInfoDict['logfilePBS'] = 'out_%CURRENT_ID%'
      if(self.__simulation.runInfoDict['NumThreads'] > 1):
        #Add the MOOSE --n-threads command afterwards
        self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']
      #print('DEBUG precommand',self.__simulation.runInfoDict['precommand'],'postcommand',self.__simulation.runInfoDict['postcommand'])
