
from __future__ import division, print_function, unicode_literals, absolute_import

import Simulation

class NewMode(Simulation.SimulationMode):
  def __init__(self,simulation):
    Simulation.SimulationMode.__init__(self,simulation)
    self.__simulation = simulation

  def modifySimulation(self):
    self.__simulation.runInfoDict['precommand'] = self.__simulation.runInfoDict['precommand']+" python "




