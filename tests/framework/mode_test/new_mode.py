
from __future__ import division, print_function, unicode_literals, absolute_import

import Simulation

class NewMode(Simulation.SimulationMode):
  def doOverrideRun(self):
    print("######################################## in NewMode ")
    return False



