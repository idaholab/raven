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

import Simulation

class NewMode(Simulation.SimulationMode):
  def __init__(self,simulation):
    Simulation.SimulationMode.__init__(self,simulation)
    self.__simulation = simulation

  def modifySimulation(self):
    self.__simulation.runInfoDict['precommand'] = self.__simulation.runInfoDict['precommand']+" python "




