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
  Implementation of survivorSelection step for new generation
  selection process in Genetic Algorithm.

  Created Apr 3, 2024
  @authors: Mohammad Abdo, Junyung Kim
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

def singleObjSurvivorSelect(self, info, rlz, traj, offspring, offspringFitVals, minObjVals, constraintVals):
  """
    Process of selecting survivors for single objective problems.
    @ In, self, Instance of GeneticAlgorithm
    @ In, info, dict, dictionary of information
    @ In, rlz, xr.Dataset, dictionary of realizations
    @ In, traj, int, trajectory identifier
    @ In, offspring, xr.DataArray, offspring individuals
    @ In, offspringFitVals, xr.Dataset, fitness of offspring
    @ In, minObjVals, list, minimization-space objective values of offspring
    @ In, constraintVals, xr.DataArray, constraint values of the offspring
    @ Out, None (updates self.population and self.pop* variables)
  """
  if self.counter > 1:
    # constraint rows travel with their individuals, so popConstraintVals stays aligned with population
    self.population, self.popFitVals, \
    self.popAges, self.popMinObjVals, self.popConstraintVals = self._survivorSelectionInstance(
        age=self.popAges,
        variables=list(self.toBeSampled),
        population=self.population,
        popFitVals=self.popFitVals,
        objVar=self._objectiveVar[0],
        newRlz=rlz,
        offspringFitVals=offspringFitVals,
        popMinObjVals=self.popMinObjVals,
        popConstraintVals=self.popConstraintVals,
        offspringConstraintVals=constraintVals
    )
  else:
    # First generation: offspring becomes the current population
    self.population = offspring
    self.popFitVals = offspringFitVals
    baseObj = minObjVals[0] if isinstance(minObjVals, list) and len(minObjVals) > 0 else rlz[self._objectiveVar[0]].data
    self.popMinObjVals = list(np.atleast_1d(baseObj))
    self.popAges = [0] * len(offspring)
    self.popConstraintVals = constraintVals
