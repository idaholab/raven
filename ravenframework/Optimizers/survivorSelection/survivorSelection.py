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
import xarray as xr
from ravenframework.utils import frontUtils
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import dataArrayToDict, datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

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
    @ In, constraintVals, xr.DataArray, constraint data
    @ Out, None (updates self.pop* variables)
  """
  if self.counter > 1:
    # Survivor selection returns the new population; keep both legacy and new attributes in sync.
    self.pop, self.popFitVals, \
    self.popAges, self.popMinObjVals = self._survivorSelectionInstance(
        age=self.popAges,
        variables=list(self.toBeSampled),
        population=self.pop,
        popFitVals=self.popFitVals,
        objVar=self._objectiveVar[0],
        newRlz=rlz,
        offspringFitVals=offspringFitVals,
        popMinObjVals=self.popMinObjVals
    )
  else:
    # First generation: offspring becomes the current population
    self.pop = offspring
    self.popFitVals = offspringFitVals
    baseObj = minObjVals[0] if isinstance(minObjVals, list) and len(minObjVals) > 0 else rlz[self._objectiveVar[0]].data
    self.popMinObjVals = list(np.atleast_1d(baseObj))
    self.popAges = [0] * len(offspring)
  self.popConstraintVals = constraintVals

  # Mirror legacy attribute names to keep downstream logic functional.
  self.population = self.pop
  self.fitVals = self.popFitVals
  self.popAge = self.popAges
  self.minObjVals = self.popMinObjVals
  self.constraintVals = self.popConstraintVals

def multiObjSurvivorSelect(self, info, rlz, traj, offspring, offspringFitVals, minObjVals, constraintVals):
  """
    Process of selecting survivors for multi-objective problems.
    Multi-objective survivor selection is handled by the GeneticAlgorithm flow;
    this stub is kept for compatibility with older call sites.
    @ In, self, instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations (including values of all objectives)
    @ In, traj, dict, dictionary of trajectories
    @ In, offspring, list, list of offspring individuals
    @ In, offspringFitVals, list, list of fitness values for offspring individuals
    @ In, minObjVals, list, values of the objectives (for ranking and crowding distance calculation)
    @ In, constraintVals, xr.DataArray, constraint data
  """
  pass
