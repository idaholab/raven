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

  Created Apr,3,2024
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

# @profile
def singleObjSurvivorSelect(self, info, rlz, traj, offspring, offspringFitness, objectiveVal, g):
  """
    Process of selecting survivors for single objective problems.

    FIXED: Now uses matingPop* variable names for consistency with multi-objective.

    @ In, self, Instance of GeneticAlgorithm
    @ In, info, dict, dictionary of information
    @ In, rlz, xr.Dataset, dictionary of realizations
    @ In, traj, int, trajectory identifier
    @ In, offspring, xr.DataArray, offspring indiciduals
    @ In, offspringFitness, xr.Dataset, fitness of offspring
    @ In, objectiveVal, list, objective values of offspring
    @ In, g, xr.DataArray, constraint data
    @ Out, None (updates self.matingPop* variables)
  """
  if self.counter > 1:
    # FIXED: Use new variable names (matingPop* instead of old names)
    self.matingPopInputs, self.matingPopFitness, \
    self.matingPopAges, self.matingPopObjVals = self._survivorSelectionInstance(
        age=self.matingPopAges,
        variables=list(self.toBeSampled),
        population=self.matingPopInputs,
        fitness=self.matingPopFitness,
        objVar=self._objectiveVar[0],
        newRlz=rlz,
        offspringFitness=offspringFitness,
        popObjectiveVal=self.matingPopObjVals
    )
  else:
    # First generation: offspring becomes mating population
    self.matingPopInputs = offspring
    self.matingPopFitness = offspringFitness
    # FIXED: Handle objectiveVal properly - it's a list of lists
    self.matingPopObjVals = objectiveVal[0] if isinstance(objectiveVal, list) and len(objectiveVal) > 0 else rlz[self._objectiveVar[0]].data
    self.matingPopAges = [0] * len(offspring)
    # FIXED: Also initialize matingPop_g for first generation
    self.matingPop_g = g

def multiObjSurvivorSelect(self, info, rlz, traj, offSprings, offSpringsFitness, objectiveVal, g):
  """
    process of selecting survivors for multi-objective problems
    @ In, self, instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations (including values of all objectives)
    @ In, traj, dict, dictionary of trajectories
    @ In, offSprings, list, list of offspring individuals
    @ In, offSpringFitness, list, list of fitness values for offspring individuals
    @ In, objectiveVal, list, values of the objectives (for ranking and crowding distance calculation)
    @ In, g, xr.DataArray, constraint data
  """
  # if self.counter > 1:
  #   self.population,self.rank, \
  #   self.popAge,self.crowdingDistance, \
  #   self.objectiveVal,self.fitness, \
  #   self.constraintsV                  = self._survivorSelectionInstance(age=self.popAge,
  #                                                                        variables=list(self.toBeSampled),
  #                                                                        population=self.population,
  #                                                                        offspring=rlz,
  #                                                                        popObjectiveVal=self.objectiveVal,
  #                                                                        offObjectiveVal=objectiveVal,
  #                                                                        popFit = self.fitness,
  #                                                                        offFit = offSpringFitness,
  #                                                                        popConstV = self.constraintsV,
  #                                                                        direction=self._minMax,
  #                                                                        offConstV = g)
  # else:
  #   self.population = offSprings
  #   self.fitness = offSpringFitness
  #   self.constraintsV = g
pass