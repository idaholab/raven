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

def singleObjSurvivorSelect(self, info, rlz, traj, offSprings, offSpringFitness, objectiveVal, g):
  """
    process of selecting survivors for single objective problems
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations
    @ In, traj, dict, dictionary of trajectories
    @ In, offSprings, list, list of offsprings
    @ In, offSpringFitness, list, list of offspring fitness
  """
  if self.counter > 1:
    self.population, self.fitness,\
    self.popAge,self.objectiveVal = self._survivorSelectionInstance(age=self.popAge,
                                                                    variables=list(self.toBeSampled),
                                                                    population=self.population,
                                                                    fitness=self.fitness,
                                                                    newRlz=rlz,
                                                                    offSpringsFitness=offSpringFitness,
                                                                    popObjectiveVal=self.objectiveVal)
  else:
    self.population = offSprings
    self.fitness = offSpringFitness
    self.objectiveVal = rlz[self._objectiveVar[0]].data

def multiObjSurvivorSelect(self, info, rlz, traj, offSprings, offSpringFitness, objectiveVal, g):
  """
    process of selecting survivors for multi objective problems
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations
    @ In, traj, dict, dictionary of trajectories
  """
  if self.counter > 1:
    self.population,self.rank, \
    self.popAge,self.crowdingDistance, \
    self.objectiveVal,self.fitness, \
    self.constraintsV                  = self._survivorSelectionInstance(age=self.popAge,
                                                                         variables=list(self.toBeSampled),
                                                                         population=self.population,
                                                                         offsprings=rlz,
                                                                         popObjectiveVal=self.objectiveVal,
                                                                         offObjectiveVal=objectiveVal,
                                                                         popFit = self.fitness,
                                                                         offFit = offSpringFitness,
                                                                         popConstV = self.constraintsV,
                                                                         offConstV = g)
  else:
    self.population = offSprings
    self.fitness = offSpringFitness
    self.constraintsV = g
    # offspringObjsVals for Rank and CD calculation
    offObjVal = []
    for i in range(len(self._objectiveVar)):
      offObjVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))
    # offspringFitVals for Rank and CD calculation
    fitVal           = datasetToDataArray(self.fitness, self._objectiveVar).data
    offspringFitVals = fitVal.tolist()
    offSpringRank = frontUtils.rankNonDominatedFrontiers(np.array(offspringFitVals))
    self.rank     = xr.DataArray(offSpringRank,
                                 dims=['rank'],
                                 coords={'rank': np.arange(np.shape(offSpringRank)[0])})
    offSpringCD           = frontUtils.crowdingDistance(rank=offSpringRank,
                                                        popSize=len(offSpringRank),
                                                        objectives=np.array(offspringFitVals))
    self.crowdingDistance = xr.DataArray(offSpringCD,
                                         dims=['CrowdingDistance'],
                                         coords={'CrowdingDistance': np.arange(np.shape(offSpringCD)[0])})
    self.objectiveVal = []
    for i in range(len(self._objectiveVar)):
      self.objectiveVal.append(list(np.atleast_1d(rlz[self._objectiveVar[i]].data)))
  self._collectOptPointMulti(self.population,
                             self.rank,
                             self.crowdingDistance,
                             self.objectiveVal,
                             self.fitness,
                             self.constraintsV)
  self._resolveNewGenerationMulti(traj, rlz, info)