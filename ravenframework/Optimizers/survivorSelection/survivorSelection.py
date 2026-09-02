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
  NOTE: this file only exists to call methods in survivorSelectors.py, making for a confusing and convoluted call stack. - rollnk

  Created Apr,3,2024
  @authors: Mohammad Abdo, Junyung Kim
"""

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import dataArrayToDict, datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

# @profile

def singleObjSurvivorSelect(self, info, rlz, traj, individuals, individualFitness, objectiveVal, g):
  """
    process of selecting survivors for single objective problems
    @ In, self, Instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations
    @ In, traj, dict, dictionary of trajectories
    @ In, individuals, list, list of individuals
    @ In, individualFitness, list, list of individual fitness
    @ In, objectiveVal, list, floats of objective values
    @ In, g, xr.DataArray, constraint data
  """
  if individualFitness is not None:
    for i in range(individuals.shape[0]):
      self._sampledPopulationInfo[tuple(individuals[i].data)] = individualFitness.to_dataarray()[:,i]

  if self.counter > 1:
    self.matingPopInputs, self.matingPopFitness,\
    self.matingPopAges,self.matingPopObjVals = self._survivorSelectionInstance(age=self.matingPopAges,
                                                                    variables=list(self.toBeSampled),
                                                                    population=self.matingPopInputs,
                                                                    fitness=self.matingPopFitness,
                                                                    objVar = self._objectiveVar[0],
                                                                    newRlz=rlz,
                                                                    individualsFitness=individualFitness,
                                                                    popObjectiveVal=self.matingPopObjVals)
  else:
    self.matingPopInputs = individuals
    self.matingPopFitness = individualFitness
    self.matingPopObjVals = rlz[self._objectiveVar[0]].data

def multiObjSurvivorSelect(self, info, rlz, traj, individuals, individualFitness, objectiveVal, g):
  """
    process of selecting survivors for multi-objective problems
    @ In, self, instance of GeneticAlgorithm. Also information to return is added to this
    @ In, info, dict, dictionary of information
    @ In, rlz, dict, dictionary of realizations (including values of all objectives)
    @ In, traj, dict, dictionary of trajectories
    @ In, individuals, list, list of individual individuals
    @ In, individualFitness, list, list of fitness values for individual individuals
    @ In, objectiveVal, list, values of the objectives (for ranking and crowding distance calculation)
    @ In, g, xr.DataArray, constraint data
  """
  if individualFitness is not None:
    for i in range(individuals.shape[0]):
      self._sampledPopulationInfo[tuple(individuals[i].data)] = individualFitness.to_dataarray()[:,i]

  if self.counter > 1:
    self.matingPopInputs,self.matingPopRanks, \
    self.matingPopAges,self.matingPopCD, \
    self.matingPopObjVals,self.matingPopFitness, \
    self.matingPop_g                  = self._survivorSelectionInstance(age=self.matingPopAges,
                                                                         variables=list(self.toBeSampled),
                                                                         population=self.matingPopInputs,
                                                                         individuals=rlz,
                                                                         popObjectiveVal=self.matingPopObjVals,
                                                                         offObjectiveVal=objectiveVal,
                                                                         popFit = self.matingPopFitness,
                                                                         offFit = individualFitness,
                                                                         popConstV = self.matingPop_g,
                                                                         direction=self._minMax,
                                                                         offConstV = g)
  else:
    self.matingPopInputs = individuals
    self.matingPopFitness = individualFitness
    self.matingPopObjVals = objectiveVal
    self.matingPop_g = g
