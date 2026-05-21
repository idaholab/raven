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
  NSGA-II optimizer implementation.

  NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a Pareto-based
  evolutionary algorithm that combines three specialisations:

  * **Fast non-dominated sorting** divides the population into Pareto fronts
    ranked by dominance depth.
  * **Crowding-distance assignment** estimates how isolated each sample is
    within its front, preserving diversity along the surface.
  * **Elitist survivor selection** merges parents and offspring, then keeps
    the best fronts while trimming overcrowded fronts using crowding distance.

  Those steps, together with tournament-based parent selection biased toward
  low rank/high diversity solutions, are what differentiates NSGA-II from the
  single-objective GA or alternative multi-objective schemes.
"""

from copy import deepcopy

import numpy as np
import xarray as xr

from ..utils import frontUtils
from ..utils.gaUtils import datasetToDataArray
from .MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm


class NSGAII(MultiObjectiveGeneticAlgorithm):
  """
  Multi-objective Genetic Algorithm implementing the NSGA-II variant.

  The algorithm executes the classic NSGA-II loop:

  Flowchart::

        +---------------------------+
        | Evaluate offspring batch  |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Fast non-dominated sort   |
        | (rank fronts F0, F1, …)   |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Assign crowding distance  |
        | within each front         |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Elitist survivor merge    |
        | (parents ∪ offspring)     |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Tournament parent select  |
        | biased by (rank, distance)|
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Crossover / mutation      |
        +---------------------------+

  The heavy lifting for ranking, crowding, and elitist survivor selection is
  provided by :class:`MultiObjectiveGeneticAlgorithm`.  This class adds the
  descriptive scaffolding so readers know the optimizer follows NSGA-II, and
  ensures the recorded flow closely mirrors the original Deb et al. (2002)
  procedure.
  """
  def __init__(self):
    """
    __init__ method.
    @ Out, None.
    """
    super().__init__()
    self.printTag = 'NSGA-II Genetic Algorithm'

  @classmethod
  def getInputSpecification(cls):
    """
    getInputSpecification method.
    @ Out, specs, InputData.ParameterInput, input specification for NSGA-II.
    """
    specs = super(NSGAII, cls).getInputSpecification()
    specs.name = 'NSGAII'
    specs.description = r"""The \xmlNode{NSGAII} optimizer extends \xmlNode{GeneticAlgorithm} to handle multi-objective problems using
                            the Non-dominated Sorting Genetic Algorithm II (NSGA-II). It preserves all configuration options of the
                            base genetic algorithm while enabling Pareto front ranking and crowding-distance based survivor selection
                            to explore trade-offs between conflicting objectives."""
    objective = specs.getSub('objective')
    if objective is not None:
      objective.description = r"""Name of the objective variable(s) to optimize. Provide at least two comma-separated variables
      to define the Pareto front for NSGA-II."""
    return specs

  def handleInput(self, paramInput):
    """
    handleInput method.
    @ In, paramInput, InputData.ParameterInput, input specification for this optimizer.
    @ Out, None.
    """
    super().handleInput(paramInput)
    if not self._isMultiObjective:
      self.raiseAnError(IOError, 'NSGA-II requires at least two objectives. Use GeneticAlgorithm for single-objective problems.')

  def _useRealization(self, info, rlz):
    """
    Process evaluated offspring following the NSGA-II sequence.

    Flowchart::

          + evaluate offspring population
          + fast non-dominated sorting (frontUtils.rankNonDominatedFrontiers)
          + crowding-distance assignment per front
          + elitist survivor selection (parents ∪ offspring -> next gen)
          + tournament parent selection biased by (rank, distance)
          + variation operators (crossover, mutation) to spawn next batch

    This override exists to document the NSGA-II-specific flow while delegating
    the actual mechanics to :class:`MultiObjectiveGeneticAlgorithm`.
    """
    super()._useRealization(info, rlz)

  def _process_generation(self, info, rlz, offspring, offspringMinObjVals,
                          offspringFitVals, offspringConstraintVals):
    """
      Execute the NSGA-II specific update: elitist merge, non-dominated sorting,
      crowding-distance assignment, survivor selection, and spawning of the next
      generation.
    """
    if not self._activeTraj:
      return

    traj = info['traj']

    if self.counter > 1:
      combinedPop = np.vstack([self.pop.data, offspring.data])
      combinedMinObjVals = [self.popMinObjVals[i] + offspringMinObjVals[i]
                         for i in range(len(self._objectiveVar))]
      combinedAges = list(map(lambda x: x + 1, self.popAges)) + [0] * len(offspring)

      popFitValsByObj = [self.popFitVals[key].data.tolist()
                     for key in self.popFitVals.keys()]
      offspringFitValsByObj = [offspringFitVals[key].data.tolist()
                     for key in offspringFitVals.keys()]
      combinedFitValsByObj = np.array([i + j for i, j in zip(popFitValsByObj, offspringFitValsByObj)])
      combinedFitVals = [list(pair) for pair in zip(*combinedFitValsByObj)]

      combinedConstraintVals = np.vstack([self.popConstraintVals.data, offspringConstraintVals.data])

      combinedExternalObjValsBySolution = np.array(
          [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
           for solution in zip(*combinedMinObjVals)], dtype=float)
      minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)
      combinedRanks = frontUtils.rankNonDominatedFrontiers(
          combinedExternalObjValsBySolution,
          minMask=minMask)

      combinedCD = frontUtils.crowdingDistance(
          rank=np.array(combinedRanks),
          popSize=len(combinedRanks),
          fitness=combinedExternalObjValsBySolution)

      objectiveNames = list(self.popFitVals.keys())
      (self.pop,
       self.popRanks,
       self.popAges,
       self.popCrowdingDist,
       self.popMinObjVals,
       self.popFitVals,
       self.popConstraintVals) = self._survivorSelectionInstance(
          age=combinedAges,
          variables=list(self.toBeSampled),
          combinedPop=combinedPop,
          combinedRanks=combinedRanks,
          combinedCD=combinedCD,
          combinedMinObjVals=combinedMinObjVals,
          combinedFitVals=combinedFitVals,
          combinedConstraintVals=combinedConstraintVals,
          popSize=self._populationSize,
          objectiveNames=objectiveNames)
    else:
      currentPopExternalObjValsBySolution = np.array(
          [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
           for solution in zip(*offspringMinObjVals)], dtype=float)
      minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)
      currentPopRanks = frontUtils.rankNonDominatedFrontiers(
          currentPopExternalObjValsBySolution,
          minMask=minMask)
      currentPopCD = frontUtils.crowdingDistance(
          rank=np.array(currentPopRanks),
          popSize=len(currentPopRanks),
          fitness=currentPopExternalObjValsBySolution)

      self.pop = offspring
      self.popFitVals = offspringFitVals
      self.popMinObjVals = offspringMinObjVals
      self.popAges = [0] * len(offspring)
      self.popRanks = xr.DataArray(currentPopRanks,
                                         dims=['rank'],
                                         coords={'rank': np.arange(len(currentPopRanks))})
      self.popCrowdingDist = xr.DataArray(currentPopCD,
                                      dims=['CrowdingDistance'],
                                      coords={'CrowdingDistance': np.arange(len(currentPopCD))})
      self.popConstraintVals = offspringConstraintVals

    self.popAgesArray = np.array(self.popAges)

    if not hasattr(self, 'prevPopInputs') or self.prevPopInputs is None:
      self.prevPopInputs = None

    self._collectOptPointMulti(rlz,
                               self.pop,
                               self.popRanks,
                               self.popCrowdingDist,
                               self.popMinObjVals,
                               self.popFitVals,
                               self.popConstraintVals)

    self._resolveNewGeneration(traj,
                               rlz,
                               info,
                               self.prevPopInputs,
                               self.popMinObjVals,
                               self.popFitVals,
                               self.popConstraintVals,
                               self.popRanks,
                               self.popCrowdingDist)

    parents = self._parentSelectionInstance(self.pop,
                                            variables=list(self.toBeSampled),
                                            fitness=self.popFitVals,
                                            kSelection=self._kSelection,
                                            nParents=self._nParents,
                                            rank=self.popRanks,
                                            crowdDistance=self.popCrowdingDist,
                                            objVar=self._objectiveVar,
                                            isMultiObjective=True)

    childrenXover = self._crossoverInstance(parents=parents,
                                            variables=list(self.toBeSampled),
                                            crossoverProb=self._crossoverProb,
                                            points=self._crossoverPoints,
                                            EQfiles=self._EQcheckfile,
                                            distDict=self.distDict)

    childrenMutated = self._mutationInstance(offspring=childrenXover,
                                             distDict=self.distDict,
                                             locs=self._mutationLocs,
                                             mutationProb=self._mutationProb,
                                             variables=list(self.toBeSampled),
                                             EQfiles=self._EQcheckfile)

    needsRepair = False
    for chrom in range(min(self._nChildren, len(childrenMutated))):
      unique = set(childrenMutated.data[chrom, :])
      if len(childrenMutated.data[chrom, :]) != len(unique):
        for var in self.toBeSampled:
          if (hasattr(self.distDict[var], 'strategy') and
              self.distDict[var].strategy == 'withoutReplacement'):
            needsRepair = True
            break
      if needsRepair:
        break

    if needsRepair:
      children = self._repairInstance(childrenMutated,
                                      variables=list(self.toBeSampled),
                                      distInfo=self.distDict)
    else:
      children = childrenMutated

    children = children[:self._populationSize, :]

    daChildren = xr.DataArray(children,
                              dims=['chromosome', 'Gene'],
                              coords={'chromosome': np.arange(np.shape(children)[0]),
                                      'Gene': list(self.toBeSampled)})

    for i in range(self.batch):
      newRlz = {}
      for _, var in enumerate(self.toBeSampled.keys()):
        newRlz[var] = float(daChildren.loc[i, var].values)
      self._submitRun(newRlz, traj, self.getIteration(traj))

    self.prevPopInputs = deepcopy(self.pop)
