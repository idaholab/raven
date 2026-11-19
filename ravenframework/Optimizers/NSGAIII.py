# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  NSGA-III optimizer implementation.

  NSGA-III (Non-dominated Sorting Genetic Algorithm III, Deb & Jain 2014) extends NSGA-II to
  many-objective problems by replacing crowding distance with a set of structured reference
  directions that guide diversity preservation. One generation performs:

    1. Evaluate offspring and merge with the current population (elitist mu+lambda).
    2. Fast non-dominated sorting on the combined set.
    3. Fill survivor slots front by front; for the first front that does not fully fit, normalize
       the objectives, associate members with reference directions, and niche-select to fill the
       remaining slots.
    4. Spawn the next generation from the survivors via tournament selection, crossover and mutation.

  Every mechanism in step 3 is a mix-and-match operator family (selected under
  <MultiObjectiveGeneticAlgorithm type="NSGA-III">), with defaults that reproduce Deb & Jain (2014):

    <referenceDirections type="dasDennis"/>          (reference-direction set)
    <objectiveNormalization type="hyperplane"/>      (adaptive normalization)
    <association type="perpendicular"/>              (point-to-direction association)
    <niching type="referencePoint"/>                 (niche preservation)

  The non-reference-point mechanics (merge, sorting, parent selection, variation operators) are
  shared with NSGA-II through MultiObjectiveGeneticAlgorithm; NSGA-III specialises only the
  reference-point survivor selection.

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
from copy import deepcopy
import numpy as np
import xarray as xr
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import frontUtils, InputData, InputTypes
from .MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm
from .referenceDirectionHandlers.referenceDirectionHandlers import returnInstance as referenceDirectionReturnInstance
from .normalizationHandlers.normalizationHandlers import returnInstance as normalizationReturnInstance
from .associationHandlers.associationHandlers import returnInstance as associationReturnInstance
from .nichingHandlers.nichingHandlers import returnInstance as nichingReturnInstance
# Internal Modules End------------------------------------------------------------------------------


class NSGAIII(MultiObjectiveGeneticAlgorithm):
  """
    NSGA-III: reference-point-based many-objective genetic algorithm (Deb & Jain 2014). A concrete
    algorithm of MultiObjectiveGeneticAlgorithm, selected via
    <MultiObjectiveGeneticAlgorithm type="NSGA-III">.
  """

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'NSGA-III Genetic Algorithm'
    # reference-direction operator family (see <referenceDirections>); default = Das-Dennis
    self._referenceDirectionType = 'dasDennis'
    self._referenceDirectionInstance = None
    # objective-normalization operator family (see <objectiveNormalization>); default = hyperplane
    self._normalizationType = 'hyperplane'
    self._normalizationInstance = None
    # association operator family (see <association>); default = perpendicular distance
    self._associationType = 'perpendicular'
    self._associationInstance = None
    # niching operator family (see <niching>); default = reference-point (Deb-Jain 2014)
    self._nichingType = 'referencePoint'
    self._nichingInstance = None
    # cached reference directions and the (numObjectives, populationSize) they were built for
    self._referenceDirections = None
    self._referenceDirectionsKey = None

  @classmethod
  def getInputSpecification(cls):
    """
      Define the acceptable input for NSGA-III, extending the multi-objective GA spec with the four
      reference-point operator-family nodes.
      @ In, cls, class, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, the NSGA-III input specification
    """
    specs = super(NSGAIII, cls).getInputSpecification()
    # keep the inherited spec name 'MultiObjectiveGeneticAlgorithm' so it matches the input node tag;
    # NSGA-III is selected via the <MultiObjectiveGeneticAlgorithm type="NSGA-III"> attribute.
    specs.description = r"""The \xmlNode{MultiObjectiveGeneticAlgorithm} with \xmlAttr{type}=\xmlString{NSGA-III}
                            replaces the crowding-distance step of NSGA-II with the reference-point niching
                            strategy of NSGA-III, which keeps the population spread over the front when there
                            are three or more objectives. Its reference-direction, normalization, association
                            and niching steps are each selectable operator families; the defaults reproduce the
                            canonical NSGA-III. More information about NSGA-III can be found in:\\

                            Deb, Kalyanmoy, and Himanshu Jain. ``An evolutionary many-objective optimization
                            algorithm using reference-point-based nondominated sorting approach, part I: solving
                            problems with box constraints.'' IEEE transactions on evolutionary computation 18.4
                            (2014): 577-601."""

    referenceDirections = InputData.parameterInputFactory('referenceDirections', strictMode=True,
        contentType=InputTypes.StringType,
        descr=r"""the structured reference directions on the unit simplex that NSGA-III uses to
                  preserve diversity.""")
    referenceDirections.addParam('type',
        InputTypes.makeEnumType('referenceDirections', 'referenceDirectionType', ['dasDennis']),
        False,
        descr=r"""the reference-direction design. \xmlString{dasDennis} (default) is the Das and Dennis
                  (1998) simplex-lattice with the two-layer construction of Deb and Jain (2014).""")
    specs.addSub(referenceDirections)

    objectiveNormalization = InputData.parameterInputFactory('objectiveNormalization', strictMode=True,
        contentType=InputTypes.StringType,
        descr=r"""how combined-population objectives are normalized before association.""")
    objectiveNormalization.addParam('type',
        InputTypes.makeEnumType('objectiveNormalization', 'objectiveNormalizationType',
                                ['hyperplane', 'simplexSum', 'none']),
        False,
        descr=r"""the normalization scheme. \xmlString{hyperplane} (default) is the Deb and Jain
                  (2014) adaptive ideal-point/extreme-point/intercept normalization;
                  \xmlString{simplexSum} projects onto the unit simplex; \xmlString{none} translates
                  by the ideal point only.""")
    specs.addSub(objectiveNormalization)

    association = InputData.parameterInputFactory('association', strictMode=True,
        contentType=InputTypes.StringType,
        descr=r"""how each solution is associated with a reference direction.""")
    association.addParam('type',
        InputTypes.makeEnumType('association', 'associationType', ['perpendicular']),
        False,
        descr=r"""the association rule. \xmlString{perpendicular} (default) associates with the
                  reference line of minimum perpendicular distance (Deb and Jain, 2014).""")
    specs.addSub(association)

    niching = InputData.parameterInputFactory('niching', strictMode=True,
        contentType=InputTypes.StringType,
        descr=r"""how the final (splitting) front's survivor slots are filled across reference
                  directions.""")
    niching.addParam('type',
        InputTypes.makeEnumType('niching', 'nichingType',
                                ['referencePoint', 'referencePointDeterministic']),
        False,
        descr=r"""the niche-preservation rule. \xmlString{referencePoint} (default) is Deb and Jain
                  (2014) Algorithm 4 (minimum perpendicular distance for an empty niche, a random
                  associated member for an occupied one); \xmlString{referencePointDeterministic}
                  always takes the minimum-distance member for a reproducible, randomness-free
                  survivor set.""")
    specs.addSub(niching)

    objective = specs.getSub('objective')
    if objective is not None:
      objective.description = r"""Name of the objective variable(s) to optimize. Provide at least two
                                  comma-separated variables; NSGA-III is most beneficial for three or
                                  more objectives."""
    return specs

  def handleInput(self, paramInput):
    """
      Read the NSGA-III portion of the input, resolving the four reference-point operator families.
      @ In, paramInput, InputData.ParameterInput, the parsed input for this optimizer
      @ Out, None
    """
    super().handleInput(paramInput)
    if not self._isMultiObjective:
      self.raiseAnError(IOError, 'NSGA-III requires at least two objectives. '
                                 'Use GeneticAlgorithm (single-objective) for one objective.')

    referenceDirectionsNode = paramInput.findFirst('referenceDirections')
    if referenceDirectionsNode is not None:
      self._referenceDirectionType = referenceDirectionsNode.parameterValues.get('type', 'dasDennis')
    self._referenceDirectionInstance = referenceDirectionReturnInstance(self, name=self._referenceDirectionType)

    normalizationNode = paramInput.findFirst('objectiveNormalization')
    if normalizationNode is not None:
      self._normalizationType = normalizationNode.parameterValues.get('type', 'hyperplane')
    self._normalizationInstance = normalizationReturnInstance(self, name=self._normalizationType)

    associationNode = paramInput.findFirst('association')
    if associationNode is not None:
      self._associationType = associationNode.parameterValues.get('type', 'perpendicular')
    self._associationInstance = associationReturnInstance(self, name=self._associationType)

    nichingNode = paramInput.findFirst('niching')
    if nichingNode is not None:
      self._nichingType = nichingNode.parameterValues.get('type', 'referencePoint')
    self._nichingInstance = nichingReturnInstance(self, name=self._nichingType)

  def _useRealization(self, info, rlz):
    """
      Handle one evaluated offspring realization; delegates to the shared multi-objective driver,
      which calls self._processGeneration for the NSGA-III-specific survivor selection.
      @ In, info, dict, identifying information about the realization (step, traj, etc.)
      @ In, rlz, xr.Dataset, evaluated offspring realization for this generation
      @ Out, None
    """
    super()._useRealization(info, rlz)

  def _ensureReferenceDirections(self, numObjectives):
    """
      Build (and cache) the reference directions for the current objective count and population size.
      @ In, numObjectives, int, number of objectives
      @ Out, None
    """
    key = (numObjectives, self._populationSize)
    if self._referenceDirections is not None and self._referenceDirectionsKey == key:
      return
    self._referenceDirections = self._referenceDirectionInstance(self, numObjectives, self._populationSize)
    self._referenceDirectionsKey = key

  def _referencePointSurvival(self, combinedExternalObjVals, combinedRanks):
    """
      NSGA-III survivor selection: fill slots front by front; on the splitting front, normalize,
      associate with reference directions, and niche-select. Returns the survivor indices (into the
      combined population) and a per-survivor distance (perpendicular distance for niched survivors,
      0.0 for members of fully accepted fronts) that is stored in the crowding-distance slot.
      @ In, combinedExternalObjVals, np.ndarray, (nTotal, nObjectives) external-space objective values
      @ In, combinedRanks, np.ndarray or list, non-dominated front rank per combined individual
      @ Out, selectedIndices, np.ndarray, sorted survivor indices into the combined population
      @ Out, survivorDistances, np.ndarray, per-survivor distance aligned to selectedIndices
    """
    numObjectives = combinedExternalObjVals.shape[1]
    self._ensureReferenceDirections(numObjectives)
    normalized = self._normalizationInstance(self, combinedExternalObjVals)

    ranks = np.asarray(combinedRanks)
    fronts = [np.where(ranks == r)[0].tolist() for r in np.unique(ranks[np.isfinite(ranks)])]

    selected = []
    nicheCounts = np.zeros(len(self._referenceDirections), dtype=int)
    distanceCache = {}
    for front in fronts:
      if len(selected) + len(front) <= self._populationSize:
        selected.extend(front)
        if front:
          assoc, _ = self._associationInstance(self, normalized[front], self._referenceDirections)
          for idx in assoc:
            nicheCounts[idx] += 1
      else:
        slots = self._populationSize - len(selected)
        chosen, chosenDistances, countsUpdate = self._nichingInstance(
            self,
            front=front,
            slots=slots,
            normalized=normalized,
            selectedIndices=selected,
            nicheCounts=nicheCounts,
            referenceDirections=self._referenceDirections,
            associate=self._associationInstance)
        selected.extend(chosen)
        nicheCounts = nicheCounts + countsUpdate
        distanceCache.update(chosenDistances)
        break

    selected = np.array(selected, dtype=int)
    # Safety net: if (degenerate) selection under-filled, top up by combined order.
    if selected.size < self._populationSize:
      remaining = np.setdiff1d(np.arange(len(combinedExternalObjVals)), selected, assume_unique=False)
      selected = np.concatenate([selected, remaining[:self._populationSize - selected.size]])
    selected = np.sort(selected[:self._populationSize])

    survivorDistances = np.array([distanceCache.get(idx, 0.0) for idx in selected], dtype=float)
    return selected, survivorDistances

  def _processGeneration(self, info, rlz, offspring, offspringMinObjVals,
                         offspringFitVals, offspringConstraintVals):
    """
      Execute the NSGA-III specific update: elitist merge, non-dominated sorting, reference-point
      survivor selection, and spawning of the next generation. Mirrors the NSGA-II scaffold but
      replaces rank+crowding survival with reference-point survival.
      @ In, info, dict, identifier of the realization being processed
      @ In, rlz, xr.Dataset, evaluated realizations for the current generation
      @ In, offspring, xr.DataArray, offspring individuals to merge with the population
      @ In, offspringMinObjVals, list, minimization-space objective values of the offspring
      @ In, offspringFitVals, xr.Dataset, fitness values of the offspring
      @ In, offspringConstraintVals, xr.DataArray, constraint values of the offspring
      @ Out, None
    """
    if not self._activeTraj:
      return

    traj = info['traj']
    minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)

    if self.counter > 1:
      combinedPop = np.vstack([self.population.data, offspring.data])
      combinedMinObjVals = [self.popMinObjVals[i] + offspringMinObjVals[i]
                            for i in range(len(self._objectiveVar))]
      combinedAges = list(map(lambda x: x + 1, self.popAges)) + [0] * len(offspring)

      popFitValsByObj = [self.popFitVals[key].data.tolist() for key in self.popFitVals.keys()]
      offspringFitValsByObj = [offspringFitVals[key].data.tolist() for key in offspringFitVals.keys()]
      combinedFitValsByObj = np.array([i + j for i, j in zip(popFitValsByObj, offspringFitValsByObj)])
      combinedFitVals = [list(pair) for pair in zip(*combinedFitValsByObj)]

      combinedConstraintVals = np.vstack([self.popConstraintVals.data, offspringConstraintVals.data])

      combinedExternalObjVals = np.array(
          [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
           for solution in zip(*combinedMinObjVals)], dtype=float)
      combinedRanks = self._rankingAlgorithmInstance(
          self,
          objVals=combinedExternalObjVals,
          constraintVals=self._rankingConstraintVals(combinedConstraintVals),
          minMask=minMask,
          epsilon=self._constraintEpsilon,
          fitVals=np.asarray(combinedFitVals, dtype=float))

      selectedIndices, survivorDistances = self._referencePointSurvival(combinedExternalObjVals, combinedRanks)

      self.population = xr.DataArray(combinedPop[selectedIndices],
                                     dims=['chromosome', 'Gene'],
                                     coords={'chromosome': np.arange(len(selectedIndices)),
                                             'Gene': list(self.toBeSampled)})
      self.popAges = [combinedAges[idx] for idx in selectedIndices]
      self.popMinObjVals = [[combinedMinObjVals[j][idx] for idx in selectedIndices]
                            for j in range(len(self._objectiveVar))]
      objectiveNames = list(self.popFitVals.keys())
      survivorFitByObj = combinedFitValsByObj[:, selectedIndices]
      self.popFitVals = xr.Dataset(
          {name: xr.DataArray(survivorFitByObj[j], dims=['chromosome'],
                              coords={'chromosome': np.arange(len(selectedIndices))})
           for j, name in enumerate(objectiveNames)})
      self.popConstraintVals = xr.DataArray(combinedConstraintVals[selectedIndices],
                                            dims=['chromosome', 'Constraint'],
                                            coords={'chromosome': np.arange(len(selectedIndices)),
                                                    'Constraint': np.arange(combinedConstraintVals.shape[1])})
      self.popRanks = xr.DataArray(np.asarray(combinedRanks)[selectedIndices],
                                   dims=['rank'],
                                   coords={'rank': np.arange(len(selectedIndices))})
      self.popCrowdingDist = xr.DataArray(survivorDistances,
                                          dims=['CrowdingDistance'],
                                          coords={'CrowdingDistance': np.arange(len(selectedIndices))})
    else:
      currentPopExternalObjVals = np.array(
          [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
           for solution in zip(*offspringMinObjVals)], dtype=float)
      offspringFitValsBySolution = np.array(
          [offspringFitVals[key].data.tolist() for key in offspringFitVals.keys()], dtype=float).T
      currentPopRanks = self._rankingAlgorithmInstance(
          self,
          objVals=currentPopExternalObjVals,
          constraintVals=self._rankingConstraintVals(offspringConstraintVals.data),
          minMask=minMask,
          epsilon=self._constraintEpsilon,
          fitVals=offspringFitValsBySolution)
      # First generation: the offspring population IS the survivor set; distances are diagnostic 0.0.
      self.population = offspring
      self.popFitVals = offspringFitVals
      self.popMinObjVals = offspringMinObjVals
      self.popAges = [0] * len(offspring)
      self.popRanks = xr.DataArray(currentPopRanks,
                                   dims=['rank'],
                                   coords={'rank': np.arange(len(currentPopRanks))})
      self.popCrowdingDist = xr.DataArray(np.zeros(len(currentPopRanks)),
                                          dims=['CrowdingDistance'],
                                          coords={'CrowdingDistance': np.arange(len(currentPopRanks))})
      self.popConstraintVals = offspringConstraintVals

    self.popAgesArray = np.array(self.popAges)

    if not hasattr(self, 'prevPopInputs') or self.prevPopInputs is None:
      self.prevPopInputs = None

    self._collectOptPointMulti(rlz,
                               self.population,
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

    parents = self._parentSelectionInstance(self.population,
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
                                            distDict=self.distDict)

    childrenMutated = self._mutationInstance(offspring=childrenXover,
                                             distDict=self.distDict,
                                             locs=self._mutationLocs,
                                             mutationProb=self._effectiveMutationProb(),
                                             variables=list(self.toBeSampled))

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

    self.prevPopInputs = deepcopy(self.population)
