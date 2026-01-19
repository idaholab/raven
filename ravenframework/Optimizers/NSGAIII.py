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

  NSGA-III (Non-dominated Sorting Genetic Algorithm III) extends NSGA-II to
  many-objective problems by introducing direction vectors (reference points)
  that guide diversity preservation when crowding distance becomes ineffective.
  The algorithm follows the canonical stages described in Deb and Jain (2014):

    1. Evaluate offspring and merge with the current population.
    2. Perform fast non-dominated sorting on the combined set.
    3. Fill survivor slots front by front; when a front cannot be fully
       accommodated, associate its members with reference directions and
       select niches using the reference counting strategy.
    4. Spawn the next generation from the survivors via tournament selection,
       crossover, and mutation.

  This implementation reuses the shared mechanics in
  :class:`MultiObjectiveGeneticAlgorithm` and specialises only the steps that
  depend on the reference-direction niching logic.
"""

from copy import deepcopy
from math import comb

import numpy as np
import xarray as xr

from ..utils import frontUtils
from ..utils.gaUtils import datasetToDataArray
from .MultiObjectiveGeneticAlgorithm import MultiObjectiveGeneticAlgorithm


class NSGAIII(MultiObjectiveGeneticAlgorithm):
  """
  Multi-objective Genetic Algorithm implementing the NSGA-III variant.
  """

  def __init__(self):
    super().__init__()
    self.printTag = 'NSGA-III Genetic Algorithm'
    self._referenceDirections = None
    self._referenceMeta = {}

  @classmethod
  def getInputSpecification(cls):
    specs = super(NSGAIII, cls).getInputSpecification()
    specs.name = 'NSGAIII'
    specs.description = r"""The \xmlNode{NSGAIII} optimizer augments the multi-objective genetic
                            algorithm with the reference-point niching strategy used in NSGA-III,
                            making it suitable for many-objective problems (three or more
                            objectives)."""
    objective = specs.getSub('objective')
    if objective is not None:
      objective.description = r"""Name of the objective variable(s) to optimise. Provide at least
                                  two comma-separated variables; NSGA-III is most beneficial for
                                  three or more objectives."""
    return specs

  def handleInput(self, paramInput):
    super().handleInput(paramInput)
    if not self._isMultiObjective:
      self.raiseAnError(IOError, 'NSGA-III requires at least two objectives. '
                                 'Use GeneticAlgorithm for single-objective problems.')

  def _useRealization(self, info, rlz):
    """
    Process evaluated offspring following the NSGA-III sequence.

    Flowchart::

          + evaluate offspring population
          + fast non-dominated sorting (rank fronts F0, F1, …)
          + associate survivors with reference directions, perform niching
          + elitist survivor merge (parents ∪ offspring -> next generation)
          + tournament parent selection biased by rank
          + variation operators (crossover, mutation) to spawn next batch
    """
    super()._useRealization(info, rlz)

  # ---------------------------------------------------------------------------
  # Algorithm specialisation
  # ---------------------------------------------------------------------------
  def _process_generation(self, info, rlz, currentPopInputs, currentPop_objvals,
                          currentPopFitness, currentPop_g):
    """
      Execute one NSGA-III survivor-selection and reproduction pass.
    """
    if not self._activeTraj:
      return

    traj = info['traj']
    num_objectives = len(self._objectiveVar)

    # -----------------------------------------------------------------------
    # Combine parents and offspring
    # -----------------------------------------------------------------------
    if self.counter > 1:
      combined_inputs = xr.concat([self.matingPopInputs, currentPopInputs], dim='chromosome')
      combined_fitness = xr.concat([self.matingPopFitness, currentPopFitness], dim='chromosome')
      combined_constraints = xr.concat([self.matingPop_g, currentPop_g], dim='chromosome')
      combined_ages = list(map(lambda x: x + 1, self.matingPopAges)) + [0] * currentPopInputs.sizes['chromosome']
      parent_matrix = np.column_stack(self.matingPopObjVals) if self.matingPopObjVals else np.empty((0, num_objectives))
    else:
      combined_inputs = currentPopInputs.copy(deep=True)
      combined_fitness = currentPopFitness.copy(deep=True)
      combined_constraints = currentPop_g.copy(deep=True)
      combined_ages = [0] * currentPopInputs.sizes['chromosome']
      parent_matrix = np.empty((0, num_objectives))

    offspring_matrix = np.column_stack(currentPop_objvals) if currentPop_objvals else np.empty((0, num_objectives))
    combined_objectives = np.vstack([parent_matrix, offspring_matrix])
    if combined_objectives.size == 0:
      self.raiseAWarning('NSGA-III received empty objective matrix; skipping generation update.')
      return

    # -----------------------------------------------------------------------
    # Generate / update reference directions
    # -----------------------------------------------------------------------
    self._ensure_reference_directions(num_objectives, self._populationSize)

    # -----------------------------------------------------------------------
    # Fast non-dominated sorting on combined population
    # -----------------------------------------------------------------------
    rank_array = np.array(frontUtils.rankNonDominatedFrontiers(combined_objectives, isFitness=True))
    fronts = self._collect_fronts(rank_array)

    # -----------------------------------------------------------------------
    # Normalise objectives using ideal point and intercepts (NSGA-III)
    # -----------------------------------------------------------------------
    ideal_point = np.min(combined_objectives, axis=0)
    translated = combined_objectives - ideal_point
    extreme_points = self._find_extreme_points(translated)
    intercepts = self._compute_intercepts(extreme_points, translated)
    normalized = self._normalize_objectives(translated, intercepts)

    # -----------------------------------------------------------------------
    # Reference-directed survivor selection
    # -----------------------------------------------------------------------
    selected_indices = []
    niche_counts = np.zeros(self._referenceDirections.shape[0], dtype=int)
    distance_cache = {}

    for front in fronts:
      if len(selected_indices) + len(front) <= self._populationSize:
        selected_indices.extend(front)
        assoc, _ = self._associate_points(normalized[front], self._referenceDirections)
        for idx in assoc:
          niche_counts[idx] += 1
      else:
        remainder = self._populationSize - len(selected_indices)
        chosen, cache_update, counts_update = self._niching_selection(
            front,
            remainder,
            normalized,
            selected_indices,
            niche_counts.copy(),
            self._referenceDirections
        )
        selected_indices.extend(chosen)
        niche_counts += counts_update
        distance_cache.update(cache_update)
        break

    selected_indices = np.array(selected_indices, dtype=int)
    if selected_indices.size != self._populationSize:
      self.raiseAWarning('NSGA-III survivor selection filled '
                         f'{selected_indices.size} individuals (expected {self._populationSize}). '
                         'Falling back to truncation.')
      missing = self._populationSize - selected_indices.size
      if missing > 0:
        remaining = np.setdiff1d(np.arange(len(combined_objectives)), selected_indices, assume_unique=True)
        selected_indices = np.concatenate([selected_indices, remaining[:missing]])

    selected_indices = np.sort(selected_indices)

    # -----------------------------------------------------------------------
    # Persist survivor population
    # -----------------------------------------------------------------------
    self.matingPopInputs = combined_inputs.isel(chromosome=selected_indices).copy(deep=True)
    self.matingPopFitness = combined_fitness.isel(chromosome=selected_indices).copy(deep=True)
    self.matingPop_g = combined_constraints.isel(chromosome=selected_indices).copy(deep=True)
    self.matingPopAges = [combined_ages[idx] for idx in selected_indices]
    self.matingPopObjVals = [combined_objectives[selected_indices, j].tolist()
                             for j in range(num_objectives)]
    self.currentPop_ages = np.array(self.matingPopAges)

    selected_ranks = rank_array[selected_indices]
    self.matingPopRanks = xr.DataArray(selected_ranks,
                                       dims=['rank'],
                                       coords={'rank': np.arange(len(selected_ranks))})

    selected_distances = np.zeros(len(selected_indices))
    for offset, idx in enumerate(selected_indices):
      if idx in distance_cache:
        selected_distances[offset] = distance_cache[idx]
    self.matingPopCD = xr.DataArray(selected_distances,
                                    dims=['CrowdingDistance'],
                                    coords={'CrowdingDistance': np.arange(len(selected_distances))})

    self.population = self.matingPopInputs
    self.fitness = self.matingPopFitness
    self.constraintsV = self.matingPop_g
    self.popAge = self.matingPopAges
    self.objectiveVal = self.matingPopObjVals

    self._collectOptPointMulti(rlz,
                               self.matingPopInputs,
                               self.matingPopRanks,
                               self.matingPopCD,
                               self.matingPopObjVals,
                               self.matingPopFitness,
                               self.matingPop_g)

    self._resolveNewGeneration(traj,
                               rlz,
                               info,
                               getattr(self, 'prevPop_inputs', None),
                               self.matingPopObjVals,
                               self.matingPopFitness,
                               self.matingPop_g,
                               self.matingPopRanks,
                               self.matingPopCD)

    # -----------------------------------------------------------------------
    # Parent selection, crossover, mutation, and queue next evaluations
    # -----------------------------------------------------------------------
    parents = self._parentSelectionInstance(self.matingPopInputs,
                                            variables=list(self.toBeSampled),
                                            fitness=self.matingPopFitness,
                                            kSelection=self._kSelection,
                                            nParents=self._nParents,
                                            rank=self.matingPopRanks,
                                            crowdDistance=self.matingPopCD,
                                            objVar=self._objectiveVar,
                                            isMultiObjective=True)

    childrenXover = self._crossoverInstance(parents=parents,
                                            variables=list(self.toBeSampled),
                                            crossoverProb=self._crossoverProb,
                                            points=self._crossoverPoints,
                                            EQfiles=self._EQcheckfile)

    childrenMutated = self._mutationInstance(offSprings=childrenXover,
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

    self.prevPop_inputs = deepcopy(self.matingPopInputs)

  # ---------------------------------------------------------------------------
  # Reference direction helpers
  # ---------------------------------------------------------------------------
  def _ensure_reference_directions(self, num_objectives, population_size):
    """
      Generate reference directions when the population layout changes.
    """
    key = (num_objectives, population_size)
    if self._referenceDirections is not None and self._referenceMeta.get('key') == key:
      return
    directions = self._generate_reference_directions(num_objectives, population_size)
    self._referenceDirections = directions
    self._referenceMeta['key'] = key

  @staticmethod
  def _generate_reference_directions(num_objectives, population_size):
    """
      Generate reference directions using the simplex-lattice design described in
      Deb and Jain (2014), optionally layering two sets of divisions.
    """
    def lattice_points(m, divisions):
      points = []
      def recurse(remaining, depth, acc):
        if depth == m - 1:
          acc.append(remaining)
          points.append(np.array(acc, dtype=float) / divisions)
          acc.pop()
          return
        for i in range(remaining + 1):
          acc.append(i)
          recurse(remaining - i, depth + 1, acc)
          acc.pop()
      recurse(divisions, 0, [])
      return points

    directions = []
    H1 = 0
    while True:
      temp = comb(H1 + num_objectives - 1, num_objectives - 1)
      if temp > population_size or H1 > 20:
        break
      H1 += 1
    H1 = max(H1 - 1, 1)
    directions.extend(lattice_points(num_objectives, H1))

    if len(directions) < population_size:
      H2 = 0
      while True:
        temp = comb(H2 + num_objectives - 1, num_objectives - 1)
        if len(directions) + temp > population_size or H2 > 10:
          break
        H2 += 1
      H2 = max(H2 - 1, 0)
      if H2 > 0:
        second = lattice_points(num_objectives, H2)
        offset = 1.0 / (2.0 * H2)
        directions.extend([(np.array(p) + offset) / (1.0 + offset * num_objectives) for p in second])

    directions = np.asarray(directions, dtype=float)
    if directions.size == 0:
      directions = np.eye(num_objectives)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions = directions / norms
    return directions

  # ---------------------------------------------------------------------------
  # Normalisation helpers
  # ---------------------------------------------------------------------------
  def _find_extreme_points(self, translated):
    """
      Identify extreme points using Achievement Scalarising Functions.
    """
    m = translated.shape[1]
    weights = np.full((m, m), 1e-6)
    np.fill_diagonal(weights, 1.0)
    extreme_points = []
    for weight in weights:
      denom = np.where(weight == 0.0, 1e-12, weight)
      asf = np.max(translated / denom, axis=1)
      idx = int(np.argmin(asf))
      extreme_points.append(translated[idx])
    return np.array(extreme_points)

  def _compute_intercepts(self, extreme_points, translated):
    """
      Compute intercepts for normalisation. Fall back to max-values when the
      hyperplane is ill-conditioned.
    """
    m = translated.shape[1]
    intercepts = None
    if extreme_points.shape[0] == m and np.linalg.matrix_rank(extreme_points) == m:
      try:
        u = np.ones(m)
        solution = np.linalg.solve(extreme_points, u)
        intercepts = 1.0 / solution
      except Exception:
        intercepts = None

    if intercepts is None or np.any(np.isnan(intercepts)) or np.any(intercepts <= 1e-12):
      intercepts = np.max(translated, axis=0)

    intercepts = np.where(intercepts <= 1e-12, 1.0, intercepts)
    return intercepts

  @staticmethod
  def _normalize_objectives(translated, intercepts):
    normalized = translated / intercepts
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    normalized = np.clip(normalized, 0.0, None)
    return normalized

  # ---------------------------------------------------------------------------
  # Niching helpers
  # ---------------------------------------------------------------------------
  @staticmethod
  def _collect_fronts(rank_array):
    fronts = []
    current = 1
    max_rank = int(np.max(rank_array))
    while current <= max_rank:
      idx = np.where(rank_array == current)[0]
      if idx.size:
        fronts.append(idx.tolist())
      current += 1
    return fronts

  @staticmethod
  def _associate_points(normalized_points, reference_dirs):
    """
      Associate each point with the nearest reference direction and return the
      direction index together with the perpendicular distance.
    """
    if normalized_points.size == 0:
      return np.array([], dtype=int), np.array([], dtype=float)

    proj = np.dot(normalized_points, reference_dirs.T)
    direction_norms = np.linalg.norm(reference_dirs, axis=1)
    direction_norms[direction_norms == 0.0] = 1.0
    proj = proj / direction_norms

    norm_sq = np.sum(np.square(normalized_points), axis=1, keepdims=True)
    distances_sq = norm_sq - np.square(proj)
    distances_sq = np.clip(distances_sq, 0.0, None)

    assoc_indices = np.argmin(distances_sq, axis=1)
    perpendicular_dist = np.sqrt(distances_sq[np.arange(len(distances_sq)), assoc_indices])
    return assoc_indices, perpendicular_dist

  def _niching_selection(self, front, slots, normalized, selected_indices,
                         niche_counts, reference_dirs):
    """
      NSGA-III niching procedure for partially filled fronts.
    """
    # work on a float copy to allow use of np.inf while tracking niche pressure
    rho = niche_counts.astype(float, copy=True)
    selected = []
    distance_cache = {}
    counts_update = np.zeros_like(niche_counts, dtype=int)

    if selected_indices:
      assoc_selected, _ = self._associate_points(normalized[selected_indices], reference_dirs)
      for idx in assoc_selected:
        rho[idx] += 1

    last_front_points = normalized[front]
    assoc_front, dist_front = self._associate_points(last_front_points, reference_dirs)
    candidates = {i: [] for i in range(len(reference_dirs))}
    for local_idx, (global_idx, direction_idx, distance) in enumerate(zip(front, assoc_front, dist_front)):
      candidates[direction_idx].append((global_idx, distance, local_idx))

    considered = set()
    while len(selected) < slots and len(considered) < len(reference_dirs):
      min_count = np.min(rho[np.isfinite(rho)])
      candidate_dirs = [idx for idx, count in enumerate(rho) if count == min_count and np.isfinite(count)]
      if not candidate_dirs:
        break

      chosen_dir = None
      candidate_item = None
      for direction in candidate_dirs:
        pool = [item for item in candidates[direction] if item[0] not in selected]
        if not pool:
          rho[direction] = np.inf
          considered.add(direction)
          continue
        pool.sort(key=lambda entry: (entry[1], entry[0]))
        chosen_dir = direction
        candidate_item = pool[0]
        break

      if candidate_item is None:
        continue

      candidate_idx, distance, _ = candidate_item
      selected.append(candidate_idx)
      distance_cache[candidate_idx] = distance
      rho[chosen_dir] += 1
      counts_update[chosen_dir] += 1
      candidates[chosen_dir] = [entry for entry in candidates[chosen_dir] if entry[0] != candidate_idx]

    if len(selected) < slots:
      remaining = [idx for idx in front if idx not in selected]
      remaining.sort()
      selected.extend(remaining[:slots - len(selected)])

    return selected, distance_cache, counts_update
