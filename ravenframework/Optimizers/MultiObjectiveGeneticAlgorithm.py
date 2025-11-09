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
  Shared functionality for multi-objective genetic algorithms.
"""

from copy import deepcopy

import numpy as np
import xarray as xr

from ..utils import mathUtils, frontUtils
from ..utils.gaUtils import datasetToDataArray
from .GeneticAlgorithm import GeneticAlgorithm
from .constraintHandling.constraintHandling import constraintHandling


class MultiObjectiveGeneticAlgorithm(GeneticAlgorithm):
  """Shared functionality for multi-objective genetic algorithms."""

  convergenceOptions = dict(GeneticAlgorithm.convergenceOptions, **{
      'hypervolume': r""" provides the maximum relative change permitted between consecutive Pareto-front
                        dominated hypervolumes before convergence is declared. Smaller values enforce tighter
                        convergence of the dominated space.""",
      'spread': r""" provides the maximum allowable value of Deb's spread metric (Δ) measuring the distance
                        between extreme and intermediate Pareto points. Once the spread drops below this value
                        the algorithm is considered converged.""",
      'spacing': r""" provides the maximum allowable spacing metric that captures the variance of
                        nearest-neighbour distances among rank-1 solutions. Lower values indicate a uniform
                        distribution and trigger convergence.""",
      'maxSpread': r""" sets the relative-change tolerance for the maximum spread (Euclidean range of objectives)
                        observed between successive Pareto fronts. Convergence occurs once the change is below this value.""",
      'rank1Ratio': r""" specifies the minimum proportion of the population that must belong to the first
                        non-dominated front to declare convergence. The ratio must remain above the provided
                        value for several successive generations."""})

  def __init__(self):
    super().__init__()
    self._canHandleMultiObjective = True
    self.matingPopRanks = None
    self.matingPopCD = None
    self.multiBestPoint = None
    self.multiBestFitness = None
    self.multiBestObjective = None
    self.multiBestConstraint = None
    self.multiBestRank = None
    self.multiBestCD = None
    self.multiBestOutputs = None
    self._populationCache = {}

  def flush(self):
    super().flush()
    self.matingPopRanks = None
    self.matingPopCD = None
    self.multiBestPoint = None
    self.multiBestFitness = None
    self.multiBestObjective = None
    self.multiBestConstraint = None
    self.multiBestRank = None
    self.multiBestCD = None
    self.multiBestOutputs = None
    self._populationCache = {}

  @classmethod
  def getInputSpecification(cls):
    specs = super(MultiObjectiveGeneticAlgorithm, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{MultiObjectiveGeneticAlgorithm} augments \xmlNode{GeneticAlgorithm} with the
                            operators required to evolve Pareto-optimal populations. It enables non-dominated sorting,
                            crowding-distance survivor selection, and multi-objective convergence metrics that are shared
                            by concrete optimizers such as \xmlNode{NSGAII}."""
    objective = specs.getSub('objective')
    if objective is not None:
      objective.description = r"""List the objective variables that jointly define the Pareto front. Two or more
                                  objectives are required when using a multi-objective genetic algorithm."""
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    names = super(MultiObjectiveGeneticAlgorithm, cls).getSolutionExportVariableNames()
    names['rank'] = 'Non-dominated sorting rank for each survivor in the population.'
    names['CD'] = 'Crowding distance used to preserve solution diversity within a front.'
    return names

  def _formatSolutionExportVariableNames(self, acceptable):
    acceptable = super()._formatSolutionExportVariableNames(acceptable)
    extras = set()
    extraVars = set(self.dependentSample.keys())
    if hasattr(self, '_targetEvaluation') and self._targetEvaluation is not None:
      extraVars.update(self._targetEvaluation.getVars('output'))
    if hasattr(self, '_solutionExport') and self._solutionExport is not None:
      outputs = self._solutionExport.getVars('output') or []
      for name in outputs:
        if not isinstance(name, str):
          continue
        if name.startswith('FitnessEvaluation_'):
          extras.add(name)
          base = name[len('FitnessEvaluation_'):]
          if base and not base.startswith('FitnessEvaluation_'):
            extraVars.add(base)
        else:
          extraVars.add(name)
    for var in extraVars:
      if isinstance(var, str) and not var.startswith('FitnessEvaluation_'):
        extras.add(f'FitnessEvaluation_{var}')
    acceptable.update(extras)
    return acceptable

  def _normalizeKeyComponent(self, value):
    """
      Convert a decision-variable value into a hashable, comparable token.
      @ In, value, object, raw value extracted from an evaluation
      @ Out, normalized, object, comparable representation
    """
    if isinstance(value, (xr.DataArray, xr.Dataset)):
      array = np.asarray(value.values)
    else:
      array = np.asarray(value)
    if array.size == 0:
      return None
    scalar = array.flatten()[0]
    if isinstance(scalar, (bytes, str)):
      return str(scalar)
    if isinstance(scalar, (np.bool_, bool)):
      return bool(scalar)
    if isinstance(scalar, (np.integer, int)):
      return int(scalar)
    try:
      float_val = float(scalar)
    except (TypeError, ValueError):
      return str(scalar)
    if np.isnan(float_val):
      return 'nan'
    return round(float_val, 12)

  def _buildChromosomeKey(self, data):
    """
      Build a stable key for identifying chromosomes based on decision variables.
      @ In, data, xr.Dataset or xr.DataArray or dict, container holding decision variables
      @ Out, key, tuple, identifying key or None if incomplete
    """
    genes = list(self.toBeSampled.keys())
    if not genes:
      return None
    key = []
    for var in genes:
      try:
        if isinstance(data, xr.Dataset):
          if var not in data.data_vars:
            return None
          val = data[var].values
        elif isinstance(data, xr.DataArray):
          if 'Gene' in data.coords and var in data.coords['Gene'].values:
            val = data.sel(Gene=var).values
          elif hasattr(data, 'loc'):
            val = data.loc[var].values
          else:
            return None
        elif isinstance(data, dict):
          if var not in data:
            return None
          val = data[var]
        else:
          return None
      except Exception:
        return None
      normalized = self._normalizeKeyComponent(val)
      if normalized is None:
        return None
      key.append(normalized)
    return tuple(key)

  def _sampleToEntry(self, sample):
    """
      Convert an xr.Dataset sample to a plain dictionary of outputs.
      @ In, sample, xr.Dataset, slice corresponding to a single chromosome evaluation
      @ Out, entry, dict, mapping variable names to numpy/python scalars or arrays
    """
    entry = {}
    for name, dataArray in sample.data_vars.items():
      arr = np.asarray(dataArray.values)
      if arr.ndim == 0:
        entry[name] = arr.item()
      else:
        entry[name] = arr.copy()
    return entry

  def _cacheEvaluations(self, dataset):
    """
      Store raw evaluation outputs so survivors keep full data across generations.
      @ In, dataset, xr.Dataset, collection of evaluated chromosomes for this batch
      @ Out, None
    """
    if not isinstance(dataset, xr.Dataset):
      return
    sample_dim = 'RAVEN_sample_ID'
    if sample_dim not in dataset.dims:
      return
    if self._populationCache is None:
      self._populationCache = {}
    count = dataset.sizes.get(sample_dim, 0)
    for idx in range(count):
      sample = dataset.isel({sample_dim: idx})
      key = self._buildChromosomeKey(sample)
      if key is None:
        continue
      self._populationCache[key] = self._sampleToEntry(sample)

  def _retrieveCachedOutputs(self, data, dataset=None):
    """
      Fetch cached outputs for the given chromosome, optionally falling back to a dataset search.
      @ In, data, xr.DataArray or dict, representation of the chromosome
      @ In, dataset, xr.Dataset or None, optional search space for fallback matching
      @ Out, outputs, dict, cached outputs (may be empty)
    """
    key = self._buildChromosomeKey(data)
    if key is None:
      return {}
    if self._populationCache is None:
      self._populationCache = {}
    if key not in self._populationCache and isinstance(dataset, xr.Dataset):
      sample_dim = 'RAVEN_sample_ID'
      if sample_dim in dataset.dims:
        count = dataset.sizes.get(sample_dim, 0)
        for idx in range(count):
          sample = dataset.isel({sample_dim: idx})
          if self._buildChromosomeKey(sample) == key:
            self._populationCache[key] = self._sampleToEntry(sample)
            break
    cached = self._populationCache.get(key)
    if cached is None:
      return {}
    return {var: (val.copy() if isinstance(val, np.ndarray) else val) for var, val in cached.items()}

  def _chromosomeDictFromPopulation(self, population, index):
    """
      Extract decision variables for a specific chromosome index from various containers.
      @ In, population, xr.DataArray or dict, storage of chromosomes
      @ In, index, int, chromosome position to extract
      @ Out, chromo, dict, mapping decision variable -> value
    """
    chromo = {}
    genes = list(self.toBeSampled.keys())
    if isinstance(population, xr.DataArray):
      slice_ = population.isel(chromosome=index)
      for var in genes:
        try:
          if 'Gene' in slice_.coords and var in slice_.coords['Gene'].values:
            val = slice_.sel(Gene=var).values
          else:
            val = slice_.loc[var].values
        except Exception:
          continue
        arr = np.asarray(val)
        if arr.size == 0:
          continue
        chromo[var] = arr.flatten()[0]
    elif isinstance(population, dict):
      for var in genes:
        if var not in population:
          continue
        arr = np.asarray(population[var])
        if arr.size <= index:
          continue
        chromo[var] = arr.flatten()[index]
    return chromo

  def _collectOutputsForPopulation(self, population, count, dataset=None):
    """
      Gather cached outputs for a collection of chromosomes.
      @ In, population, xr.DataArray or dict, survivor representation
      @ In, count, int, number of chromosomes to extract
      @ In, dataset, xr.Dataset or None, fallback data source for matching
      @ Out, outputs, dict(str -> list), collected outputs per requested variable
    """
    if count <= 0 or not hasattr(self, '_solutionExport') or self._solutionExport is None:
      return {}
    exportOutputs = self._solutionExport.getVars('output')
    if not exportOutputs:
      return {}
    exportOutputs = [var for var in exportOutputs if var not in self.toBeSampled and var not in self._objectiveVar]
    if not exportOutputs:
      return {}
    collected = {var: [] for var in exportOutputs}
    for idx in range(count):
      chromo = self._chromosomeDictFromPopulation(population, idx)
      cached = self._retrieveCachedOutputs(chromo, dataset=dataset)
      for var in exportOutputs:
        value = cached.get(var, np.nan)
        if isinstance(value, np.ndarray):
          if value.size == 1:
            value = value.item()
          else:
            value = value.copy()
        collected[var].append(value)
    return collected

  def handleInput(self, paramInput):
    super().handleInput(paramInput)
    if not self._isMultiObjective:
      self.raiseAnError(IOError, 'At least two objectives are required for a multi-objective genetic algorithm.')
    if self._parentSelectionType != 'tournamentSelection':
      self.raiseAnError(IOError, 'Multi-objective genetic algorithms currently support only "tournamentSelection" as <parentSelection>.')
    if self._survivorSelectionType != 'rankNcrowdingBased':
      self.raiseAnError(IOError, 'Multi-objective genetic algorithms require <survivorSelection> to be "rankNcrowdingBased".')

  def _addToSolutionExport(self, traj, rlz, acceptable):
    toAdd = super()._addToSolutionExport(traj, rlz, acceptable)
    if 'rank' in rlz:
      toAdd['rank'] = rlz['rank']
    elif self.multiBestRank is not None:
      toAdd['rank'] = np.atleast_1d(self.multiBestRank)
    if 'CD' in rlz:
      toAdd['CD'] = rlz['CD']
    elif self.multiBestCD is not None:
      toAdd['CD'] = np.atleast_1d(self.multiBestCD)
    return toAdd

  def _collectOptPointMulti(self, rlz, population, rank, CD, objVal, fitness, constraintsV):
    rankOneIDX = np.where(rank.data == 1)[0].tolist()
    optPoints = population[rankOneIDX]
    optObjVal = np.array(objVal)[:, rankOneIDX].T

    fitSet = None
    for count, key in enumerate(fitness.keys()):
      data = fitness[key][rankOneIDX]
      if count == 0:
        fitSet = data.to_dataset(name=key)
      else:
        fitSet[key] = data

    optConstraintsV = constraintsV.data[rankOneIDX]
    optRank = rank.data[rankOneIDX]
    optCD = CD.data[rankOneIDX]

    optPointsDic = {var: np.array(optPoints)[:, i] for i, var in enumerate(population.Gene.data)}
    optConstNew = [list(y) for y in zip(*optConstraintsV)]
    if len(optConstNew) > 0:
      optConstNew = xr.DataArray(optConstNew,
                                 dims=['Constraint', 'Evaluation'],
                                 coords={'Constraint': [y.name for y in (self._constraintFunctions + self._impConstraintFunctions)],
                                         'Evaluation': np.arange(np.shape(optConstNew)[1])})

    self.multiBestPoint = optPointsDic
    self.multiBestFitness = fitSet
    self.multiBestObjective = optObjVal
    self.multiBestConstraint = optConstNew
    self.multiBestRank = optRank
    self.multiBestCD = optCD
    self.multiBestOutputs = self._collectOutputsForPopulation(optPointsDic,
                                                              len(optRank),
                                                              dataset=rlz)
    return optPointsDic

  def _resolveNewGeneration(self, traj, rlz, info, pastPop, objectiveVal, fitness, g, ranks=None, CD=None):
    """
      Handle generation resolution for multi-objective GA variants.

      Flowchart::

            +---------------+
            | non-dom rank  |
            +-------+-------+
                    |
                    v
            +---------------+
            | record fronts |
            +-------+-------+
                    |
                    v
            +---------------+
            | export fronts |
            +-------+-------+
                    |
                    v
            +---------------+
            | update Pareto |
            +---------------+
    """
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving multi-objective generation ...')
    self._stepTracker[traj]['opt'] = (rlz, info)
    acceptable = 'accepted' if self.counter > 1 else 'first'
    converged = self._updateConvergence(traj, rlz, pastPop, acceptable)
    if converged:
      self._closeTrajectory(traj, 'converge', 'converged', self.multiBestObjective)

    if self._writeSteps == 'every':
      pop_size = rlz.sizes.get('RAVEN_sample_ID', 0)
      self.raiseADebug(f"### rlz.sizes['RAVEN_sample_ID'] = {pop_size}")
      solutionExportVars = set()
      solutionExportOutputs = []
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        inputs = self._solutionExport.getVars('input') or []
        solutionExportOutputs = self._solutionExport.getVars('output') or []
        outputs = solutionExportOutputs
        solutionExportVars.update(inputs)
        solutionExportVars.update(outputs)
      solutionExportVars.update(self.dependentSample.keys())
      for i in range(pop_size):
        survivorSlice = self.matingPopInputs.isel(chromosome=i)
        rlzDict = survivorSlice.to_series().to_dict()
        for j in range(len(self._objectiveVar)):
          rlzDict[self._objectiveVar[j]] = self.matingPopObjVals[j][i]
        rlzDict['batchId'] = self.batchId
        rlzDict['rank'] = np.atleast_1d(ranks.data)[i] if ranks is not None else np.atleast_1d(self.matingPopRanks.data)[i]
        rlzDict['CD'] = np.atleast_1d(CD.data)[i] if CD is not None else np.atleast_1d(self.matingPopCD.data)[i]
        if self.matingPopAges is not None:
          rlzDict['age'] = self.matingPopAges[i]
        fitnessContainer = fitness if isinstance(fitness, dict) else self.matingPopFitness
        for fitName in fitnessContainer.keys():
          rlzDict[f'FitnessEvaluation_{fitName}'] = fitnessContainer[fitName].data[i]
        for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
          rlzDict[f'ConstraintEvaluation_{consName}'] = g.data[i, ind]
        cachedOutputs = self._retrieveCachedOutputs(survivorSlice, dataset=rlz)
        for var in solutionExportVars:
          if var in rlzDict:
            continue
          if isinstance(var, str) and var.startswith('FitnessEvaluation_'):
            baseVar = var[len('FitnessEvaluation_'):]
            value = None
            if isinstance(fitnessContainer, dict) and baseVar in fitnessContainer:
              value = fitnessContainer[baseVar].data[i]
            elif hasattr(self.matingPopFitness, 'keys') and baseVar in self.matingPopFitness:
              value = self.matingPopFitness[baseVar].data[i]
            elif baseVar in cachedOutputs:
              value = cachedOutputs[baseVar]
            elif baseVar in rlz.data_vars:
              baseArray = np.asarray(rlz[baseVar].data)
              if baseArray.ndim == 0:
                value = baseArray.item()
              elif baseArray.shape[0] > i:
                value = np.take(baseArray, i, axis=0)
            if value is not None:
              rlzDict[var] = value
            continue
          if var in cachedOutputs:
            rlzDict[var] = cachedOutputs[var]
        self._updateSolutionExport(traj, rlzDict, acceptable, None)

    if acceptable in ('accepted', 'first'):
      exportInputs = self._solutionExport.getVars('input') if self._solutionExport is not None else []
      exportOutputs = self._solutionExport.getVars('output') if self._solutionExport is not None else []
      requestedInputs = set(exportInputs or []) | set(self.toBeSampled.keys())
      requestedOutputs = set(exportOutputs or [])
      bestRlz = {}
      if isinstance(self.multiBestPoint, dict):
        for var in requestedInputs:
          if var in self.multiBestPoint:
            bestRlz[var] = np.atleast_1d(self.multiBestPoint[var])
      if isinstance(self.multiBestOutputs, dict):
        for var, values in self.multiBestOutputs.items():
          if var in requestedOutputs and var not in bestRlz:
            bestRlz[var] = np.asarray(values)
      for i in range(len(self._objectiveVar)):
        bestRlz[self._objectiveVar[i]] = [item[i] for item in self.multiBestObjective]
      bestRlz['rank'] = self.multiBestRank
      bestRlz['CD'] = self.multiBestCD
      if self.multiBestConstraint is not None and len(self.multiBestConstraint) != 0:
        for ind, consName in enumerate(self.multiBestConstraint.Constraint):
          name = consName.item() if hasattr(consName, 'item') else str(consName)
          bestRlz[f'ConstraintEvaluation_{name}'] = self.multiBestConstraint[ind].values
      for fitName in self.multiBestFitness.keys():
        bestRlz[f'FitnessEvaluation_{fitName}'] = self.multiBestFitness[fitName].data
      if isinstance(self.multiBestOutputs, dict):
        for name in requestedOutputs:
          if isinstance(name, str) and name.startswith('FitnessEvaluation_') and name not in bestRlz:
            baseVar = name[len('FitnessEvaluation_'):]
            if baseVar in self.multiBestFitness:
              bestRlz[name] = self.multiBestFitness[baseVar].data
            elif baseVar in self.multiBestOutputs:
              bestRlz[name] = np.asarray(self.multiBestOutputs[baseVar])
      elif hasattr(self.multiBestFitness, 'keys'):
        for name in requestedOutputs:
          if isinstance(name, str) and name.startswith('FitnessEvaluation_') and name not in bestRlz:
            baseVar = name[len('FitnessEvaluation_'):]
            if baseVar in self.multiBestFitness:
              bestRlz[name] = self.multiBestFitness[baseVar].data
      bestRlz.update(self.multiBestPoint)
      self._optPointHistory[traj].append((bestRlz, info))

  def _validateFinalFront(self, candidate):
    try:
      if not hasattr(self, 'matingPopFitness') or self.matingPopFitness is None:
        return
      if not hasattr(self, 'matingPopObjVals') or self.matingPopObjVals is None:
        return

      def _as_array(values):
        return np.asarray(values.data if hasattr(values, 'data') else values, dtype=float)

      pop_obj = np.column_stack([np.atleast_1d(_as_array(vals)) for vals in self.matingPopObjVals])
      multipliers = np.array([self._objMult[obj] for obj in self._objectiveVar], dtype=float)
      pop_obj = pop_obj * multipliers

      try:
        candidate_obj = np.array([float(np.atleast_1d(candidate[obj])[0]) for obj in self._objectiveVar], dtype=float)
      except KeyError:
        self.raiseADebug('Final-front validation skipped: candidate lacks objective entries.')
        return
      candidate_obj = candidate_obj * multipliers

      combined = np.vstack([pop_obj, candidate_obj])
      ranks = np.array(frontUtils.rankNonDominatedFrontiers(combined))
      candidate_rank = ranks[-1]
      best_rank = ranks[:-1].min() if combined.shape[0] > 1 else candidate_rank
      if candidate_rank > best_rank:
        self.raiseAWarning('Final export candidate is dominated by existing population members '
                           f'(rank {candidate_rank} vs best rank {best_rank}).')
    except Exception as exc:
      self.raiseADebug(f'Final-front validation encountered an exception: {exc}')

  def _useRealization(self, info, rlz):
    """Process evaluated offspring for multi-objective optimization."""
    info['step'] = self.counter
    traj = info['traj']
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently GA is single trajectory', 0)
    self.incrementIteration(traj)

    self._cacheEvaluations(rlz)

    files = self.assemblerDict['Files']
    self._EQcheckfile = files if any("EQinput" in sublist for sublist in files) else None

    currentPopInputs = datasetToDataArray(rlz, list(self.toBeSampled))
    currentPop_objvals = [list(np.atleast_1d(rlz[obj].data)) for obj in self._objectiveVar]

    currentPop_g = constraintHandling(self, info, rlz, currentPopInputs,
                                      currentPop_objvals, multiObjective=True)

    norm_rlz = deepcopy(rlz)
    if self._normalizeFitness:
      constrVarsList = self._constraintFunctions + self._impConstraintFunctions
      varsToNormalize = []
      for func in constrVarsList:
        varsToNormalize += func.parameterNames()
      varsToNormalize = set(varsToNormalize + self._objectiveVar)

      self.normScores = {}
      for var in varsToNormalize:
        if self._normalizeFitness == 'zscore':
          self.normScores[var] = (np.mean(rlz[var].to_dataframe().values),
                                  np.std(rlz[var].to_dataframe().values))
          for idx in range(len(rlz[var])):
            norm_rlz[var][idx] = (rlz[var][idx] - self.normScores[var][0]) / self.normScores[var][1]
            if np.isnan(norm_rlz[var][idx]):
              norm_rlz[var][idx] = 0.0

      for i in range(len(currentPop_g)):
        for j in range(len(constrVarsList)):
          denom = self.normScores[constrVarsList[j].parameterNames()[0]][1]
          currentPop_g[i][j] = currentPop_g[i][j] / denom
          if np.isnan(currentPop_g[i][j]):
            currentPop_g[i][j] = 0.0

    currentPopFitness = self._fitnessInstance(norm_rlz,
                                              objVar=self._objectiveVar,
                                              a=self._objCoeff,
                                              b=self._penaltyCoeff,
                                              penalty=None,
                                              constraintFunction=currentPop_g,
                                              constraintNum=self._numOfConst,
                                              type=self._minMax)

    self._process_generation(info,
                             rlz,
                             currentPopInputs,
                             currentPop_objvals,
                             currentPopFitness,
                             currentPop_g)

  def _process_generation(self, info, rlz, currentPopInputs, currentPop_objvals,
                          currentPopFitness, currentPop_g):
    """
      Perform the strategy-specific portion of the multi-objective update.
      Subclasses (e.g. NSGA-II) must implement this method.
    """
    raise NotImplementedError(f'{self.__class__.__name__} must implement "_process_generation".')

  def _checkConvHypervolume(self, traj, **kwargs):
    if len(self._optPointHistory[traj]) < 2:
      return False
    if not hasattr(self, 'matingPopRanks') or not hasattr(self, 'matingPopObjVals'):
      return False

    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    if len(rank1_indices) == 0:
      return False

    current_front = []
    for idx in rank1_indices:
      point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
      current_front.append(point)

    prev_opt, _ = self._optPointHistory[traj][-2]
    if 'rank' not in prev_opt:
      return False
    prev_rank1_indices = np.where(np.array(prev_opt['rank']) == 1)[0]
    if len(prev_rank1_indices) == 0:
      return False

    prev_front = []
    for idx in prev_rank1_indices:
      point = [prev_opt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
      prev_front.append(point)

    all_points = current_front + prev_front
    if not all_points:
      return False
    nadir = [max(p[i] for p in all_points) for i in range(len(self._objectiveVar))]
    reference = [n * 1.1 for n in nadir]

    current_hv = self._computeHypervolume(current_front, reference)
    prev_hv = self._computeHypervolume(prev_front, reference)

    if not hasattr(self, '_hvHistory'):
      self._hvHistory = {}
    if traj not in self._hvHistory:
      self._hvHistory[traj] = []
    self._hvHistory[traj].append(current_hv)

    if mathUtils.compareFloats(prev_hv, 0.0, 1e-12):
      rel_improvement = float('inf')
    else:
      rel_improvement = abs(current_hv - prev_hv) / prev_hv

    threshold = self._convergenceCriteria.get('hypervolume', 0.01)
    converged = rel_improvement < threshold

    self.raiseADebug(self.convFormat.format(
        name='Hypervolume',
        conv=str(converged),
        got=rel_improvement,
        req=threshold))

    return converged

  def _computeHypervolume(self, front, reference):
    if not front:
      return 0.0
    n_objectives = len(front[0])
    if n_objectives == 2:
      return self._hypervolume2D(front, reference)
    if n_objectives == 3:
      return self._hypervolume3D(front, reference)
    return self._hypervolumeWFG(front, reference)

  def _hypervolume2D(self, front, reference):
    sorted_front = sorted(front, key=lambda p: p[0])
    hv = 0.0
    prev_x = reference[0]
    for point in sorted_front:
      width = prev_x - point[0]
      height = reference[1] - point[1]
      hv += width * height
      prev_x = point[0]
    return hv

  def _hypervolume3D(self, front, reference):
    sorted_front = sorted(front, key=lambda p: p[0])
    hv = 0.0
    for i, point in enumerate(sorted_front):
      x_extent = reference[0] - point[0]
      remaining_front = [p[1:] for p in sorted_front[:i + 1]]
      remaining_ref = reference[1:]
      slice_hv = self._hypervolume2D(remaining_front, remaining_ref)
      hv += x_extent * slice_hv
    return hv

  def _hypervolumeWFG(self, front, reference):
    if len(reference) == 1:
      return reference[0] - min(p[0] for p in front)

    sorted_front = sorted(front, key=lambda p: p[-1])
    hv = 0.0
    for i, point in enumerate(sorted_front):
      lower_dim_front = [p[:-1] for p in sorted_front[:i + 1]]
      lower_dim_ref = reference[:-1]
      lower_hv = self._hypervolumeWFG(lower_dim_front, lower_dim_ref)
      height = reference[-1] - point[-1]
      hv += height * lower_hv
    return hv

  def _checkConvSpread(self, traj, **kwargs):
    if not hasattr(self, 'matingPopRanks') or not hasattr(self, 'matingPopObjVals'):
      return False
    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    if len(rank1_indices) < 3:
      return False
    front = []
    for idx in rank1_indices:
      point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
      front.append(point)
    spread = self._computeSpread(front)
    threshold = self._convergenceCriteria.get('spread', 0.5)
    converged = spread < threshold
    self.raiseADebug(self.convFormat.format(
        name='Spread',
        conv=str(converged),
        got=spread,
        req=threshold))
    return converged

  def _computeSpread(self, front):
    n = len(front)
    if n < 2:
      return 0.0
    front = sorted(front, key=lambda x: x[0])
    distances = []
    for i in range(n - 1):
      dist = np.linalg.norm(np.array(front[i + 1]) - np.array(front[i]))
      distances.append(dist)
    ideal = np.array([min(p[i] for p in front) for i in range(len(front[0]))])
    nadir = np.array([max(p[i] for p in front) for i in range(len(front[0]))])
    d_f = np.linalg.norm(np.array(front[0]) - ideal)
    d_l = np.linalg.norm(np.array(front[-1]) - nadir)
    mean_d = np.mean(distances)
    if mathUtils.compareFloats(mean_d, 0.0, 1e-14):
      return 0.0
    spread = (d_f + d_l + sum(abs(d - mean_d) for d in distances)) / (d_f + d_l + (len(distances)) * mean_d)
    return spread

  def _checkConvSpacing(self, traj, **kwargs):
    if not hasattr(self, 'matingPopRanks') or not hasattr(self, 'matingPopObjVals'):
      return False
    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    if len(rank1_indices) < 3:
      return False
    front = []
    for idx in rank1_indices:
      point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
      front.append(point)
    spacing = self._computeSpacing(front)
    threshold = self._convergenceCriteria.get('spacing', 0.5)
    converged = spacing < threshold
    self.raiseADebug(self.convFormat.format(
        name='Spacing',
        conv=str(converged),
        got=spacing,
        req=threshold))
    return converged

  def _computeSpacing(self, front):
    n = len(front)
    if n < 2:
      return 0.0
    distances = []
    for i in range(n):
      point_i = np.array(front[i])
      min_dist = float('inf')
      for j in range(n):
        if i == j:
          continue
        point_j = np.array(front[j])
        dist = np.linalg.norm(point_i - point_j)
        if dist < min_dist:
          min_dist = dist
      distances.append(min_dist)
    mean_dist = np.mean(distances)
    spacing = np.sqrt(np.mean([(d - mean_dist) ** 2 for d in distances]))
    return spacing

  def _checkConvMaxSpread(self, traj, **kwargs):
    if len(self._optPointHistory[traj]) < 2:
      return False
    if not hasattr(self, 'matingPopRanks') or not hasattr(self, 'matingPopObjVals'):
      return False
    rank1_indices = np.where(self.matingPopRanks.data == 1)[0]
    if len(rank1_indices) == 0:
      return False
    current_front = []
    for idx in rank1_indices:
      point = [self.matingPopObjVals[j][idx] for j in range(len(self._objectiveVar))]
      current_front.append(point)

    prev_opt, _ = self._optPointHistory[traj][-2]
    if 'rank' not in prev_opt:
      return False
    prev_rank1 = np.where(np.array(prev_opt['rank']) == 1)[0]
    if len(prev_rank1) == 0:
      return False
    prev_front = []
    for idx in prev_rank1:
      point = [prev_opt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
      prev_front.append(point)

    current_ms = self._computeMaxSpread(current_front)
    prev_ms = self._computeMaxSpread(prev_front)

    if mathUtils.compareFloats(prev_ms, 0.0, 1e-14):
      rel_change = float('inf')
    else:
      rel_change = abs(current_ms - prev_ms) / prev_ms

    threshold = self._convergenceCriteria.get('maxSpread', 0.05)
    converged = rel_change < threshold

    self.raiseADebug(self.convFormat.format(
        name='MaxSpread',
        conv=str(converged),
        got=rel_change,
        req=threshold))

    return converged

  def _computeMaxSpread(self, front):
    n = len(front)
    if n < 2:
      return 0.0
    ranges = []
    for i in range(len(front[0])):
      obj_values = [point[i] for point in front]
      ranges.append(max(obj_values) - min(obj_values))
    ms = np.sqrt(sum(r ** 2 for r in ranges))
    return ms

  def _checkConvRank1Ratio(self, traj, **kwargs):
    if not hasattr(self, 'matingPopRanks'):
      return False
    if not hasattr(self, '_populationSize') or self._populationSize == 0:
      return False
    rank1_count = np.sum(self.matingPopRanks.data == 1)
    ratio = rank1_count / self._populationSize
    if not hasattr(self, '_rank1History'):
      self._rank1History = {}
    if traj not in self._rank1History:
      self._rank1History[traj] = []
    self._rank1History[traj].append(ratio)
    threshold = self._convergenceCriteria.get('rank1Ratio', 0.5)
    stable_generations = 3
    if len(self._rank1History[traj]) < stable_generations:
      converged = False
    else:
      recent_ratios = self._rank1History[traj][-stable_generations:]
      all_above_threshold = all(r >= threshold for r in recent_ratios)
      variation = max(recent_ratios) - min(recent_ratios)
      converged = all_above_threshold and variation < 0.1
    self.raiseADebug(self.convFormat.format(
        name='Rank1Ratio',
        conv=str(converged),
        got=ratio,
        req=threshold))
    return converged
