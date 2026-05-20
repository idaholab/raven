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
    """
    __init__ method.
    @ Out, None.
    """
    super().__init__()
    self._canHandleMultiObjective = True
    self.popRanks = None
    self.popCrowdingDist = None
    self.multiBestPoint = None
    self.multiBestFitVals = None
    self.multiBestMinObjVals = None
    self.multiBestConstraintVals = None
    self.multiBestRank = None
    self.multiBestCD = None
    self.multiBestOutputs = None
    self._populationCache = {}

  def flush(self):
    """
    flush method.
    @ Out, None.
    """
    super().flush()
    self.popRanks = None
    self.popCrowdingDist = None
    self.multiBestPoint = None
    self.multiBestFitVals = None
    self.multiBestMinObjVals = None
    self.multiBestConstraintVals = None
    self.multiBestRank = None
    self.multiBestCD = None
    self.multiBestOutputs = None
    self._populationCache = {}

  @classmethod
  def getInputSpecification(cls):
    """
    getInputSpecification method.
    @ Out, None.
    """
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
    """
    getSolutionExportVariableNames method.
    @ Out, None.
    """
    names = super(MultiObjectiveGeneticAlgorithm, cls).getSolutionExportVariableNames()
    names['rank'] = 'Non-dominated sorting rank for each survivor in the population.'
    names['CD'] = 'Crowding distance used to preserve solution diversity within a front.'
    return names

  def _formatSolutionExportVariableNames(self, acceptable):
    """
    _formatSolutionExportVariableNames method.
    @ In, acceptable, set(str), candidate variable names allowed for solution export.
    @ Out, acceptable, set(str), updated acceptable solution export names.
    """
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
    """
    handleInput method.
    @ In, paramInput, InputData.ParameterInput, input specification for this optimizer.
    @ Out, None.
    """
    super().handleInput(paramInput)
    if not self._isMultiObjective:
      self.raiseAnError(IOError, 'At least two objectives are required for a multi-objective genetic algorithm.')
    if self._parentSelectionType != 'tournamentSelection':
      self.raiseAnError(IOError, 'Multi-objective genetic algorithms currently support only "tournamentSelection" as <parentSelection>.')
    if self._survivorSelectionType != 'rankNcrowdingBased':
      self.raiseAnError(IOError, 'Multi-objective genetic algorithms require <survivorSelection> to be "rankNcrowdingBased".')

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
    _addToSolutionExport method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, rlz, dict, realization dictionary for the current generation.
    @ In, acceptable, set(str), candidate variable names allowed for solution export.
    @ Out, toAdd, dict, solution export additions for this realization.
    """
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

  def _collectOptPointMulti(self, rlz, population, rank, CD, minObjVals, fitVals, constraintVals):
    """
    _collectOptPointMulti method.
    @ In, rlz, dict, realization dictionary for the current generation.
    @ In, population, xr.DataArray, population decision vectors (Gene dimension).
    @ In, rank, xr.DataArray, non-dominated sorting ranks per individual.
    @ In, CD, xr.DataArray, crowding distances per individual.
    @ In, minObjVals, array-like, minimization-space objective values per individual and objective.
    @ In, fitVals, dict, fitness values keyed by objective name.
    @ In, constraintVals, xr.DataArray, constraint values per individual.
    @ Out, optPointsDic, dict, rank-1 optimal points keyed by variable.
    """
    rankOneIDX = np.where(rank.data == 1)[0].tolist()
    optPoints = population[rankOneIDX]
    optMinObjVals = np.array(minObjVals)[:, rankOneIDX].T

    fitSet = None
    for count, key in enumerate(fitVals.keys()):
      data = fitVals[key][rankOneIDX]
      if count == 0:
        fitSet = data.to_dataset(name=key)
      else:
        fitSet[key] = data

    optConstraintVals = constraintVals.data[rankOneIDX]
    optRank = rank.data[rankOneIDX]
    optCD = CD.data[rankOneIDX]

    optPointsDic = {var: np.array(optPoints)[:, i] for i, var in enumerate(population.Gene.data)}
    optConstNew = [list(y) for y in zip(*optConstraintVals)]
    if len(optConstNew) > 0:
      optConstNew = xr.DataArray(optConstNew,
                                 dims=['Constraint', 'Evaluation'],
                                 coords={'Constraint': [y.name for y in (self._constraintFunctions + self._impConstraintFunctions)],
                                         'Evaluation': np.arange(np.shape(optConstNew)[1])})

    self.multiBestPoint = optPointsDic
    self.multiBestFitVals = fitSet
    self.multiBestMinObjVals = optMinObjVals
    self.multiBestConstraintVals = optConstNew
    self.multiBestRank = optRank
    self.multiBestCD = optCD
    self.multiBestOutputs = self._collectOutputsForPopulation(optPointsDic,
                                                              len(optRank),
                                                              dataset=rlz)
    return optPointsDic

  def _resolveNewGeneration(self, traj, rlz, info, pastPop, minObjVals, fitVals, constraintVals, ranks=None, CD=None):
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
      self._closeTrajectory(traj, 'converge', 'converged', self.multiBestMinObjVals)

    if self._writeSteps == 'every':
      popSize = rlz.sizes.get('RAVEN_sample_ID', 0)
      self.raiseADebug(f"### rlz.sizes['RAVEN_sample_ID'] = {popSize}")
      solutionExportVars = set()
      solutionExportOutputs = []
      if hasattr(self, '_solutionExport') and self._solutionExport is not None:
        inputs = self._solutionExport.getVars('input') or []
        solutionExportOutputs = self._solutionExport.getVars('output') or []
        outputs = solutionExportOutputs
        solutionExportVars.update(inputs)
        solutionExportVars.update(outputs)
      solutionExportVars.update(self.dependentSample.keys())
      for i in range(popSize):
        survivorSlice = self.pop.isel(chromosome=i)
        rlzDict = survivorSlice.to_series().to_dict()
        for j in range(len(self._objectiveVar)):
          rlzDict[self._objectiveVar[j]] = self.popMinObjVals[j][i]
        rlzDict['batchId'] = self.batchId
        rlzDict['rank'] = np.atleast_1d(ranks.data)[i] if ranks is not None else np.atleast_1d(self.popRanks.data)[i]
        rlzDict['CD'] = np.atleast_1d(CD.data)[i] if CD is not None else np.atleast_1d(self.popCrowdingDist.data)[i]
        if self.popAges is not None:
          rlzDict['age'] = self.popAges[i]
        fitValsContainer = fitVals if isinstance(fitVals, dict) else self.popFitVals
        for fitName in fitValsContainer.keys():
          rlzDict[f'FitnessEvaluation_{fitName}'] = fitValsContainer[fitName].data[i]
        for ind, consName in enumerate([y.name for y in (self._constraintFunctions + self._impConstraintFunctions)]):
          rlzDict[f'ConstraintEvaluation_{consName}'] = constraintVals.data[i, ind]
        cachedOutputs = self._retrieveCachedOutputs(survivorSlice, dataset=rlz)
        for var in solutionExportVars:
          if var in rlzDict:
            continue
          if isinstance(var, str) and var.startswith('FitnessEvaluation_'):
            baseVar = var[len('FitnessEvaluation_'):]
            value = None
            if isinstance(fitValsContainer, dict) and baseVar in fitValsContainer:
              value = fitValsContainer[baseVar].data[i]
            elif hasattr(self.popFitVals, 'keys') and baseVar in self.popFitVals:
              value = self.popFitVals[baseVar].data[i]
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
        bestRlz[self._objectiveVar[i]] = [item[i] for item in self.multiBestMinObjVals]
      bestRlz['rank'] = self.multiBestRank
      bestRlz['CD'] = self.multiBestCD
      if self.multiBestConstraintVals is not None and len(self.multiBestConstraintVals) != 0:
        for ind, consName in enumerate(self.multiBestConstraintVals.Constraint):
          name = consName.item() if hasattr(consName, 'item') else str(consName)
          bestRlz[f'ConstraintEvaluation_{name}'] = self.multiBestConstraintVals[ind].values
      for fitName in self.multiBestFitVals.keys():
        bestRlz[f'FitnessEvaluation_{fitName}'] = self.multiBestFitVals[fitName].data
      if isinstance(self.multiBestOutputs, dict):
        for name in requestedOutputs:
          if isinstance(name, str) and name.startswith('FitnessEvaluation_') and name not in bestRlz:
            baseVar = name[len('FitnessEvaluation_'):]
            if baseVar in self.multiBestFitVals:
              bestRlz[name] = self.multiBestFitVals[baseVar].data
            elif baseVar in self.multiBestOutputs:
              bestRlz[name] = np.asarray(self.multiBestOutputs[baseVar])
      elif hasattr(self.multiBestFitVals, 'keys'):
        for name in requestedOutputs:
          if isinstance(name, str) and name.startswith('FitnessEvaluation_') and name not in bestRlz:
            baseVar = name[len('FitnessEvaluation_'):]
            if baseVar in self.multiBestFitVals:
              bestRlz[name] = self.multiBestFitVals[baseVar].data
      bestRlz.update(self.multiBestPoint)
      self._optPointHistory[traj].append((bestRlz, info))

  def _validateFinalFront(self, candidate):
    """
    _validateFinalFront method.
    @ In, candidate, dict, realization for the candidate final-front point.
    @ Out, None.
    """
    try:
      if not hasattr(self, 'popFitVals') or self.popFitVals is None:
        return
      if not hasattr(self, 'popMinObjVals') or self.popMinObjVals is None:
        return

      def _as_array(values):
        """
        _as_array method.
        @ In, values, array-like, values from a DataArray or list to cast to ndarray.
        @ Out, array, np.ndarray, values converted to a float array.
        """
        return np.asarray(values.data if hasattr(values, 'data') else values, dtype=float)

      popMinObjVals = np.column_stack([np.atleast_1d(_as_array(vals)) for vals in self.popMinObjVals])
      # Finalization candidates are also in minimization space, so do not reapply _objMult.
      try:
        candidateMinObjVals = np.array([float(np.atleast_1d(candidate[obj])[0]) for obj in self._objectiveVar], dtype=float)
      except KeyError:
        self.raiseADebug('Final-front validation skipped: candidate lacks objective entries.')
        return

      combined = np.vstack([popMinObjVals, candidateMinObjVals])
      ranks = np.array(frontUtils.rankNonDominatedFrontiers(combined))
      candidateRank = ranks[-1]
      bestRank = ranks[:-1].min() if combined.shape[0] > 1 else candidateRank
      if candidateRank > bestRank:
        self.raiseAWarning('Final export candidate is dominated by existing population members '
                           f'(rank {candidateRank} vs best rank {bestRank}).')
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

    offspring = datasetToDataArray(rlz, list(self.toBeSampled))
    # minObjVals are objective values in RAVEN minimization space; maximization objectives
    # have already been multiplied by -1 by RavenSampled before GA/NSGA-II ranking.
    offspringMinObjVals = [list(np.atleast_1d(rlz[obj].data)) for obj in self._objectiveVar]

    offspringConstraintVals = constraintHandling(self, info, rlz, offspring,
                                      offspringMinObjVals, multiObjective=True)

    normRlz = deepcopy(rlz)
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
            normRlz[var][idx] = (rlz[var][idx] - self.normScores[var][0]) / self.normScores[var][1]
            if np.isnan(normRlz[var][idx]):
              normRlz[var][idx] = 0.0

      for i in range(len(offspringConstraintVals)):
        for j in range(len(constrVarsList)):
          denom = self.normScores[constrVarsList[j].parameterNames()[0]][1]
          offspringConstraintVals[i][j] = offspringConstraintVals[i][j] / denom
          if np.isnan(offspringConstraintVals[i][j]):
            offspringConstraintVals[i][j] = 0.0

    offspringFitVals = self._fitnessInstance(normRlz,
                                              objVar=self._objectiveVar,
                                              a=self._objCoeff,
                                              b=self._penaltyCoeff,
                                              penalty=None,
                                              constraintFunction=offspringConstraintVals,
                                              constraintNum=self._numOfConst,
                                              type=self._minMax)

    self._process_generation(info,
                             rlz,
                             offspring,
                             offspringMinObjVals,
                             offspringFitVals,
                             offspringConstraintVals)

  def _process_generation(self, info, rlz, offspring, offspringMinObjVals,
                          offspringFitVals, offspringConstraintVals):
    """
      Perform the strategy-specific portion of the multi-objective update.
      Subclasses (e.g. NSGA-II) must implement this method.
    """
    raise NotImplementedError(f'{self.__class__.__name__} must implement "_process_generation".')

  def _checkConvHypervolume(self, traj, **kwargs):
    """
    _checkConvHypervolume method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if hypervolume criterion is satisfied.
    """
    if len(self._optPointHistory[traj]) < 2:
      return False
    if not hasattr(self, 'popRanks') or not hasattr(self, 'popMinObjVals'):
      return False

    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) == 0:
      return False

    currentFront = []
    for idx in rank1Indices:
      point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
      currentFront.append(point)

    prev_opt, _ = self._optPointHistory[traj][-2]
    if 'rank' not in prev_opt:
      return False
    prev_rank1Indices = np.where(np.array(prev_opt['rank']) == 1)[0]
    if len(prev_rank1Indices) == 0:
      return False

    prev_front = []
    for idx in prev_rank1Indices:
      point = [prev_opt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
      prev_front.append(point)

    all_points = currentFront + prev_front
    if not all_points:
      return False
    nadir = [max(p[i] for p in all_points) for i in range(len(self._objectiveVar))]
    reference = [n * 1.1 for n in nadir]

    current_hv = self._computeHypervolume(currentFront, reference)
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
    """
    _computeHypervolume method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ In, reference, list(float), hypervolume reference point.
    @ Out, hv, float, hypervolume of the front.
    """
    if not front:
      return 0.0
    n_objectives = len(front[0])
    if n_objectives == 2:
      return self._hypervolume2D(front, reference)
    if n_objectives == 3:
      return self._hypervolume3D(front, reference)
    return self._hypervolumeWFG(front, reference)

  def _hypervolume2D(self, front, reference):
    """
    _hypervolume2D method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ In, reference, list(float), hypervolume reference point.
    @ Out, hv, float, 2D hypervolume of the front.
    """
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
    """
    _hypervolume3D method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ In, reference, list(float), hypervolume reference point.
    @ Out, hv, float, 3D hypervolume of the front.
    """
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
    """
    _hypervolumeWFG method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ In, reference, list(float), hypervolume reference point.
    @ Out, hv, float, hypervolume computed by recursive WFG method.
    """
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
    """
    _checkConvSpread method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if spread criterion is satisfied.
    """
    if not hasattr(self, 'popRanks') or not hasattr(self, 'popMinObjVals'):
      return False
    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) < 3:
      return False
    front = []
    for idx in rank1Indices:
      point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
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
    """
    _computeSpread method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ Out, spread, float, spread metric for the front.
    """
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
    """
    _checkConvSpacing method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if spacing criterion is satisfied.
    """
    if not hasattr(self, 'popRanks') or not hasattr(self, 'popMinObjVals'):
      return False
    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) < 3:
      return False
    front = []
    for idx in rank1Indices:
      point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
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
    """
    _computeSpacing method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ Out, spacing, float, spacing metric for the front.
    """
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
    """
    _checkConvMaxSpread method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if max-spread criterion is satisfied.
    """
    if len(self._optPointHistory[traj]) < 2:
      return False
    if not hasattr(self, 'popRanks') or not hasattr(self, 'popMinObjVals'):
      return False
    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) == 0:
      return False
    currentFront = []
    for idx in rank1Indices:
      point = [self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
      currentFront.append(point)

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

    current_ms = self._computeMaxSpread(currentFront)
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
    """
    _computeMaxSpread method.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ Out, max_spread, float, max-spread metric for the front.
    """
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
    """
    _checkConvRank1Ratio method.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if rank-1 ratio criterion is satisfied.
    """
    if not hasattr(self, 'popRanks'):
      return False
    if not hasattr(self, '_populationSize') or self._populationSize == 0:
      return False
    rank1_count = np.sum(self.popRanks.data == 1)
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
