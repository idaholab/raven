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

from ..utils import mathUtils, frontUtils, InputData, InputTypes
from ..utils.gaUtils import datasetToDataArray
from .GeneticAlgorithm import GeneticAlgorithm
from .constraintHandling.constraintHandling import constraintHandling


class MultiObjectiveGeneticAlgorithm(GeneticAlgorithm):
  """Shared functionality for multi-objective genetic algorithms."""

  convergenceOptions = dict(GeneticAlgorithm.convergenceOptions, **{
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
                        value for several successive generations.""",
      'hypervolume': r""" sets the relative-change tolerance for the Pareto-front hypervolume indicator
                        between successive generations. The hypervolume is measured in RAVEN's internal
                        minimization space against a common nadir-based reference point (offset by a positive
                        margin so it is valid for zero or negative objectives); convergence is declared once
                        the relative change in hypervolume falls below the provided value."""})

  def __init__(self):
    """
    Constructor. Initializes the multi-objective bookkeeping attributes used across
    generations:
      self._canHandleMultiObjective, bool, flag enabling multi-objective handling.
      self.popRanks, xr.DataArray or None, non-dominated sorting rank per population member.
      self.popCrowdingDist, xr.DataArray or None, crowding distance per population member.
      self.multiBestPoint, dict or None, rank-1 optimal decision points keyed by variable.
      self.multiBestFitVals, xr.Dataset or None, fitness values for the rank-1 survivors.
      self.multiBestMinObjVals, np.ndarray or None, minimization-space objective values of the rank-1 front.
      self.multiBestConstraintVals, xr.DataArray or None, constraint values for the rank-1 front.
      self.multiBestRank, np.ndarray or None, ranks of the retained best (rank-1) points.
      self.multiBestCD, np.ndarray or None, crowding distances of the retained best points.
      self.multiBestOutputs, dict or None, cached non-decision/non-objective outputs for the best points.
      self._populationCache, dict, maps a chromosome key (tuple) to a dict of its evaluated outputs,
        preserving full evaluation data for survivors across generations.
    @ Out, None.
    """
    super().__init__()
    self._canHandleMultiObjective = True
    self._crowdingNormalization = 'front'           # 'front' or 'population' normalization for crowding distance
    self._paretoArchiveEnabled = False              # if True, accumulate non-dominated solutions across generations
    self._paretoArchiveMaxSize = None               # optional cap on archive size (None = unbounded)
    self._paretoArchive = None                      # accumulated archive records (dict), see _mergeIntoParetoArchive
    self._constraintEpsilon = 0.0                    # epsilon-constrained dominance relaxation (0.0 = strict)
    self._adaptiveMutation = False                   # if True, anneal mutation probability over generations
    self._adaptiveMutationFinal = None               # final mutation probability (None = 1/nVariables)
    self._stochasticRanking = False                  # if True, stochastically rank by objectives only (Runarsson & Yao)
    self._stochasticRankingPf = 0.45                 # probability of objective-only ranking per generation
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
    self._paretoArchive = None
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
    Build and return the input specification for the multi-objective genetic algorithm,
    extending the base GeneticAlgorithm specification with multi-objective descriptions.
    @ In, cls, the class for which the input specification is being generated.
    @ Out, specs, InputData.ParameterInput, input specification for MultiObjectiveGeneticAlgorithm.
    """
    specs = super(MultiObjectiveGeneticAlgorithm, cls).getInputSpecification()
    specs.name = 'MultiObjectiveGeneticAlgorithm'
    specs.description = r"""The \xmlNode{MultiObjectiveGeneticAlgorithm} augments \xmlNode{GeneticAlgorithm} with the
                            operators required to evolve Pareto-optimal populations. It enables non-dominated sorting,
                            crowding-distance survivor selection, and multi-objective convergence metrics that are shared
                            by concrete optimizers such as \xmlNode{NSGAII}."""
    objective = specs.getSub('objective')
    if objective is not None:
      objective.description = r"""List the objective variables that jointly define the Pareto front. Two or more
                                  objectives are required when using a multi-objective genetic algorithm."""
    crowdingDistanceNormalization = InputData.parameterInputFactory('crowdingDistanceNormalization', strictMode=True,
        contentType=InputTypes.makeEnumType('crowdingDistanceNormalization', 'cdNormType', ['front', 'population']),
        descr=r"""selects how objective gaps are normalized when computing the NSGA-II crowding distance.
                  \textit{front} (default) normalizes each objective by the range observed within each
                  non-dominated front, the classic Deb et al. (2002) formulation. \textit{population}
                  normalizes by the range of each objective over the whole combined population, so crowding
                  distances are comparable across fronts and generations; this can improve diversity
                  preservation on problems whose objectives differ greatly in scale.""",
        default='front')
    specs.addSub(crowdingDistanceNormalization)
    paretoArchive = InputData.parameterInputFactory('paretoArchive', strictMode=True,
        contentType=InputTypes.BoolType,
        descr=r"""if True, maintain an external archive of the non-dominated (rank-1) solutions found
                  across all generations and report it as the final Pareto front. NSGA-II's elitist
                  (mu+lambda) survivor selection already protects the rank-1 front between consecutive
                  generations, but an archive additionally guarantees that the best front ever found is
                  reported even if a later generation drops a previously discovered point. Defaults to
                  False (only the final generation's rank-1 front is reported).""",
        default=False)
    paretoArchive.addParam('maxSize', InputTypes.IntegerType, required=False,
        descr=r"""optional cap on the number of solutions retained in the archive. When the archive
                  exceeds this size the most crowded solutions are removed first (boundary solutions are
                  always kept). If omitted the archive is unbounded.""")
    specs.addSub(paretoArchive)
    constraintEpsilon = InputData.parameterInputFactory('constraintEpsilon', strictMode=True,
        contentType=InputTypes.FloatType,
        descr=r"""enables epsilon-constrained dominance (Takahama \& Sato) in the non-dominated sorting:
                  solutions whose total constraint violation does not exceed this value are treated as
                  feasible and therefore compete on objectives rather than being strictly dominated by
                  fully feasible solutions. This relaxation can improve exploration along active
                  constraint boundaries. Defaults to 0.0 (strict Deb constrained dominance).""",
        default=0.0)
    specs.addSub(constraintEpsilon)
    adaptiveMutation = InputData.parameterInputFactory('adaptiveMutation', strictMode=True,
        contentType=InputTypes.BoolType,
        descr=r"""if True, anneal the mutation probability linearly across the generation budget, from the
                  configured \xmlNode{mutationProb} (initial, exploratory) down to the \xmlAttr{final}
                  value (late, exploitative). This favors broad exploration early and fine refinement of
                  the Pareto front late. Defaults to False (constant \xmlNode{mutationProb}).""",
        default=False)
    adaptiveMutation.addParam('final', InputTypes.FloatType, required=False,
        descr=r"""the mutation probability reached at the final generation when \xmlNode{adaptiveMutation}
                  is True. If omitted it defaults to 1/(number of decision variables), the customary
                  per-gene NSGA-II mutation rate.""")
    specs.addSub(adaptiveMutation)
    stochasticRanking = InputData.parameterInputFactory('stochasticRanking', strictMode=True,
        contentType=InputTypes.BoolType,
        descr=r"""if True, use stochastic ranking (Runarsson \& Yao) for constraint handling: each
                  generation the non-dominated sort is performed by objectives only (ignoring
                  constraints) with probability \xmlAttr{pf}, and by Deb constrained dominance otherwise.
                  This stochastically balances objective progress against feasibility and can keep the
                  strict feasible-first rule from stalling exploration near active constraints. Defaults
                  to False.""",
        default=False)
    stochasticRanking.addParam('pf', InputTypes.FloatType, required=False,
        descr=r"""probability of ranking by objectives only (ignoring constraints) in a given generation
                  when \xmlNode{stochasticRanking} is enabled. Defaults to 0.45 (the Runarsson \& Yao
                  recommended value); values below 0.5 favor feasibility on average.""")
    specs.addSub(stochasticRanking)
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
      Convert a decision-variable value into a hashable, comparable token. The incoming
      ``value`` may be an xr.DataArray, xr.Dataset, numpy array, or python scalar (bytes,
      str, bool, integer, or float); the first scalar element is extracted and coerced to
      a stable bool/int/str token or a float rounded to 12 decimals so that chromosomes
      can be compared and used as dictionary keys regardless of source container type.
      @ In, value, object, raw value extracted from an evaluation
      @ Out, normalized, object, comparable representation (bool, int, str, or rounded float)
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
      floatVal = float(scalar)
    except (TypeError, ValueError):
      return str(scalar)
    if np.isnan(floatVal):
      return 'nan'
    return round(floatVal, 12)

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
    sampleDim = 'RAVEN_sample_ID'
    if sampleDim not in dataset.dims:
      return
    if self._populationCache is None:
      self._populationCache = {}
    count = dataset.sizes.get(sampleDim, 0)
    for idx in range(count):
      sample = dataset.isel({sampleDim: idx})
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
      sampleDim = 'RAVEN_sample_ID'
      if sampleDim in dataset.dims:
        count = dataset.sizes.get(sampleDim, 0)
        for idx in range(count):
          sample = dataset.isel({sampleDim: idx})
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
      chromoSlice = population.isel(chromosome=index)
      for var in genes:
        try:
          if 'Gene' in chromoSlice.coords and var in chromoSlice.coords['Gene'].values:
            val = chromoSlice.sel(Gene=var).values
          else:
            val = chromoSlice.loc[var].values
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
      @ Out, outputs, dict(str -> list), collected outputs per requested variable. Each
        appended entry (``value`` below) is either a python scalar (when the cached output
        is a size-1 numpy array or already scalar) or a copied numpy.ndarray (for vector
        outputs); missing outputs are recorded as numpy.nan.
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
    crowdingNormNode = paramInput.findFirst('crowdingDistanceNormalization')
    if crowdingNormNode is not None:
      self._crowdingNormalization = crowdingNormNode.value
    paretoArchiveNode = paramInput.findFirst('paretoArchive')
    if paretoArchiveNode is not None:
      self._paretoArchiveEnabled = paretoArchiveNode.value
      self._paretoArchiveMaxSize = paretoArchiveNode.parameterValues.get('maxSize', None)
    constraintEpsilonNode = paramInput.findFirst('constraintEpsilon')
    if constraintEpsilonNode is not None:
      self._constraintEpsilon = constraintEpsilonNode.value
    adaptiveMutationNode = paramInput.findFirst('adaptiveMutation')
    if adaptiveMutationNode is not None:
      self._adaptiveMutation = adaptiveMutationNode.value
      self._adaptiveMutationFinal = adaptiveMutationNode.parameterValues.get('final', None)
    stochasticRankingNode = paramInput.findFirst('stochasticRanking')
    if stochasticRankingNode is not None:
      self._stochasticRanking = stochasticRankingNode.value
      self._stochasticRankingPf = stochasticRankingNode.parameterValues.get('pf', 0.45)

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
    if self._paretoArchiveEnabled:
      self._mergeIntoParetoArchive(optPointsDic, optMinObjVals, fitSet, optConstNew, self.multiBestOutputs)
      self._applyParetoArchiveToMultiBest()
    return optPointsDic

  def _mergeIntoParetoArchive(self, pointsDic, minObjVals, fitSet, constraintVals, outputs):
    """
    Merge the current generation's rank-1 front into the persistent Pareto archive,
    keeping only the mutually non-dominated solutions (minimization space). All parallel
    per-solution data (decision variables, objectives, fitness, constraints, outputs) is
    carried so the archived front can be reported with the same columns as a normal front.
    @ In, pointsDic, dict, rank-1 decision values keyed by variable name.
    @ In, minObjVals, np.ndarray, (nPoints, nObjectives) minimization-space objective values.
    @ In, fitSet, xr.Dataset, fitness values keyed by objective name for the rank-1 points.
    @ In, constraintVals, xr.DataArray or list, (Constraint, Evaluation) constraint values, or [] if none.
    @ In, outputs, dict, cached non-decision/non-objective outputs keyed by variable name.
    @ Out, None.
    """
    varNames = list(pointsDic.keys())
    curDecision = np.column_stack([np.asarray(pointsDic[v], dtype=float) for v in varNames]) if varNames else np.empty((minObjVals.shape[0], 0))
    curObj = np.asarray(minObjVals, dtype=float)
    fitKeys = list(fitSet.keys())
    curFit = np.column_stack([np.asarray(fitSet[k].data, dtype=float) for k in fitKeys]) if fitKeys else np.empty((curObj.shape[0], 0))
    hasConstr = hasattr(constraintVals, 'values') and getattr(constraintVals, 'size', 0) != 0
    constrNames = [str(c) for c in constraintVals.Constraint.values] if hasConstr else []
    curConstr = np.asarray(constraintVals.values, dtype=float).T if hasConstr else np.empty((curObj.shape[0], 0))
    outNames = list(outputs.keys()) if isinstance(outputs, dict) else []
    curOut = {name: list(np.atleast_1d(outputs[name])) for name in outNames}

    if self._paretoArchive is None:
      prevObj = np.empty((0, curObj.shape[1]))
      prevDecision = np.empty((0, curDecision.shape[1]))
      prevFit = np.empty((0, curFit.shape[1]))
      prevConstr = np.empty((0, curConstr.shape[1]))
      prevOut = {name: [] for name in outNames}
    else:
      prevObj = self._paretoArchive['obj']
      prevDecision = self._paretoArchive['decision']
      prevFit = self._paretoArchive['fit']
      prevConstr = self._paretoArchive['constr']
      prevOut = self._paretoArchive['outputs']

    minMask = np.ones(curObj.shape[1], dtype=bool)  # archive objectives are already in minimization space
    _, kept = frontUtils.updateParetoArchive(prevObj, curObj, minMask=minMask,
                                             maxArchiveSize=self._paretoArchiveMaxSize)
    stackedDecision = np.vstack([prevDecision, curDecision])
    stackedObj = np.vstack([prevObj, curObj])
    stackedFit = np.vstack([prevFit, curFit])
    stackedConstr = np.vstack([prevConstr, curConstr])
    stackedOut = {name: list(prevOut.get(name, [])) + curOut.get(name, []) for name in outNames}

    self._paretoArchive = {
        'varNames': varNames,
        'decision': stackedDecision[kept],
        'obj': stackedObj[kept],
        'fitKeys': fitKeys,
        'fit': stackedFit[kept],
        'constrNames': constrNames,
        'constr': stackedConstr[kept],
        'outNames': outNames,
        'outputs': {name: [stackedOut[name][i] for i in kept] for name in outNames},
    }

  def _applyParetoArchiveToMultiBest(self):
    """
    Overwrite the multiBest* solution-export containers with the accumulated Pareto archive
    so the reported front is the best non-dominated set found over the whole run. Ranks are
    all 1 (the archive is mutually non-dominated) and crowding distances are recomputed on the
    archived objectives.
    @ Out, None.
    """
    archive = self._paretoArchive
    nPoints = archive['obj'].shape[0]
    self.multiBestPoint = {var: archive['decision'][:, i] for i, var in enumerate(archive['varNames'])}
    self.multiBestMinObjVals = archive['obj']
    fitSet = None
    for count, key in enumerate(archive['fitKeys']):
      dataArray = xr.DataArray(archive['fit'][:, count], dims=['chromosome'], coords={'chromosome': np.arange(nPoints)})
      if count == 0:
        fitSet = dataArray.to_dataset(name=key)
      else:
        fitSet[key] = dataArray
    self.multiBestFitVals = fitSet
    if archive['constrNames']:
      self.multiBestConstraintVals = xr.DataArray(archive['constr'].T,
                                                  dims=['Constraint', 'Evaluation'],
                                                  coords={'Constraint': archive['constrNames'],
                                                          'Evaluation': np.arange(nPoints)})
    else:
      self.multiBestConstraintVals = []
    self.multiBestRank = np.ones(nPoints, dtype=int)
    self.multiBestCD = frontUtils.crowdingDistance(np.ones(nPoints, dtype=int), nPoints, archive['obj'])
    self.multiBestOutputs = {name: archive['outputs'][name] for name in archive['outNames']}

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

      def asArray(values):
        """
        Cast a DataArray or list of objective values to a 1-D float numpy array.
        @ In, values, array-like, values from a DataArray or list to cast to ndarray.
        @ Out, array, np.ndarray, values converted to a float array.
        """
        return np.asarray(values.data if hasattr(values, 'data') else values, dtype=float)

      popMinObjVals = np.column_stack([np.atleast_1d(asArray(vals)) for vals in self.popMinObjVals])
      # Finalization candidates are stored in minimization space; convert to original
      # signs below only for Pareto ranking with explicit objective directions.
      try:
        candidateMinObjVals = np.array([float(np.atleast_1d(candidate[obj])[0]) for obj in self._objectiveVar], dtype=float)
      except KeyError:
        self.raiseADebug('Final-front validation skipped: candidate lacks objective entries.')
        return

      combined = np.vstack([popMinObjVals, candidateMinObjVals])
      # Rank final-front candidates with user-facing objective signs and
      # explicit objective directions, matching the NSGA-II survivor ranking.
      combinedExternalObjVals = np.array(
          [[self._objMult[obj] * val for obj, val in zip(self._objectiveVar, solution)]
           for solution in combined], dtype=float)
      minMask = np.array([optType == "min" for optType in self._minMax], dtype=bool)
      ranks = np.array(frontUtils.rankNonDominatedFrontiers(combinedExternalObjVals, minMask=minMask))
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
    Check convergence on the relative change of the Pareto-front hypervolume indicator
    between successive generations. The rank-1 front is taken in RAVEN's internal
    minimization space, and the current and previous fronts are measured against a
    common reference point (the union nadir offset by a positive margin) so the two
    hypervolumes are directly comparable. Convergence is declared once the relative
    change falls below the user threshold.
    @ In, traj, int, trajectory identifier for the current optimization run.
    @ In, **kwargs, dict, additional convergence inputs (unused).
    @ Out, converged, bool, True if the hypervolume criterion is satisfied.
    """
    if not hasattr(self, 'popRanks') or not hasattr(self, 'popMinObjVals'):
      return False
    rank1Indices = np.where(self.popRanks.data == 1)[0]
    if len(rank1Indices) == 0:
      return False
    currentFront = [[self.popMinObjVals[j][idx] for j in range(len(self._objectiveVar))]
                    for idx in rank1Indices]
    if not hasattr(self, '_paretoFrontHistory'):
      self._paretoFrontHistory = {}
    previousFront = self._paretoFrontHistory.get(traj)
    self._paretoFrontHistory[traj] = currentFront
    if previousFront is None:
      # No earlier front to compare against yet; record this one and continue.
      return False
    reference = self._hypervolumeReference(currentFront, previousFront)
    currentHV = frontUtils.hypervolume(currentFront, reference)
    previousHV = frontUtils.hypervolume(previousFront, reference)
    if not hasattr(self, '_hvHistory'):
      self._hvHistory = {}
    self._hvHistory.setdefault(traj, []).append(currentHV)
    if mathUtils.compareFloats(previousHV, 0.0, 1e-14):
      relChange = 0.0 if mathUtils.compareFloats(currentHV, 0.0, 1e-14) else float('inf')
    else:
      relChange = abs(currentHV - previousHV) / abs(previousHV)
    threshold = self._convergenceCriteria.get('hypervolume', 0.01)
    converged = relChange < threshold
    self.raiseADebug(self.convFormat.format(
        name='Hypervolume',
        conv=str(converged),
        got=relChange,
        req=threshold))
    return converged

  def _hypervolumeReference(self, currentFront, previousFront, marginFraction=0.1):
    """
    Build a reference point that is strictly worse (larger, in minimization space) than every
    point of both fronts. The reference is the per-objective nadir of the union of the two fronts
    plus a positive margin proportional to each objective's range. Using an additive margin rather
    than a multiplicative factor (the previous implementation used nadir * 1.1) keeps the reference
    valid when objectives are zero or negative, which is common after RAVEN converts maximization
    objectives to minimization space.
    @ In, currentFront, list(list(float)), current rank-1 front in minimization space.
    @ In, previousFront, list(list(float)), previous rank-1 front in minimization space.
    @ In, marginFraction, float, fractional margin added beyond the union nadir.
    @ Out, reference, list(float), reference point for the hypervolume computation.
    """
    union = np.array(currentFront + previousFront, dtype=float)
    nadir = union.max(axis=0)
    ideal = union.min(axis=0)
    ranges = nadir - ideal
    margins = np.where(ranges > 0, marginFraction * ranges, marginFraction * np.maximum(np.abs(nadir), 1.0))
    reference = nadir + np.maximum(margins, 1e-12)
    return reference.tolist()

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
    Compute Deb's spread metric (Delta) for a Pareto front, quantifying how uniformly the
    solutions are distributed. The front points are sorted along the first objective; the
    Euclidean distances between consecutive points are compared against their mean, and the
    distances from the extreme front points to the ideal/nadir corners are added in. Lower
    values indicate a more uniform spread of non-dominated solutions.
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
    distFirst = np.linalg.norm(np.array(front[0]) - ideal)
    distLast = np.linalg.norm(np.array(front[-1]) - nadir)
    meanDist = np.mean(distances)
    if mathUtils.compareFloats(meanDist, 0.0, 1e-14):
      return 0.0
    spread = (distFirst + distLast + sum(abs(d - meanDist) for d in distances)) / (distFirst + distLast + (len(distances)) * meanDist)
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
    Compute the spacing metric for a Pareto front: the standard deviation of each point's
    nearest-neighbour distance to the other front points. A lower value indicates the
    non-dominated solutions are more evenly spaced along the front.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ Out, spacing, float, spacing metric for the front.
    """
    n = len(front)
    if n < 2:
      return 0.0
    distances = []
    for i in range(n):
      pointI = np.array(front[i])
      minDist = float('inf')
      for j in range(n):
        if i == j:
          continue
        pointJ = np.array(front[j])
        dist = np.linalg.norm(pointI - pointJ)
        if dist < minDist:
          minDist = dist
      distances.append(minDist)
    meanDist = np.mean(distances)
    spacing = np.sqrt(np.mean([(d - meanDist) ** 2 for d in distances]))
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

    prevOpt, _ = self._optPointHistory[traj][-2]
    if 'rank' not in prevOpt:
      return False
    prevRank1 = np.where(np.array(prevOpt['rank']) == 1)[0]
    if len(prevRank1) == 0:
      return False
    prevFront = []
    for idx in prevRank1:
      point = [prevOpt[self._objectiveVar[j]][idx] for j in range(len(self._objectiveVar))]
      prevFront.append(point)

    currentMaxSpread = self._computeMaxSpread(currentFront)
    prevMaxSpread = self._computeMaxSpread(prevFront)

    if mathUtils.compareFloats(prevMaxSpread, 0.0, 1e-14):
      relChange = float('inf')
    else:
      relChange = abs(currentMaxSpread - prevMaxSpread) / prevMaxSpread

    threshold = self._convergenceCriteria.get('maxSpread', 0.05)
    converged = relChange < threshold

    self.raiseADebug(self.convFormat.format(
        name='MaxSpread',
        conv=str(converged),
        got=relChange,
        req=threshold))

    return converged

  def _computeMaxSpread(self, front):
    """
    Compute the maximum spread of a Pareto front: the Euclidean norm of the per-objective
    ranges (max minus min over the front for each objective). This captures how widely the
    non-dominated solutions extend in objective space.
    @ In, front, list(list(float)), Pareto front points in objective space.
    @ Out, maxSpread, float, max-spread metric for the front.
    """
    n = len(front)
    if n < 2:
      return 0.0
    ranges = []
    for i in range(len(front[0])):
      objValues = [point[i] for point in front]
      ranges.append(max(objValues) - min(objValues))
    maxSpread = np.sqrt(sum(r ** 2 for r in ranges))
    return maxSpread

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
    rank1Count = np.sum(self.popRanks.data == 1)
    ratio = rank1Count / self._populationSize
    if not hasattr(self, '_rank1History'):
      self._rank1History = {}
    if traj not in self._rank1History:
      self._rank1History[traj] = []
    self._rank1History[traj].append(ratio)
    threshold = self._convergenceCriteria.get('rank1Ratio', 0.5)
    stableGenerations = 3
    if len(self._rank1History[traj]) < stableGenerations:
      converged = False
    else:
      recentRatios = self._rank1History[traj][-stableGenerations:]
      allAboveThreshold = all(r >= threshold for r in recentRatios)
      variation = max(recentRatios) - min(recentRatios)
      converged = allAboveThreshold and variation < 0.1
    self.raiseADebug(self.convFormat.format(
        name='Rank1Ratio',
        conv=str(converged),
        got=ratio,
        req=threshold))
    return converged
