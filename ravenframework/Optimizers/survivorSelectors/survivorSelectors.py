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
  Implementation of survivorSelectors (Elitism) for new generation
  selection process of Genetic Algorithm. Currently the implemented
  survivorSelectors algorithms are:
  1.  ageBased
  2.  fitnessBased

  Created June 16, 2020
  @authors: Mohammad Abdo, Junyung Kim, Diego Mandelli, Andrea Alfonsi
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
import xarray as xr
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ...utils.gaUtils import datasetToDataArray
# Internal Modules End------------------------------------------------------------------------------

# @profile

def _constraintRows(values, nRows):
  """
    Return constraint values as a 2-D (nRows, nConstraints) array, or None when unconstrained.
    @ In, values, xr.DataArray or np.array or None, constraint values
    @ In, nRows, int, expected number of individuals
    @ Out, rows, np.array or None, constraint values with one row per individual
  """
  if values is None:
    return None
  rows = np.asarray(getattr(values, 'data', values), dtype=float)
  if rows.size == 0:
    return None
  return rows.reshape(nRows, -1)

def _constraintArray(rows, template):
  """
    Wrap selected constraint rows back into a DataArray shaped like the input.
    @ In, rows, np.array or None, selected constraint rows (nRows, nConstraints)
    @ In, template, xr.DataArray or None, original constraint values (for dims/coords)
    @ Out, array, xr.DataArray or None, constraint values aligned with the new population
  """
  if rows is None:
    return None
  if isinstance(template, xr.DataArray) and template.ndim == 2:
    dims = template.dims
    coords = {dims[0]: np.arange(rows.shape[0]), dims[1]: template.coords[dims[1]].values} if dims[1] in template.coords \
             else {dims[0]: np.arange(rows.shape[0])}
    return xr.DataArray(rows, dims=dims, coords=coords)
  return xr.DataArray(rows, dims=['chromosome', 'Constraint'], coords={'chromosome': np.arange(rows.shape[0])})

def ageBased(newRlz,**kwargs):
  """
    ageBased survivorSelection mechanism for new generation selection.
    It replaces the oldest parents with the new children regardless of the fitness.
    @ In, newRlz, xr.Dataset, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this survivor selection method:
          age, list, age list for each chromosome of the previous population
          variables, list of variable names to be sampled
          popFitVals, xr.DataArrays, fitness values of the previous generation
          offspringFitVals, xr.DataArray, fitness of each new child, i.e., np.shape(offspringFitVals) = nChildren x nGenes
          population, xr.DataArray, population from previous generation
          popMinObjVals, list, minimization-space objective values of the previous population
          popConstraintVals, xr.DataArray, optional, constraint values of the previous population
          offspringConstraintVals, xr.DataArray, optional, constraint values of the offspring
    @ Out, newPop, xr.DataArray, newPop for the new generation, i.e. np.shape(newPop) = populationSize x nGenes.
    @ Out, newFitVals, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
    @ Out, newMinObjVals, list, minimization-space objective values aligned with newPop
    @ Out, newConstraintVals, xr.DataArray or None, constraint values aligned with newPop (None if unconstrained)
  """
  popSize = np.shape(kwargs['population'])[0]
  if ('age' not in kwargs.keys() or kwargs['age'] is None):
    popAge = [0] * popSize
  else:
    popAge = kwargs['age']
  popFitValsInput = kwargs['popFitVals']
  offspringFitValsInput = kwargs['offspringFitVals']
  popMinObjVals = kwargs['popMinObjVals']
  offspringFitVals = datasetToDataArray(offspringFitValsInput, list(offspringFitValsInput.keys())).data
  offspring = xr.DataArray(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()),
                            dims=['chromosome','Gene'],
                            coords={'chromosome':np.arange(np.shape(np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose()))[0]),
                                    'Gene': kwargs['variables']})
  population = np.atleast_2d(kwargs['population'].data)
  popFitVals = datasetToDataArray(popFitValsInput, list(popFitValsInput.keys())).data.reshape(-1)
  nOffspring = np.shape(offspring)[0]
  parentObj = np.asarray(popMinObjVals, dtype=float).reshape(-1)
  offspringObj = np.asarray(newRlz[kwargs['objVar']].data, dtype=float).reshape(-1)
  parentCons = _constraintRows(kwargs.get('popConstraintVals'), popSize)
  offspringCons = _constraintRows(kwargs.get('offspringConstraintVals'), nOffspring)
  # Order parents youngest-first (equal ages: fitter first); the oldest are at the end and are
  # replaced by the offspring. Every per-individual array follows the same order.
  order = sorted(range(popSize), key=lambda idx: (popAge[idx], -popFitVals[idx]))
  newPop = population[order].copy()
  newFitVals = popFitVals[order].copy()
  newAge = [popAge[idx] + 1 for idx in order]
  newMinObjVals = parentObj[order].copy()
  newCons = parentCons[order].copy() if parentCons is not None else None
  # The last nOffspring slots get the offspring, listed in reverse order (offspring 0 goes last).
  replaced = slice(-1, -nOffspring-1, -1)
  newPop[replaced] = offspring
  newFitVals[replaced] = np.squeeze(offspringFitVals)
  newAge[-1:-nOffspring-1:-1] = [0]*nOffspring
  newMinObjVals[replaced] = offspringObj
  if newCons is not None and offspringCons is not None:
    newCons[replaced] = offspringCons
  # converting back to DataArrays
  newPop = xr.DataArray(newPop,
                               dims=['chromosome','Gene'],
                               coords={'chromosome':np.arange(np.shape(newPop)[0]),
                                       'Gene': kwargs['variables']})
  newFitValsDS = xr.Dataset()
  newFitValsDS[kwargs['objVar']] = xr.DataArray(newFitVals,
                               dims=['chromosome'],
                               coords={'chromosome':np.arange(np.shape(newFitVals)[0])})
  return newPop,newFitValsDS,newAge,list(newMinObjVals),_constraintArray(newCons, kwargs.get('offspringConstraintVals'))


# @profile
def fitnessBased(newRlz,**kwargs):
  """
    fitnessBased survivorSelection mechanism for new generation selection
    It combines the parents and children/offspring then keeps the fittest individuals
    to revert to the same population size.
    @ In, newRlz, xr.Dataset, containing either a single realization, or a batch of realizations.
    @ In, kwargs, dict, dictionary of parameters for this survivor selection method:
          age, list, ages of each chromosome in the population of the previous generation
          offspringFitVals, xr.DataArray, fitness of each new child, i.e., np.shape(offspringFitVals) = nChildren x nGenes
          variables, list, decision variable names
          population, xr.DataArray, population from previous generation
          popFitVals, xr.Dataset, fitness values of the previous population
          popMinObjVals, list, minimization-space objective values of the previous population
          popConstraintVals, xr.DataArray, optional, constraint values of the previous population
          offspringConstraintVals, xr.DataArray, optional, constraint values of the offspring
    @ Out, newPop, xr.DataArray, newPop for the new generation, i.e. np.shape(newPop) = populationSize x nGenes.
    @ Out, newFitVals, xr.DataArray, fitness of the new population
    @ Out, newAge, list, Ages of each chromosome in the new population.
    @ Out, newMinObjVals, list, minimization-space objective values aligned with newPop
    @ Out, newConstraintVals, xr.DataArray or None, constraint values aligned with newPop (None if unconstrained)
  """
  def _toNumericArray(values, defaultSize):
    """
      Convert an incoming objective list into a 1-D numpy array of length defaultSize.
      Missing entries are padded with NaN and extra entries are truncated so the
      returned array always matches the requested length.
      @ In, values, list or np.array or None, objective values to normalize.
      @ In, defaultSize, int, target length of the returned 1-D array.
      @ Out, array, np.array, 1-D array of length defaultSize.
    """
    if values is None:
      return np.full(defaultSize, np.nan)
    array = np.asarray(values)
    if array.size == 0:
      return np.full(defaultSize, np.nan)
    array = array.reshape(-1)
    if array.size < defaultSize:
      pad = np.full(defaultSize - array.size, np.nan)
      array = np.concatenate([array, pad])
    return array[:defaultSize]

  popSize = np.shape(kwargs['population'])[0]
  popAge = list(kwargs['age']) if kwargs.get('age') is not None else [0] * popSize
  if len(popAge) < popSize:
    popAge.extend([0] * (popSize - len(popAge)))

  # Parent data
  popFitValsInput = kwargs['popFitVals']
  offspringFitValsInput = kwargs['offspringFitVals']
  popMinObjValsInput = kwargs['popMinObjVals']
  parentPopulation = np.atleast_2d(kwargs['population'].data)
  parentFitVals = datasetToDataArray(popFitValsInput, list(popFitValsInput.keys())).data.reshape(-1)
  parentMinObjVals = _toNumericArray(popMinObjValsInput, popSize)

  # Offspring data
  offspringFitVals = datasetToDataArray(offspringFitValsInput, list(offspringFitValsInput.keys())).data.reshape(-1)
  offspringPopulation = np.atleast_2d(newRlz[kwargs['variables']].to_array().transpose().data)
  objVar = kwargs['objVar']
  offspringMinObjVals = np.asarray(newRlz[objVar].data).reshape(-1)
  # Align the objective-value vector to the offspring fitness vector length:
  # pad with NaN when shorter, truncate when longer, so both pools concatenate cleanly below.
  if offspringMinObjVals.size < offspringFitVals.size:
    pad = np.full(offspringFitVals.size - offspringMinObjVals.size, np.nan)
    offspringMinObjVals = np.concatenate([offspringMinObjVals, pad])
  elif offspringMinObjVals.size > offspringFitVals.size:
    offspringMinObjVals = offspringMinObjVals[:offspringFitVals.size]

  # Merge parent and offspring pools
  combinedPopulation = np.concatenate([parentPopulation, offspringPopulation])
  combinedFitVals = np.concatenate([parentFitVals, offspringFitVals])
  combinedAge = [age + 1 for age in popAge] + [0] * len(offspringFitVals)
  combinedMinObjVals = np.concatenate([parentMinObjVals, offspringMinObjVals])

  # Select the top popSize individuals by fitness (desc) with age tie-break
  indices = list(range(len(combinedFitVals)))
  indices.sort(key=lambda idx: (combinedFitVals[idx], -combinedAge[idx]), reverse=True)
  selected = indices[:popSize]

  newPopSorted = combinedPopulation[selected]
  newFitVals = combinedFitVals[selected]
  newAge = [combinedAge[idx] for idx in selected]
  newMinObjVals = [combinedMinObjVals[idx] for idx in selected]
  parentCons = _constraintRows(kwargs.get('popConstraintVals'), popSize)
  offspringCons = _constraintRows(kwargs.get('offspringConstraintVals'), len(offspringFitVals))
  newCons = np.concatenate([parentCons, offspringCons])[selected] if parentCons is not None and offspringCons is not None else None

  newPopArray = xr.DataArray(newPopSorted,
                                    dims=['chromosome','Gene'],
                                    coords={'chromosome': np.arange(np.shape(newPopSorted)[0]),
                                            'Gene': kwargs['variables']})
  newFitValsDS = xr.Dataset()
  newFitValsDS[objVar] = xr.DataArray(newFitVals,
                                      dims=['chromosome'],
                                      coords={'chromosome': np.arange(np.shape(newFitVals)[0])})
  return newPopArray, newFitValsDS, newAge, newMinObjVals, _constraintArray(newCons, kwargs.get('offspringConstraintVals'))

# @profile
def rankNcrowdingBased(**kwargs):
  """
    NSGA-II compliant survivor selection with proper elitism.
    Now receives PRE-COMPUTED ranks and crowding distances for the combined population.
    Selects the best N individuals based on these values.

    Compatible with frontUtils.rankNonDominatedFrontiers and frontUtils.crowdingDistance

    @ In, kwargs, dict, must contain:
          - combinedPop: R(t) = P(t) ∪ Q(t) as np.array (nPoints, nGenes)
          - combinedRanks: ranks for all individuals in R(t) as list
          - combinedCD: crowding distances for all individuals in R(t) as np.array
          - combinedMinObjVals: minimization-space objective values for R(t) as list of lists
          - combinedFitVals: fitness values for R(t) as list of lists (nPoints, nObjectives)
          - combinedConstraintVals: constraint values for R(t) as np.array
          - age: ages for R(t) as list
          - popSize: target population size (N)
          - variables: variable names
          - objectiveNames: names of objectives
    @ Out, tuple: (newPop, newRank, newAge, newCrowdingDist, newMinObjVals, newFitVals, newConstraintVals)
  """

  # Extract parameters
  popSize = kwargs['popSize']
  combinedPop = kwargs['combinedPop']
  combinedRanks = kwargs['combinedRanks']  # list from frontUtils
  combinedCD = kwargs['combinedCD']  # np.array from frontUtils
  combinedMinObjVals = kwargs['combinedMinObjVals']
  combinedFitVals = kwargs['combinedFitVals']
  combinedConstraintVals = kwargs['combinedConstraintVals']
  combinedAge = kwargs['age']
  variables = kwargs['variables']
  objectiveNames = kwargs.get('objectiveNames', [f'obj{i}' for i in range(len(combinedMinObjVals))])

  # ============================================================
  # NSGA-II Elitist Selection
  # ============================================================

  # Group individuals by rank (front)
  fronts = {}
  for idx, rank in enumerate(combinedRanks):
    if rank not in fronts:
      fronts[rank] = []
    fronts[rank].append(idx)

  # Get sorted front numbers
  sortedFrontNums = sorted(fronts.keys())

  # Select individuals front by front
  selectedIndices = []

  for frontNum in sortedFrontNums:
    currentFront = fronts[frontNum]

    # If the whole front fits within the remaining slots, take it entirely;
    # otherwise fill the leftover slots from this front by crowding distance.
    if len(selectedIndices) + len(currentFront) <= popSize:
      # Entire front fits - add all individuals
      selectedIndices.extend(currentFront)
    else:
      # Front doesn't fit entirely - select by crowding distance
      remaining = popSize - len(selectedIndices)

      # Get crowding distances for this front
      frontWithCD = [(idx, combinedCD[idx]) for idx in currentFront]

      # Sort by crowding distance (descending - higher CD is better)
      frontWithCD.sort(key=lambda x: x[1], reverse=True)

      # Take top 'remaining' individuals
      selectedIndices.extend([idx for idx, _ in frontWithCD[:remaining]])
      break

  # Ensure we have exactly popSize individuals
  selectedIndices = selectedIndices[:popSize]

  # ============================================================
  # Extract Selected Population P(t+1)
  # ============================================================

  # Extract data for selected individuals
  newPopInputs = combinedPop[selectedIndices]
  newRanks = [combinedRanks[i] for i in selectedIndices]
  newCrowdingDist = combinedCD[selectedIndices]
  newAge = [combinedAge[i] for i in selectedIndices]
  newConstraints = combinedConstraintVals[selectedIndices]

  # Extract objectives
  newMinObjVals = []
  for objValues in combinedMinObjVals:
    newMinObjVals.append([objValues[i] for i in selectedIndices])

  # Extract fitness
  newFitVals = [combinedFitVals[i] for i in selectedIndices]

  # ============================================================
  # Convert to xarray/Dataset Format
  # ============================================================

  # Population as DataArray
  newPopArray = xr.DataArray(newPopInputs,
                                    dims=['chromosome','Gene'],
                                    coords={'chromosome': np.arange(len(newPopInputs)),
                                            'Gene': variables})

  # Ranks as DataArray
  newRankArray = xr.DataArray(newRanks,
                              dims=['rank'],
                              coords={'rank': np.arange(len(newRanks))})

  # Crowding Distance as DataArray
  newCrowdingDistArray = xr.DataArray(newCrowdingDist,
                           dims=['CrowdingDistance'],
                           coords={'CrowdingDistance': np.arange(len(newCrowdingDist))})

  # Fitness as Dataset
  nObjectives = len(objectiveNames)
  newFitValsSet = xr.Dataset()
  for i, objName in enumerate(objectiveNames):
    fitnessValues = [fit[i] for fit in newFitVals]
    newFitValsSet[objName] = xr.DataArray(fitnessValues,
                                          dims=['chromosome'],
                                          coords={'chromosome': np.arange(len(fitnessValues))})

  # Constraints as DataArray, with the same 'Constraint' dim (and names, when given) that
  # constraintHandling builds for the first generation, so popConstraintVals keeps one layout
  constraintNames = kwargs.get('constraintNames')
  if constraintNames is None or len(constraintNames) != newConstraints.shape[1]:
    constraintNames = np.arange(newConstraints.shape[1])
  newConstraintValsArray = xr.DataArray(newConstraints,
                               dims=['chromosome','Constraint'],
                               coords={'chromosome': np.arange(len(newConstraints)),
                                       'Constraint': constraintNames})

  return newPopArray, newRankArray, newAge, newCrowdingDistArray, newMinObjVals, newFitValsSet, newConstraintValsArray

__survivorSelectors = {}
__survivorSelectors['ageBased'] = ageBased
__survivorSelectors['fitnessBased'] = fitnessBased
__survivorSelectors['rankNcrowdingBased'] = rankNcrowdingBased

def returnInstance(cls, name):
  """
    Method designed to return class instance
    @ In, cls, class type
    @ In, name, string, name of class
    @ Out, __survivorSelectors[name], the registered survivor selection operator
  """
  if name not in __survivorSelectors:
    cls.raiseAnError (IOError, "{} is not a valid option for survivor selector. Please review the spelling of the survivor selector. ".format(name))
  return __survivorSelectors[name]
