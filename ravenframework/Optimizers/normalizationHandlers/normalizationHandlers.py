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
  Objective-normalization handlers for the NSGA-III many-objective genetic algorithm.

  A normalization handler maps the combined population's minimization-space objective values onto a
  common adaptive scale before association with reference directions, so that objectives with
  different magnitudes contribute comparably. It is the mix-and-match operator selected via
  <objectiveNormalization type="..."> under <MultiObjectiveGeneticAlgorithm type="NSGA-III">.

  Implemented handlers (with their literature source):
    1. hyperplane -- Deb & Jain (2014) adaptive normalization: translate by the ideal point, find per-
       objective extreme points via an Achievement Scalarizing Function, build the linear hyperplane
       through them, and divide by its intercepts (default).
    2. simplexSum  -- translate by the ideal point and divide each point by the sum of its translated
       objectives, projecting onto the unit simplex. Cheap, robust when the hyperplane is ill-posed.
    3. none        -- translate by the ideal point only (no scaling).

  Handler contract:
    def handler(caller, objectives, **kwargs) -> np.ndarray
      @ In, caller, object, the optimizer instance (for logging / attribute access)
      @ In, objectives, np.ndarray, (nPoints, nObjectives) minimization-space objective values
      @ Out, normalized, np.ndarray, (nPoints, nObjectives) translated/scaled objectives (>= 0)

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------


def _findExtremePoints(translated):
  """
    Identify the per-objective extreme points of a translated point set using an Achievement
    Scalarizing Function (ASF): for objective j the extreme point minimizes max_i(f_i / w_i) with
    a weight vector that is 1 on axis j and a tiny epsilon elsewhere (Deb & Jain 2014).
    @ In, translated, np.ndarray, (nPoints, nObjectives) ideal-translated objective values (>= 0)
    @ Out, extremePoints, np.ndarray, (nObjectives, nObjectives) one extreme point per objective
  """
  numObjectives = translated.shape[1]
  weights = np.full((numObjectives, numObjectives), 1e-6)
  np.fill_diagonal(weights, 1.0)
  extremePoints = []
  for weight in weights:
    denom = np.where(weight == 0.0, 1e-12, weight)
    asf = np.max(translated / denom, axis=1)
    extremePoints.append(translated[int(np.argmin(asf))])
  return np.array(extremePoints)


def _computeIntercepts(extremePoints, translated):
  """
    Compute the intercepts of the linear hyperplane through the extreme points (solve
    extremePoints @ a = 1, intercepts = 1/a). Fall back to the per-objective maxima when the
    hyperplane is degenerate or ill-conditioned.
    @ In, extremePoints, np.ndarray, (nObjectives, nObjectives) extreme points
    @ In, translated, np.ndarray, (nPoints, nObjectives) ideal-translated objective values
    @ Out, intercepts, np.ndarray, (nObjectives,) per-objective intercepts (> 0)
  """
  numObjectives = translated.shape[1]
  intercepts = None
  if extremePoints.shape[0] == numObjectives and np.linalg.matrix_rank(extremePoints) == numObjectives:
    try:
      solution = np.linalg.solve(extremePoints, np.ones(numObjectives))
      intercepts = 1.0 / solution
    except np.linalg.LinAlgError:
      intercepts = None
  if intercepts is None or np.any(np.isnan(intercepts)) or np.any(intercepts <= 1e-12):
    intercepts = np.max(translated, axis=0)
  intercepts = np.where(intercepts <= 1e-12, 1.0, intercepts)
  return intercepts


def hyperplane(caller, objectives, **kwargs):
  """
    Deb & Jain (2014) adaptive hyperplane normalization.
    @ In, caller, object, the optimizer instance (for logging)
    @ In, objectives, np.ndarray, (nPoints, nObjectives) minimization-space objective values
    @ Out, normalized, np.ndarray, (nPoints, nObjectives) normalized objectives (>= 0)
  """
  translated = objectives - np.min(objectives, axis=0)
  intercepts = _computeIntercepts(_findExtremePoints(translated), translated)
  normalized = translated / intercepts
  normalized = np.where(np.isfinite(normalized), normalized, 0.0)
  return np.clip(normalized, 0.0, None)


def simplexSum(caller, objectives, **kwargs):
  """
    Translate by the ideal point and scale each point by the sum of its translated objectives,
    projecting onto the unit simplex (degenerate zero-sum points map to the origin).
    @ In, caller, object, the optimizer instance (for logging)
    @ In, objectives, np.ndarray, (nPoints, nObjectives) minimization-space objective values
    @ Out, normalized, np.ndarray, (nPoints, nObjectives) simplex-projected objectives (>= 0)
  """
  translated = objectives - np.min(objectives, axis=0)
  rowSums = np.sum(translated, axis=1, keepdims=True)
  rowSums = np.where(rowSums <= 1e-12, 1.0, rowSums)
  return np.clip(translated / rowSums, 0.0, None)


def none(caller, objectives, **kwargs):
  """
    Translate by the ideal point only, with no scaling.
    @ In, caller, object, the optimizer instance (for logging)
    @ In, objectives, np.ndarray, (nPoints, nObjectives) minimization-space objective values
    @ Out, normalized, np.ndarray, (nPoints, nObjectives) ideal-translated objectives (>= 0)
  """
  return np.clip(objectives - np.min(objectives, axis=0), 0.0, None)


__normalizationHandlers = {}
__normalizationHandlers['hyperplane'] = hyperplane
__normalizationHandlers['simplexSum'] = simplexSum
__normalizationHandlers['none'] = none


def returnInstance(cls, name):
  """
    Return the normalization handler registered under `name`.
    @ In, cls, class type, the caller (for raiseAnError on an unknown name)
    @ In, name, str, the <objectiveNormalization type="..."> value
    @ Out, __normalizationHandlers[name], function, the normalization handler
  """
  if name not in __normalizationHandlers:
    cls.raiseAnError(IOError, "{} is not a valid option for <objectiveNormalization>. Valid options are: {}. "
                     "Please review the spelling of the normalization handler.".format(
                       name, list(__normalizationHandlers.keys())))
  return __normalizationHandlers[name]
