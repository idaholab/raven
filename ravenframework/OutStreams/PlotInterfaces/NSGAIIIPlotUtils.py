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
"""Shared math helpers for NSGA-III visual diagnostics."""

from math import comb

import numpy as np


def generateReferenceDirections(numObjectives, populationSize):
  """Replicates the simplex-lattice reference direction generator used in NSGA-III."""
  def latticePoints(m, divisions):
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
    temp = comb(H1 + numObjectives - 1, numObjectives - 1)
    if temp > populationSize or H1 > 20:
      break
    H1 += 1
  H1 = max(H1 - 1, 1)
  directions.extend(latticePoints(numObjectives, H1))

  if len(directions) < populationSize:
    H2 = 0
    while True:
      temp = comb(H2 + numObjectives - 1, numObjectives - 1)
      if len(directions) + temp > populationSize or H2 > 10:
        break
      H2 += 1
    H2 = max(H2 - 1, 0)
    if H2 > 0:
      second = latticePoints(numObjectives, H2)
      offset = 1.0 / (2.0 * H2)
      directions.extend([(np.array(p) + offset) / (1.0 + offset * numObjectives) for p in second])

  directions = np.asarray(directions, dtype=float)
  if directions.size == 0:
    directions = np.eye(numObjectives)
  norms = np.linalg.norm(directions, axis=1, keepdims=True)
  norms[norms == 0.0] = 1.0
  unitDirs = directions / norms
  simplexDirs = directions.copy()
  sums = simplexDirs.sum(axis=1, keepdims=True)
  sums[sums == 0.0] = 1.0
  simplexDirs = simplexDirs / sums
  return unitDirs, simplexDirs


def normalizeObjectives(values):
  """Apply NSGA-III style objective normalisation."""
  if values.size == 0:
    return values
  ideal = np.min(values, axis=0)
  translated = values - ideal
  extreme = _findExtremePoints(translated)
  intercepts = _computeIntercepts(extreme, translated)
  return _normalize(translated, intercepts)


def _findExtremePoints(translated):
  if translated.size == 0:
    return np.zeros((0, 0))
  m = translated.shape[1]
  weights = np.full((m, m), 1e-6)
  np.fill_diagonal(weights, 1.0)
  extremePoints = []
  for weight in weights:
    denom = np.where(weight == 0.0, 1e-12, weight)
    asf = np.max(translated / denom, axis=1)
    idx = int(np.argmin(asf))
    extremePoints.append(translated[idx])
  return np.array(extremePoints)


def _computeIntercepts(extremePoints, translated):
  if translated.size == 0:
    return np.ones(translated.shape[1] if translated.ndim > 1 else 1)
  m = translated.shape[1]
  intercepts = None
  if extremePoints.shape[0] == m and np.linalg.matrix_rank(extremePoints) == m:
    try:
      u = np.ones(m)
      solution = np.linalg.solve(extremePoints, u)
      intercepts = 1.0 / solution
    except Exception:
      intercepts = None
  if intercepts is None or np.any(np.isnan(intercepts)) or np.any(intercepts <= 1e-12):
    intercepts = np.max(translated, axis=0)
  intercepts = np.where(intercepts <= 1e-12, 1.0, intercepts)
  return intercepts


def _normalize(translated, intercepts):
  normalized = translated / intercepts
  normalized = np.where(np.isfinite(normalized), normalized, 0.0)
  return np.clip(normalized, 0.0, None)


def associatePoints(normalizedPoints, referenceDirs):
  """Assign each sample to the nearest reference direction."""
  if normalizedPoints.size == 0:
    return np.array([], dtype=int), np.array([], dtype=float)
  proj = np.dot(normalizedPoints, referenceDirs.T)
  directionNorms = np.linalg.norm(referenceDirs, axis=1)
  directionNorms[directionNorms == 0.0] = 1.0
  proj = proj / directionNorms
  normSq = np.sum(np.square(normalizedPoints), axis=1, keepdims=True)
  distancesSq = normSq - np.square(proj)
  distancesSq = np.clip(distancesSq, 0.0, None)
  assocIndices = np.argmin(distancesSq, axis=1)
  perpendicular = np.sqrt(distancesSq[np.arange(len(distancesSq)), assocIndices])
  return assocIndices, perpendicular
