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


def generate_reference_directions(num_objectives, population_size):
  """Replicates the simplex-lattice reference direction generator used in NSGA-III."""
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
  unit_dirs = directions / norms
  simplex_dirs = directions.copy()
  sums = simplex_dirs.sum(axis=1, keepdims=True)
  sums[sums == 0.0] = 1.0
  simplex_dirs = simplex_dirs / sums
  return unit_dirs, simplex_dirs


def normalize_objectives(values):
  """Apply NSGA-III style objective normalisation."""
  if values.size == 0:
    return values
  ideal = np.min(values, axis=0)
  translated = values - ideal
  extreme = _find_extreme_points(translated)
  intercepts = _compute_intercepts(extreme, translated)
  return _normalize(translated, intercepts)


def _find_extreme_points(translated):
  if translated.size == 0:
    return np.zeros((0, 0))
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


def _compute_intercepts(extreme_points, translated):
  if translated.size == 0:
    return np.ones(translated.shape[1] if translated.ndim > 1 else 1)
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


def _normalize(translated, intercepts):
  normalized = translated / intercepts
  normalized = np.where(np.isfinite(normalized), normalized, 0.0)
  return np.clip(normalized, 0.0, None)


def associate_points(normalized_points, reference_dirs):
  """Assign each sample to the nearest reference direction."""
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
  perpendicular = np.sqrt(distances_sq[np.arange(len(distances_sq)), assoc_indices])
  return assoc_indices, perpendicular
