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
Analytic implementation of the three-objective DTLZ3 benchmark problem.
"""

import math

_NUM_OBJECTIVES = 3


def _is_sequence(value):
  if isinstance(value, (str, bytes)):
    return False
  return hasattr(value, '__len__') and hasattr(value, '__getitem__')


def _extract_decision_vectors(inputs):
  decision_keys = sorted(key for key in inputs if key.lower().startswith('x'))
  if not decision_keys:
    raise ValueError('DTLZ3_model requires decision variables named x1 ... xN.')
  template = inputs[decision_keys[0]]
  vectorized = _is_sequence(template)
  if vectorized:
    num_samples = len(template)
    vectors = []
    for key in decision_keys:
      values = inputs[key]
      if not _is_sequence(values):
        raise ValueError(f"Inconsistent data for '{key}'; expected a sequence.")
      if len(values) != num_samples:
        raise ValueError(f"Mismatched sample lengths for '{key}'.")
    for idx in range(num_samples):
      vectors.append([inputs[key][idx] for key in decision_keys])
  else:
    vectors = [[inputs[key] for key in decision_keys]]
  return vectors


def _dtlz3_objectives(vector, num_objectives):
  num_decisions = len(vector)
  k = num_decisions - num_objectives + 1
  if k <= 0:
    raise ValueError('DTLZ3 requires at least num_objectives - 1 free variables.')
  tail = vector[-k:]
  g = 100.0 * (k + sum((value - 0.5) ** 2 - math.cos(20.0 * math.pi * (value - 0.5))
                       for value in tail))
  objectives = []
  for m in range(1, num_objectives + 1):
    prefix = 1.0 + g
    for i in range(1, num_objectives - m + 1):
      prefix *= math.cos(0.5 * math.pi * vector[i - 1])
    if m > 1:
      prefix *= math.sin(0.5 * math.pi * vector[num_objectives - m])
    objectives.append(prefix)
  return objectives


def evaluate(inputs):
  if not inputs:
    raise ValueError('DTLZ3_model.evaluate received no decision variables.')
  decision_vectors = _extract_decision_vectors(inputs)
  aggregated = [[] for _ in range(_NUM_OBJECTIVES)]
  for decision_vector in decision_vectors:
    objective_vals = _dtlz3_objectives(decision_vector, _NUM_OBJECTIVES)
    for bucket, val in zip(aggregated, objective_vals):
      bucket.append(val)
  if len(decision_vectors) == 1:
    return tuple(bucket[0] for bucket in aggregated)
  return tuple(aggregated)


def run(self, inputs):
  data = inputs or {}
  if not any(key.lower().startswith('x') for key in data):
    data = {key: value for key, value in self.__dict__.items()
            if key.lower().startswith('x')}
  if not data:
    raise ValueError('DTLZ3_model.run received no decision variables.')
  f1, f2, f3 = evaluate(data)
  self.f1 = f1
  self.f2 = f2
  self.f3 = f3
