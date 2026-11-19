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
  Association handlers for the NSGA-III many-objective genetic algorithm.

  An association handler assigns each normalized point to its closest reference direction and
  returns the distance used to break ties during niching. It is the mix-and-match operator selected
  via <association type="..."> under <MultiObjectiveGeneticAlgorithm type="NSGA-III">.

  Implemented handlers (with their literature source):
    1. perpendicular -- Deb & Jain (2014): associate with the reference line of minimum perpendicular
       distance (the point's distance to the line through the origin and the reference point).

  Handler contract:
    def handler(caller, normalizedPoints, referenceDirections, **kwargs) -> (np.ndarray, np.ndarray)
      @ In, caller, object, the optimizer instance (for logging / attribute access)
      @ In, normalizedPoints, np.ndarray, (nPoints, nObjectives) normalized objectives
      @ In, referenceDirections, np.ndarray, (nRef, nObjectives) reference points on the unit simplex
      @ Out, assocIndices, np.ndarray, (nPoints,) index of the associated reference direction per point
      @ Out, distances, np.ndarray, (nPoints,) distance to the associated reference line per point

  @authors: Mohammad Abdo (@Jimmy-INL)
"""
# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------


def perpendicular(caller, normalizedPoints, referenceDirections, **kwargs):
  """
    Associate each normalized point with the reference direction minimizing the perpendicular
    distance from the point to the line through the origin and the reference point (Deb & Jain 2014).
    @ In, caller, object, the optimizer instance (for logging)
    @ In, normalizedPoints, np.ndarray, (nPoints, nObjectives) normalized objectives
    @ In, referenceDirections, np.ndarray, (nRef, nObjectives) reference points on the unit simplex
    @ Out, assocIndices, np.ndarray, (nPoints,) associated reference-direction index per point
    @ Out, distances, np.ndarray, (nPoints,) perpendicular distance to the associated line per point
  """
  if normalizedPoints.size == 0:
    return np.array([], dtype=int), np.array([], dtype=float)
  directionNorms = np.linalg.norm(referenceDirections, axis=1)
  directionNorms[directionNorms == 0.0] = 1.0
  # Scalar projection of each point onto each (unit-normalized) reference line.
  projection = np.dot(normalizedPoints, referenceDirections.T) / directionNorms
  pointNormSq = np.sum(np.square(normalizedPoints), axis=1, keepdims=True)
  distancesSq = np.clip(pointNormSq - np.square(projection), 0.0, None)
  assocIndices = np.argmin(distancesSq, axis=1)
  distances = np.sqrt(distancesSq[np.arange(len(distancesSq)), assocIndices])
  return assocIndices, distances


__associationHandlers = {}
__associationHandlers['perpendicular'] = perpendicular


def returnInstance(cls, name):
  """
    Return the association handler registered under `name`.
    @ In, cls, class type, the caller (for raiseAnError on an unknown name)
    @ In, name, str, the <association type="..."> value
    @ Out, __associationHandlers[name], function, the association handler
  """
  if name not in __associationHandlers:
    cls.raiseAnError(IOError, "{} is not a valid option for <association>. Valid options are: {}. "
                     "Please review the spelling of the association handler.".format(
                       name, list(__associationHandlers.keys())))
  return __associationHandlers[name]
