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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for NDsplineRom
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import math
import copy
import numpy as np
from itertools import product
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
interpolationND = utils.findCrowModule("interpolationND")
from .NDinterpolatorRom import NDinterpolatorRom
#Internal Modules End--------------------------------------------------------------------------------

class NDsplineRom(NDinterpolatorRom):
  """
    An N-dimensional Spline model
  """
  ROMtype         = 'NDsplineRom'
  def __init__(self, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    NDinterpolatorRom.__init__(self, **kwargs)
    self.printTag = 'ND-SPLINE ROM'
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.NDSpline())

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples. This is a specialization of the
      Spline Interpolator (since it will create a Cartesian Grid in case
      the samples are not a tensor)
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    import sklearn.neighbors
    numDiscrPerDimension = int(math.ceil(len(targetVals)**(1./len(self.features))))
    newNumberSamples     = numDiscrPerDimension**len(self.features)
    # get discretizations
    discretizations = [ list(set(featureVals[:,d].tolist())) for d in range(len(self.features))]
    # check if it is a tensor grid or not
    tensorGrid = False if np.prod( [len(d) for d in discretizations] ) != len(targetVals) else True
    if not tensorGrid:
      self.raiseAWarning("Training set for NDSpline is not a cartesian grid. The training Tensor Grid is going to be create by interpolation!")
      # isolate training data
      featureVals = copy.deepcopy(featureVals)
      targetVals  = copy.deepcopy(targetVals)
      # new discretization
      newDiscretizations = [np.linspace(min(discretizations[d]), max(discretizations[d]), num=numDiscrPerDimension, dtype=float).tolist() for d in range(len(self.features))]
      # new feature values
      newFeatureVals = np.atleast_2d(np.asarray(list(product(*newDiscretizations))))
      # new valuesContainer
      newTargetVals = np.zeros( (newNumberSamples,len(self.target)) )
      for index in range(len(self.target)):
        # not a tensor grid => interpolate
        nr = sklearn.neighbors.KNeighborsRegressor(n_neighbors= min(2**len(self.features),len(targetVals)), weights='distance')
        nr.fit(featureVals, targetVals[:,index])
        # new target values
        newTargetVals[:,index] = nr.predict(newFeatureVals)
      targetVals  = newTargetVals
      featureVals = newFeatureVals
    # fit the model
    self.featv, self.targv = featureVals,targetVals
    featv = interpolationND.vectd2d(featureVals[:][:])
    for index, target in enumerate(self.target):
      targv = interpolationND.vectd(targetVals[:,index])
      self.interpolator[index].fit(featv,targv)

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    for index in range(len(self.target)):
      self.interpolator[index].reset()


