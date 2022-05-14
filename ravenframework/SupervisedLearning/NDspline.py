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

  @author: mandd, talbpaul, wangc
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for NDspline
"""
#External Modules------------------------------------------------------------------------------------
import math
import copy
import numpy as np
from itertools import product
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
interpolationND = utils.findCrowModule("interpolationND")
from .NDinterpolatorRom import NDinterpolatorRom
#Internal Modules End--------------------------------------------------------------------------------

class NDspline(NDinterpolatorRom):
  """
    An N-dimensional Spline model
  """
  info = {'problemtype':'regression', 'normalize':True}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""\xmlNode{NDspline} is a ROM based on an $N$-dimensional
                            spline interpolation/extrapolation scheme.
                            In spline interpolation, the regressor is a special type of piece-wise
                            polynomial called tensor spline.
                            The interpolation error can be made small even when using low degree polynomials
                            for the spline.
                            Spline interpolation avoids the problem of Runge's phenomenon, in which
                            oscillation can occur between points when interpolating using higher degree
                            polynomials.
                            In order to use this ROM, the \xmlNode{ROM} attribute \xmlAttr{subType} needs to
                            be \xmlString{NDspline}
                            No further XML sub-nodes are required.
                            \nb This ROM type must be trained from a regular Cartesian grid.
                            Thus, it can only be trained from the outcomes of a grid sampling strategy.
                            \zNormalizationPerformed{NDspline}
                        """
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'ND-SPLINE ROM'

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    self.setInterpolator()

  def initializeFromDict(self, inputDict):
    """
      Function which initializes the ROM given a the information contained in inputDict
      @ In, inputDict, dict, dictionary containing the values required to initialize the ROM
      @ Out, None
    """
    super().initializeFromDict(inputDict)
    self.setInterpolator()

  def _train(self,featureVals,targetVals):
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

  def setInterpolator(self):
    """
      Set up the interpolator
      @ In, None
      @ Out, None
    """
    self.interpolator = []
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.NDSpline())

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.setInterpolator()
