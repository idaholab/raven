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
  Specific ROM implementation for NDinterpolatorRom
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
interpolationND = utils.findCrowModule("interpolationND")
from .SupervisedLearning import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------


class NDinterpolatorRom(SupervisedLearning):
  """
  A Reduced Order Model for interpolating N-dimensional data
  """
  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.interpolator = []    # pointer to the C++ (crow) interpolator (list of targets)
    self.featv        = None  # list of feature variables
    self.targv        = None  # list of target variables
    self.printTag = 'ND Interpolation ROM'

  def __getstate__(self):
    """
      Overwrite state (for pickle-ing)
      we do not pickle the HDF5 (C++) instance
      but only the info to re-load it
      @ In, None
      @ Out, state, dict, namespace dictionary
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    if 'interpolator' in state.keys():
      a = state.pop("interpolator")
      del a
    return state

  def __setstate__(self, state):
    """
      Initialize the ROM with the data contained in state
      @ In, state, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(state)
    self.setInterpolator()
    #only train if the original copy was trained
    if self.amITrained:
      self._train(self.featv,self.targv)

  def setInterpolator(self):
    """
      Set up the interpolator
      @ In, None
      @ Out, None
    """
    pass

  def _train(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.

      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    self.featv, self.targv = featureVals,targetVals
    featv = interpolationND.vectd2d(featureVals[:][:])
    for index, target in enumerate(self.target):
      targv = interpolationND.vectd(targetVals[:,index])
      self.interpolator[index].fit(featv,targv)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'NDinterpRom   : __confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, numpy.array 2-D, features
      @ Out, prediction, numpy.array 1-D, predicted values
    """
    prediction = {} #np.zeros((featureVals.shape[0]))
    for index, target in enumerate(self.target):
      prediction[target] = np.zeros((featureVals.shape[0]))
      for n_sample in range(featureVals.shape[0]):
        featv = interpolationND.vectd(featureVals[n_sample][:])
        prediction[target][n_sample] = self.interpolator[index].interpolateAt(featv)
      self.raiseAMessage('NDinterpRom   : Prediction by ' + self.name + ' for target '+target+'. Predicted value is ' + str(prediction[target][n_sample]))
    return prediction

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    self.raiseAnError(NotImplementedError,'NDinterpRom   : __returnCurrentSettingLocal__ method must be implemented!')
