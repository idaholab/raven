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
Filters which mask values based on some criterion.
"""

import abc
import numpy as np
from sklearn.base import TransformerMixin


class FilterBase(TransformerMixin):
  """ Base class for transformers which filter or mask data """
  def __init__(self, maskFill=np.nan):
    """
      @ In, maskFill, float or None, value used to replace masked values; if maskFill=None,
                                      the masked values will be dropped
    """
    # NOTE using np.nan to fill the masked values is advantageous because it preserves
    ## the full shape and spacing of the masked array. However, this requires any
    ## subsequent models to be able to correctly handle NaN values! Using maskFill=None
    ## will drop the masked values instead of filling them with a different value.
    self._maskFill = maskFill
    self._mask = None
    self._hiddenValues = None

  @abc.abstractmethod
  def criterion(self, X):
    """
      Criterion for being masked. Evaluates to False if the value should be
      masked and evaluates to True otherwise.
      @ In, X, numpy.ndarray, data array
      @ In, tol, float, tolerance for the criterion
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    pass

  def fit(self, X):
    """
      Fits the mask to the array using the defined criterion
      @ In, X, np.ndarray, array of data
      @ Out, self, FilterBase, class instance
    """
    # find indices to mask based on criterion
    self._mask = self.criterion(X)
    # save the masked (hidden) values
    self._hiddenValues = np.ma.MaskedArray(X, mask=~self._mask)
    return self

  def transform(self, X):
    """
      Applies mask to data
      @ In, X, np.ndarray, array of data
      @ Out, xMasked, np.ndarray, array of masked data
    """
    xMasked = np.ma.MaskedArray(X, mask=self._mask, fill_value=self._maskFill)
    if self._maskFill is None:
      xMasked = xMasked.compressed()
    else:
      xMasked = xMasked.filled()
    if xMasked.ndim == 1:
      # X is passed in as a column vector, and masking can flatten the array.
      ## Reshaping the array here ensures xMasked is returned as a column
      ## vector (2-d array) rather than as a flat, 1-d array.
      xMasked = xMasked.reshape(-1, 1)
    return xMasked

  def inverse_transform(self, X):
    """
      Restores the masked values to the data array X
      @ In, X, np.ndarray, array of data
      @ Out, xUnmasked, np.ndarray, array of data with the masked values restored
    """
    xUnmasked = np.ma.MaskedArray(X, mask=self._mask).filled(0) + self._hiddenValues.filled(0)
    return xUnmasked


class ZeroFilter(FilterBase):
  """ Masks any values that are near zero """
  def criterion(self, X):
    """
      Criterion for being masked. Evaluates to False if the value should be
      masked and evaluates to True otherwise.
      @ In, X, numpy.ndarray, data array
      @ In, tol, float, tolerance for the criterion
      @ Out, mask, numpy.ndarray, numpy array of boolean values that masks values of X
    """
    return np.isclose(X, 0)


__all__ = [
  "ZeroFilter"
]
