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
"""pysensors-compatible basis classes for SparseSensing."""

import numpy as np


class HOSVDBasis:
  """Higher-order SVD basis for 3-D tensors shaped as (sample, time, space)."""

  def __init__(self, n_basis_modes):
    self.n_basis_modes = int(n_basis_modes)
    self.basis_matrix_ = None

  def fit(self, X):
    """Fit the spatial factor from a 3-D tensor, or ignore pysensors' 2-D refit."""
    X = np.asarray(X)
    if X.ndim == 2 and self.basis_matrix_ is not None:
      return self
    if X.ndim != 3:
      raise ValueError(f"HOSVDBasis requires a 3-D input tensor on first fit; got ndim={X.ndim}")
    nSamples, nTime, nSpace = X.shape
    spatial = X.transpose(2, 0, 1).reshape(nSpace, nSamples * nTime)
    u, _s, _vt = np.linalg.svd(spatial, full_matrices=False)
    self.basis_matrix_ = u[:, :self.n_basis_modes]
    return self

  def matrix_representation(self, n_basis_modes=None):
    """Return the basis in the same shape pysensors expects."""
    if n_basis_modes is None:
      return self.basis_matrix_
    return self.basis_matrix_[:, :n_basis_modes]
