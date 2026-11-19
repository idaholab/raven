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
  This file contains the methods converting from and xr datasets/arrays used in the framework.
  Some of the methods were in the PostProcessor.py
  created on 05/25/2021
  @author: Mohammad Abdo (@Jimmy-INL)
"""
import copy
import numpy as np
import xarray as xr
import collections
import sys

def dataArrayToDict(singlePointDataArray):
  """
    Converts the point from realization DataSet to a Dictionary
    @ In, singlePointDataarray, xr.dataarray, the data array containing a single point in the realization
    @ Out, pointDict, dict, a dictionary containing the realization without the objective function
  """
  pointDict = collections.OrderedDict()
  for var in singlePointDataArray.indexes['Gene']:
    pointDict[var] = singlePointDataArray.loc[var].data
  return pointDict

def datasetToDataArray(rlzDataset,vars):
  """
    Converts the realization DataSet to a DataArray
    @ In, rlzDataset, xr.dataset, the data set containing the batched realizations
    @ In, vars, list, the list of decision variables
    @ Out, dataset, xr.dataarray, a data array containing the realization with
                   dims = ['chromosome','Gene']
                   chromosomes are named 0,1,2...
                   Genes are named after variables to be sampled
  """
  dataset = xr.DataArray(np.atleast_2d(rlzDataset[vars].to_array().transpose()),
                            dims=['chromosome','Gene'],
                            coords={'chromosome': np.arange(rlzDataset[vars[0]].data.size),
                                    'Gene':vars})
  return dataset

def _usableBound(value):
  """
    Whether a distribution bound can be used as a real-coded operator limit. RAVEN stores the
    bounds of unbounded distributions as +/- sys.float_info.max, which is finite but makes the
    span (high - low) overflow to inf and the operators produce NaN, so treat it as unbounded.
    @ In, value, float or None, a distribution bound
    @ Out, usable, bool, True if value is a real, finite, non-sentinel bound
  """
  return value is not None and np.isfinite(value) and abs(value) < sys.float_info.max / 2

def finiteGeneBounds(dist, *observedValues):
  r"""
    Return finite (lower, upper) bounds for a gene/decision variable, used by the
    real-coded operators (SBX crossover, polynomial mutation). Prefers the distribution's
    explicit bounds, falls back to extreme quantiles via the ppf, and finally to a
    padded range around the observed parent/child values so the operator is always
    well-defined even for unbounded distributions.
    @ In, dist, Distribution or None, distribution associated with the gene.
    @ In, \*observedValues, float, one or more current gene values (parents/child), passed as separate positional args.
    @ Out, (low, high), tuple(float, float), finite lower and upper bounds with high > low.
  """
  low = getattr(dist, 'lowerBound', None) if dist is not None else None
  high = getattr(dist, 'upperBound', None) if dist is not None else None
  if not _usableBound(low) and dist is not None and hasattr(dist, 'ppf'):
    try:
      low = float(dist.ppf(1e-6))
    except Exception:
      low = None
  if not _usableBound(high) and dist is not None and hasattr(dist, 'ppf'):
    try:
      high = float(dist.ppf(1.0 - 1e-6))
    except Exception:
      high = None
  vMin = min(observedValues)
  vMax = max(observedValues)
  span = abs(vMax - vMin) if vMax != vMin else 1.0
  if not _usableBound(low):
    low = vMin - span
  if not _usableBound(high):
    high = vMax + span
  if high <= low:
    high = low + 1.0
  return float(low), float(high)
