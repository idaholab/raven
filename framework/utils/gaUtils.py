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
