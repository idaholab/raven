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
  Specialized implementation of DataObject for objects with only single-valued inputs and outputs
  for each realization.
"""
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys,os
import __builtin__
import functools
import copy
import cPickle as pk
import xml.etree.ElementTree as ET

import abc
import numpy as np
import pandas as pd
import xarray as xr

from BaseClasses import BaseType
from Files import StaticXMLOutput
from utils import utils, cached_ndarray, InputData, xmlUtils, mathUtils
try:
  from .XDataSet import DataSet
except ValueError: #attempted relative import in non-package
  from XDataSet import DataSet

# for profiling with kernprof
try:
  __builtin__.profile
except AttributeError:
  # profiler not preset, so pass through
  def profile(func):
    """
      Dummy for when profiler is not in use.
      @ In, func, method, method to run
      @ Out, func, method, method to run
    """
    return func

#
#
#
#
class PointSet(DataSet):
  """
    DataObject developed Oct 2017 to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.  This form is a shortcut for ASSUMED only-float input, output spaces
  """
  # only a few changes from the base class; the external API is identical.

  ### INITIALIZATION ###
  # These are the necessary functions to construct and initialize this data object
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    DataSet.__init__(self)
    self.name      = 'PointSet'
    self.type      = 'PointSet'
    self.printTag  = self.name

  def _readMoreXML(self,xmlNode):
    """
      Initializes data object based on XML input.
      @ In, xmlNode, xml.etree.ElementTree.Element or InputData.ParameterInput, input specification
      @ Out, None
    """
    DataSet._readMoreXML(self,xmlNode)
    # default to taking last point if no other spec was used
    # TODO throw a warning here, once we figure out how to give message handler in all cases
    if self._selectInput is None:
      self._selectInput = ('inputRow',-1)
    if self._selectOutput is None:
      self._selectOutput = ('outputRow',-1)

  ### INTERNAL USE FUNCTIONS ###
  def _collapseNDtoDataArray(self,data,var,labels=None):
    """
      Converts a row of numpy samples into a single DataArray suitable for a xr.Dataset.
      @ In, data, np.ndarray, array of either float or xr.DataArray; array must be single-dimension
      @ In, var, str, name of the variable being acted on
      @ In, labels, list, list of labels to use for collapsed array under self.sampleTag title
      @ Out, DataArray, xr.DataArray, single dataarray object
    """
    # TODO this is slightly different but quite similar to the base class.  Should it be separate?
    assert(isinstance(data,np.ndarray))
    assert(len(data.shape) == 1)
    if labels is None:
      labels = range(len(data))
    else:
      assert(len(labels) == len(data))
    # ALL should be floats or otherwise 1d
    #assert(isinstance(data[0],(float,str,unicode,int,type(None)))) # --> in LimitSurfaceSearch, first can be "None", floats come later
    try:
      assert(isinstance(data[0],(float,str,unicode,int,))) # --> in LimitSurfaceSearch, first can be "None", floats come later
    except AssertionError as e:
      raise e
    array = xr.DataArray(data,
                         dims=[self.sampleTag],
                         coords={self.sampleTag:labels},
                         name=var)
    array.rename(var)
    return array

  def _convertFinalizedDataRealizationToDict(self,rlz):
    """
      After collapsing into xr.Dataset, all entries are stored as xr.DataArrays.
      This converts them into a dictionary like the realization sent in.
      @ In, rlz, dict(varname:xr.DataArray), "row" from self._data
      @ Out, new, dict(varname:value), where "value" could be singular (float,str) or xr.DataArray
    """
    new = {}
    for k,v in rlz.items():
      # only singular, so eliminate dataarray container
      new[k] = v.item(0)
    return new

  def _selectiveRealization(self,rlz):
    """
      Uses "options" parameters from input to select part of the collected data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO costly for loop
    # TODO overwrites rlz by reference; is this okay?
    # data was previously formatted by _formatRealization
    # then select the point we want
    toRemove = []
    for var,val in rlz.items():
      # only modify it if it is not already scalar
      if not isinstance(val,(float,int,str,unicode)):
        # treat inputs, outputs differently TODO this should extend to per-variable someday
        ## inputs
        if var in self._inputs:
          method,indic = self._selectInput
        elif var in self._outputs or var in self._metavars:
          # TODO where does metadata get picked from?  Seems like output fits best?
          method,indic = self._selectOutput
        # pivot variables might be included here; try removing them
        elif var in self.indexes:
          continue # don't need to handle coordinate dimensions, they come with values
        else:
          toRemove.append(var)
          print('DEBUGG unhandled:',var)
          continue
        if method in ['inputRow','outputRow']:
          # zero-d xarrays give false behavior sometimes
          # TODO formatting should not be necessary once standardized history,float realizations are established
          if type(val) == list:
            val = np.array(val)
          elif type(val).__name__ == 'DataArray':
            val = val.values
          # FIXME this is largely a biproduct of old length-one-vector approaches in the deprecataed data objects
          if val.size == 1:
            rlz[var] = float(val)
          else:
            rlz[var] = float(val[indic])
        elif method in ['inputPivotValue','outputPivotValue']:
          pivotParam = self.getDimensions(var)
          assert(len(pivotParam) == 1) # TODO only handle History for now
          pivotParam = pivotParam[var][0]
          rlz[var] = float(val.sel(**{pivotParam:indic, 'method':b'nearest'})) #casting as str not unicode
          # TODO allowing inexact matches; it's finding the nearest
        elif method == 'operator':
          if indic == 'max':
            rlz[var] = float(val.max())
          elif indic == 'min':
            rlz[var] = float(val.min())
          elif indic in ['mean','expectedValue','average']:
            rlz[var] = float(val.mean())
      # otherwise, leave it alone
    for var in toRemove:
      del rlz[var]
    return rlz
