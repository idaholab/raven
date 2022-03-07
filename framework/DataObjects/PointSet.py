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
  Specialized implementation of DataSet for all-scalar dataobjects
"""
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import

import sys,os
import functools
import copy
try:
  import cPickle as pk
except ImportError:
  import pickle as pk
import xml.etree.ElementTree as ET

import abc
import numpy as np
import pandas as pd
import xarray as xr

from ..BaseClasses import BaseType
from ..utils import utils, cached_ndarray, InputData, xmlUtils, mathUtils
try:
  from .DataSet import DataSet
except ValueError: #attempted relative import in non-package
  from DataSet import DataSet

# for profiling with kernprof
try:
  import __builtin__
  __builtin__.profile
except (AttributeError,ImportError):
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
    Specialized implementation of DataSet for dataobjects with only single-valued inputs and outputs
    for each realization.
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
    self.name = 'PointSet'
    self.type = 'PointSet'
    self.printTag = self.name
    self._neededForReload = [] # PointSet doesn't need anything to reload

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
  def _convertFinalizedDataRealizationToDict(self,rlz, unpackXArray=False):
    """
      After collapsing into xr.Dataset, all entries are stored as xr.DataArrays.
      This converts them into a dictionary like the realization sent in.
      @ In, rlz, dict(varname:xr.DataArray), "row" from self._data
      @ In, unpackXArray, bool, optional, For now it is just a by-pass here
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
    for var, val in rlz.items():
      if var in self.protectedTags:
        continue
      # only modify it if it is not already scalar
      if not mathUtils.isSingleValued(val):
        # treat inputs, outputs differently TODO this should extend to per-variable someday
        ## inputs
        if var in self._inputs:
          method,indic = self._selectInput
        elif var in self._outputs or var in self._metavars:
          # TODO where does metadata get picked from?  Seems like output fits best?
          method, indic = self._selectOutput
        # pivot variables are included here in "else"; remove them after they're used in operators
        else:
          toRemove.append(var)
          continue
        if method in ['inputRow', 'outputRow']:
          # zero-d xarrays give false behavior sometimes
          # TODO formatting should not be necessary once standardized history,float realizations are established
          if type(val) == list:
            val = np.array(val)
          elif type(val).__name__ == 'DataArray':
            val = val.values
          # FIXME this is largely a biproduct of old length-one-vector approaches in the deprecataed data objects
          if val.size == 1:
            rlz[var] = val[0]
          else:
            rlz[var] = val[indic]
        elif method in ['inputPivotValue', 'outputPivotValue']:
          pivotParam = self.getDimensions(var)
          assert(len(pivotParam) == 1) # TODO only handle History for now
          pivotParam = pivotParam[var][0]
          idx = (np.abs(rlz[pivotParam] - indic)).argmin()
          rlz[var] = rlz[var][idx]
          # if history is dataarray -> not currently possible, but keep for when it's needed
          #if type(rlz[var]).__name__ == 'DataArray':
          #  rlz[var] = float(val.sel(**{pivotParam:indic, 'method':b'nearest'})) #casting as str not unicode
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

  def _toCSV(self,fileName,start=0,**kwargs):
    """
      Writes this data object to CSV file (except the general metadata, see _toCSVXML)
      @ In, fileName, str, path/name to write file
      @ In, start, int, optional, first realization to start printing from (if > 0, implies append mode)
      @ In, kwargs, dict, optional, keywords for options
            Possibly includes:
                'clusterLabel': name of variable to cluster printing by.  If included then triggers history-like printing.
      @ Out, None
    """
    startIndex = 0 if 'RAVEN_isEnding' in self.getVars() else start
    # hierarchical flag controls the printing/plotting of the dataobject in case it is an hierarchical one.
    # If True, all the branches are going to be printed/plotted independenttly, otherwise the are going to be reconstructed
    # In this case, if self.hierarchical is False, the histories are going to be reconstructed
    # (see _constructHierPaths for further explainations)
    if not self.hierarchical and 'RAVEN_isEnding' in self.getVars():
      if not np.all(self._data['RAVEN_isEnding'].values):
        keep = self._getRequestedElements(kwargs)
        toDrop = list(var for var in self.getVars() if var not in keep)
        #FIXME: THIS IS EXTREMELY SLOW
        full = self._constructHierPaths()[startIndex:]
        # set up data to write
        mode = 'a' if startIndex > 0 else 'w'

        self.raiseADebug('Printing data to CSV: "{}"'.format(fileName+'.csv'))
        # get the list of elements the user requested to write
        # order data according to user specs # TODO might be time-inefficient, allow user to skip?
        ordered = list(i for i in self._inputs if i in keep)
        ordered += list(o for o in self._outputs if o in keep)
        ordered += list(m for m in self._metavars if m in keep)
        for data in full:
          data = data.drop(toDrop)
          data = data.where(data[self.sampleTag]==data[self.sampleTag].values[-1],drop=True)
          self._usePandasWriteCSV(fileName,data,ordered,keepSampleTag = self.sampleTag in keep,mode=mode)
          mode = 'a'
      else:
        DataSet._toCSV(self, fileName, startIndex, **kwargs)
    else:
      DataSet._toCSV(self, fileName, startIndex, **kwargs)
