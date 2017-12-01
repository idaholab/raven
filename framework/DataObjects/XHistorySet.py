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

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys,os
import __builtin__
import functools
import copy
import cPickle as pk
import itertools
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
  def profile(func): return func

#
#
#
#
class HistorySet(DataSet):
  """
    DataObject developed Oct 2017 to obtain linear performance from data objects when appending, over
    thousands of variables and millions of samples.  Wraps np.ndarray for collecting and uses xarray.Dataset
    for final form.  This form is a shortcut for ASSUMED only-float inputs and shared-single-pivot outputs
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
    self.name      = 'HistorySet'
    self.type      = 'HistorySet'
    self.printTag  = self.name
    self._tempPivotParam = None

  def _setDefaultPivotParams(self):
    """
      Sets default pivot parameters.
      @ In, None
      @ Out, None
    """
    # If not set, use "time" as default.
    if self._tempPivotParam is None:
      self.raiseAWarning('No pivot parameter provided; defaulting to \"time\".')
      self._tempPivotParam = 'time'
    # propagate provided pivot parameter to all variables.
    # don't use setter, set directly, since there's only one var
    self._pivotParams = {self._tempPivotParam:self._outputs[:]}

  ### INTERNAL USE FUNCTIONS ###
  def _collapseNDtoDataArray(self,data,var,labels=None):
    """
      Converts a row of numpy samples into a single DataArray suitable for a xr.Dataset.
      @ In, data, np.ndarray, array of either float or xr.DataArray; array must be single-dimension
      @ In, var, str, name of the variable being acted on
      @ In, labels, list, list of labels to use for collapsed array under self.sampleTag title
      @ Out, DataArray, xr.DataArray, single dataarray object
    """
    # TODO this is only type-checking before using the base class implementation.
    ## TODO these assertions are identical to the base class right now; should abstract
    assert(isinstance(data,np.ndarray))
    assert(len(data.shape) == 1)
    if labels is None:
      labels = range(len(data))
    else:
      assert(len(labels) == len(data))
    ## these assertions are specific to history sets -> should they be in addRealization instead?
    # Inputs and meta should all be single entries, outputs should all be xr.DataArray that depend only on pivotParam
    if var in self._inputs:
      assert(isinstance(data[0],(float,str,unicode,int)))
    elif var in self._outputs:
      # all outputs are xr.DataArrays
      assert(isinstance(data[0],xr.DataArray))
      # all outputs have a single independent coordinate
      assert(len(data[0].dims) == 1)
      # all outputs depend only on the pivot parameter
      assert(data[0].dims[0] == self._pivotParams.keys()[0])
    return DataSet._collapseNDtoDataArray(self,data,var,labels)

  def _fromCSV(self,fname,**kwargs):
    """
      Loads a dataset from custom RAVEN history csv.
      @ In, fname, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    # data dict for loading data
    data = {}
    # load in XML, if present
    meta = self._fromCSVXML(fname)
    self.samplerTag = meta.get('sampleTag',self.sampleTag)
    # TODO do selective inputs! consistency check here instead
    dims = meta.get('pivotParams',{})
    if len(dims)>0:
      self.setPivotParams(dims)
    self._inputs = meta.get('inputs',self._inputs)
    self._outputs = meta.get('outputs',self._outputs)
    self._metavars = meta.get('metavars',self._metavars)
    self._allvars = self._inputs + self._outputs + self._metavars
    # load in main CSV
    ## read into dataframe
    main = pd.read_csv(fname+'.csv')
    nSamples = len(main.index)
    ## collect input space data
    for inp in self._inputs + self._metavars:
      data[inp] = main[inp].values # TODO dtype?
    ## get the samplerTag values if they're present, in case it's not just range
    if self.samplerTag in main:
      labels = main[self.samplerTag].values
    else:
      labels = None
    # load subfiles for output spaces
    subFiles = main['filename'].values
    # pre-build realization spots
    for out in self._outputs + self.indexes:
      data[out] = np.zeros(nSamples,dtype=object)
    for i,sub in enumerate(subFiles):
      # first time create structures
      subDat = pd.read_csv(sub)
      for out in self._outputs + self.indexes:
        data[out][i] = subDat[out].values # TODO dtype?

    self.load(data,style='dict',dims=self.getDimensions())
    # read in secondary CSVs
    # construct final data object

  def _selectiveRealization(self,rlz):
    """
      Uses "options" parameters from input to select part of the collected data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO someday needs to be implemented for when ND data is collected!  For now, use base class.
    return DataSet._selectiveRealization(self,rlz)

  def _toCSV(self,fname,start=0,**kwargs):
    """
      Writes this data objcet to CSV file (for metadata see _toCSVXML)
      @ In, fname, str, path/name to write file
      @ In, start, int, optional, starting realization to print
      @ In, kwargs, dict, optional, keywords for options
      @ Out, None
    """
    # specialized to write custom RAVEN-style history CSVs
    # TODO some overlap with DataSet implementation, but not much.
    keep = self._getRequestedElements(kwargs)
    # don't rewrite everything; if we've written some already, just append (using mode)
    if start > 0:
      # slice data starting at "start"
      sl = slice(start,None,None)
      data = self._data.isel(**{self.sampleTag:sl})
      mode = 'a'
    else:
      data = self._data
      mode = 'w'
    toDrop = list(var for var in self._allvars if var not in keep)
    data = data.drop(toDrop)
    self.raiseADebug('Printing data to CSV: "{}"'.format(fname+'.csv'))
    # specific implementation
    ## write input space CSV with pointers to history CSVs
    ### get list of input variables to keep
    ordered = list(i for i in itertools.chain(self._inputs,self._metavars) if i in keep)
    ### select input part of dataset
    inpData = data[ordered]
    ### add column for realization information, pointing to the appropriate CSV
    subFiles = np.array(list('{}_{}.csv'.format(fname,rid) for rid in data[self.sampleTag].values),dtype=object)
    ### add column to dataset
    column = self._collapseNDtoDataArray(subFiles,'filename',labels=data[self.sampleTag])
    inpData = inpData.assign(filename=column)
    ### also add column name to "ordered"
    ordered += ['filename']
    ### write CSV
    self._usePandasWriteCSV(fname,inpData,ordered,keepSampleTag = self.sampleTag in keep,mode=mode)
    ## obtain slices to write subset CSVs
    ordered = list(o for o in self.getVars('output') if o in keep)
    for i in range(len(data[self.sampleTag].values)):
      rlz = data.isel(**{self.sampleTag:i})[ordered].dropna(self.indexes[0])
      filename = subFiles[i][:-4]
      self._usePandasWriteCSV(filename,rlz,ordered,keepIndex=True)
