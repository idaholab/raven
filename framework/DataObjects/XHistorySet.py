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
  Specialized implementation of DataObject for data with single-valued inputs and outputs that share
  a single common pivot parameter in the outputs, for each realization.
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
  def profile(func):
    """
      Dummy for when profiler is not present.
      @ In, func, method, method to run
      @ Out, func, method, method to run
    """
    return func

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

  def _fromCSV(self,fileName,**kwargs):
    """
      Loads a dataset from custom RAVEN history csv.
      @ In, fileName, str, filename to load from (not including .csv or .xml)
      @ In, kwargs, dict, optional arguments
      @ Out, None
    """
    # data dict for loading data
    data = {}
    # load in metadata
    self._loadCsvMeta(fileName)
    # load in main CSV
    ## read into dataframe
    main = self._readPandasCSV(fileName+'.csv')
    nSamples = len(main.index)
    ## collect input space data
    for inp in self._inputs + self._metavars:
      data[inp] = main[inp].values
    ## get the sampleTag values if they're present, in case it's not just range
    if self.sampleTag in main:
      labels = main[self.sampleTag].values
    else:
      labels = None
    # load subfiles for output spaces
    subFiles = main['filename'].values
    # pre-build realization spots
    for out in self._outputs + self.indexes:
      data[out] = np.zeros(nSamples,dtype=object)
    # read in secondary CSVs
    for i,sub in enumerate(subFiles):
      subFile = sub
      # check if the sub has an absolute path, otherwise take it from the master file (fileName)
      if not os.path.isabs(subFile):
        subFile = os.path.join(os.path.dirname(fileName),subFile)
      # read in file
      subDat = self._readPandasCSV(subFile)
      # first time create structures
      if len(set(subDat.keys()).intersection(self.indexes)) != len(self.indexes):
        self.raiseAnError(IOError,'Importing HistorySet from .csv: the pivot parameters "'+', '.join(self.indexes)+'" have not been found in the .csv file. Check that the '
                                  'correct <pivotParameter> has been specified in the dataObject or make sure the <pivotParameter> is included in the .csv files')
      for out in self._outputs+self.indexes:
        data[out][i] = subDat[out].values

    # construct final data object
    self.load(data,style='dict',dims=self.getDimensions())

  def _identifyVariablesInCSV(self,fileName):
    """
      Gets the list of available variables from the file "fileName.csv".
      @ In, fileName, str, name of base file without extension.
      @ Out, varList, list(str), list of variables
    """
    with open(fileName+'.csv','r') as base:
      inputAvail = list(s.strip() for s in base.readline().split(','))
      # get one of the subCSVs from the first row of data in the base file
      # ASSUMES that filename is the last column.  Currently, there's no way for that not to be true.
      subFile = base.readline().split(',')[-1].strip()
      # check if abs path otherwise take the dirpath from the master file (fileName)
      if not os.path.isabs(subFile):
        subFile = os.path.join(os.path.dirname(fileName),subFile)
    with open(subFile,'r') as sub:
      outputAvail = list(s.strip() for s in sub.readline().split(','))
    return inputAvail + outputAvail

  def _selectiveRealization(self,rlz):
    """
      Uses "options" parameters from input to select part of the collected data
      @ In, rlz, dict, {var:val} format (see addRealization)
      @ Out, rlz, dict, {var:val} modified
    """
    # TODO this could be much more efficient on the parallel (finalizeCodeOutput) than on serial
    # TODO someday needs to be implemented for when ND data is collected!  For now, use base class.
    return DataSet._selectiveRealization(self,rlz)

  def _toCSV(self,fileName,start=0,**kwargs):
    """
      Writes this data objcet to CSV file (for metadata see _toCSVXML)
      @ In, fileName, str, path/name to write file
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
    self.raiseADebug('Printing data to CSV: "{}"'.format(fileName+'.csv'))
    # specific implementation
    ## write input space CSV with pointers to history CSVs
    ### get list of input variables to keep
    ordered = list(i for i in itertools.chain(self._inputs,self._metavars) if i in keep)
    ### select input part of dataset
    inpData = data[ordered]
    ### add column for realization information, pointing to the appropriate CSV
    subFiles = np.array(list('{}_{}.csv'.format(fileName,rid) for rid in data[self.sampleTag].values),dtype=object)
    ### add column to dataset
    column = self._collapseNDtoDataArray(subFiles,'filename',labels=data[self.sampleTag])
    inpData = inpData.assign(filename=column)
    ### also add column name to "ordered"
    ordered += ['filename']
    ### write CSV
    self._usePandasWriteCSV(fileName,inpData,ordered,keepSampleTag = self.sampleTag in keep,mode=mode)
    ## obtain slices to write subset CSVs
    ordered = list(o for o in self.getVars('output') if o in keep)
    for i in range(len(data[self.sampleTag].values)):
      rlz = data.isel(**{self.sampleTag:i})[ordered].dropna(self.indexes[0])
      filename = subFiles[i][:-4]
      self._usePandasWriteCSV(filename,rlz,ordered,keepIndex=True)
