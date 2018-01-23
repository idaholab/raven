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
    # TODO externalize it in the DataObject base class
    toRemove = []
    for var,val in rlz.items():
      if var in self.protectedTags:
        continue
      # only modify it if it is not already scalar
      if not mathUtils.isSingleValued(val):
        # treat inputs, outputs differently TODO this should extend to per-variable someday
        ## inputs
        if var in self._inputs:
          method,indic = self._selectInput
        # pivot variables are included here in "else"; remove them after they're used in operators
        else:
          continue
        if method in ['inputRow']:
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
        elif method in ['inputPivotValue']:
          pivotParam = self.getDimensions(var)
          assert(len(pivotParam) == 1) # TODO only handle History for now
          pivotParam = pivotParam[var][0]
          idx = (np.abs(rlz[pivotParam] - indic)).argmin()
          rlz[var] = rlz[var][idx]
        elif method == 'operator':
          if indic == 'max':
            rlz[var] = float(val.max())
          elif indic == 'min':
            rlz[var] = float(val.min())
          elif indic in ['mean','expectedValue','average']:
            rlz[var] = float(val.mean())
      # otherwise, leave it alone
    return rlz

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
    toDrop = list(var for var in self._orderedVars if var not in keep)
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
    if len(ordered):
      for i in range(len(data[self.sampleTag].values)):
        filename = subFiles[i][:-4]
        rlz = data.isel(**{self.sampleTag:i})[ordered].dropna(self.indexes[0])
        self._usePandasWriteCSV(filename,rlz,ordered,keepIndex=True)
    else:
      self.raiseAWarning('No output space variables have been requested for DataObject "{}"! No history files will be printed!'.format(self.name))
